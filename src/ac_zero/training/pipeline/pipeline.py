from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.navigation_reward import AlphaUpdater
from ac_zero.models.registry import create_trainable_model, model_from_json
from ac_zero.models.torch_utils import use_single_torch_thread
from ac_zero.system.manifests import ReproducibilityManifest
from ac_zero.system.parallel import describe_worker_pool
from ac_zero.training.checkpointing.checkpoint_name import derive_checkpoint_name
from ac_zero.training.checkpointing.checkpoint_writer import RunCheckpointer
from ac_zero.training.logging.callbacks import CallbackManager, default_training_callbacks
from ac_zero.training.logging.events import LogLevel, Verbosity
from ac_zero.training.navigation.navigation_metrics import log_navigation
from ac_zero.training.pipeline.instance_source import build_instance_source
from ac_zero.training.pipeline.pipeline_artifacts import render_plots, write_fixture_certificate
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig, run_description
from ac_zero.training.pipeline.pipeline_episodes import (
    EpisodeMetrics,
    ReplayExample,
    batch_return_and_success,
    collect_episodes,
)
from ac_zero.training.pipeline.pipeline_showcase import EpisodeShowcase
from ac_zero.training.pipeline.pipeline_summary import (
    MetricsRow,
    TrainingPipelineSummary,
    _RunDirectories,
)
from ac_zero.training.ppo.losses import PolicyValueLoss
from ac_zero.training.ppo.ppo import PPOTrainer
from ac_zero.training.ppo.replay_buffer import ReplayBuffer
from ac_zero.training.supervised.pipeline import run_supervised_pipeline


def run_training_pipeline(
    config: TrainingPipelineConfig,
    seed: int,
    callbacks: CallbackManager | None = None,
) -> TrainingPipelineSummary:
    """Run config-driven data generation, replay training, and artifact writing.

    ``agent: supervised`` hands off to the labelled-dataset pretrainer instead, which
    writes the same artifacts (run directory, metrics, plots, checkpoint bundle) so
    every caller -- the CLI, the Kaggle notebook, the scheduler -- is agnostic to
    which of the three backends produced the model.
    """
    if config.agent == "supervised":
        return run_supervised_pipeline(config, seed, callbacks)
    return _TrainingRun(config, seed, callbacks).execute()


class _TrainingRun:
    """One config-driven training run: self-play, replay training, and artifacts.

    Holds the state shared across the run — the model, replay buffer, RNG, log
    manager, and the running optimizer-step/loss/episode counters — so the step
    methods operate on `self` instead of threading that state through arguments.
    """

    def __init__(
        self, config: TrainingPipelineConfig, seed: int, callbacks: CallbackManager | None
    ) -> None:
        config.validate()
        use_single_torch_thread()
        self.config = config
        self.seed = seed
        self.dirs = _RunDirectories.create(config.run_directory)
        self.manager = callbacks or default_training_callbacks(
            self.dirs.run, verbosity=config.verbosity
        )
        self.replay = ReplayBuffer[ReplayExample](config.replay_capacity)
        self.encoder = StateEncoder(config.max_relator_tokens)
        # Built once up front so the run can log what it trains on, and so the
        # dataset sidecar is compiled here rather than raced for by every worker.
        self._instance_source = build_instance_source(config)
        self.rng = random.Random(seed)
        self.model = create_trainable_model(
            config.model,
            seed=seed,
            device=config.device,
            # The sizes are architecture hyperparameters, never a `device` override.
            **cast(dict[str, Any], config.model_config),
        )
        self._warm_start_payload: dict[str, Any] | None = None
        self.warm_started_from = self._warm_start()
        self.checkpointer = RunCheckpointer(
            self.dirs.run,
            config=config,
            checkpoint_name=config.checkpoint_name or derive_checkpoint_name(config),
            run_id=f"{int(time.time())}-{seed}",
            warm_started_from=self.warm_started_from,
            seed=seed,
        )
        self.metrics_rows: list[MetricsRow] = []
        self.optimizer_step = 0
        # Iterations actually run: the wall-clock budget can end the loop early,
        # so this -- not `config.iterations` -- is what the summary reports.
        self.completed_iterations = 0
        self.deadline = (
            None if config.time_limit_s is None else time.monotonic() + config.time_limit_s
        )
        self.final_loss = PolicyValueLoss(0.0, 0.0, 0.0)
        self.total_episodes = 0
        # Best model is picked by an EMA of self-play mean return, smoothing the
        # per-iteration noise a raw pick would lock onto (see save-checkpoint).
        self.return_ema: float | None = None
        self.last_mean_return = 0.0
        self.last_success_rate = 0.0
        # Navigation selects the best model by success rate, not shaped return
        # (which a long, never-solving shaping path can inflate).
        self.success_ema: float | None = None
        self.alpha_updater = (
            AlphaUpdater(config.reward_config) if config.reward_mode == "navigation" else None
        )
        # Continue the adaptive shaping weight from the warm-start checkpoint so a
        # resumed lineage does not snap alpha back to its config initial
        # mid-training.
        self._restore_learning_state()
        self.ppo = (
            PPOTrainer(config, self.encoder, self._instance_source)
            if config.agent == "ppo"
            else None
        )
        # Prints one played-out episode every few hours (see `_checkpoint`). Off at
        # `quiet`, whose contract is milestones and diagnostics only.
        self.showcase = (
            EpisodeShowcase(
                config,
                self.encoder,
                self._instance_source,
                every_hours=config.showcase_every_hours,
            )
            if config.showcase_every_hours > 0 and config.verbosity >= Verbosity.SUMMARY
            else None
        )

    def _warm_start(self) -> str | None:
        """Initialize the model from a prior checkpoint when configured.

        Returns a short provenance string (the source path) or ``None`` when the
        run trains from scratch. The checkpoint's saved architecture must match
        the configured model so its weights load into the fresh network.
        """
        if not self.config.warm_start:
            print("[warm-start] no checkpoint configured; training a fresh model")
            return None
        data = json.loads(Path(self.config.warm_start).read_text(encoding="utf-8"))
        # Kept so the adaptive learning state can be restored once the alpha
        # updater exists (see _restore_learning_state).
        self._warm_start_payload = data
        self.model = model_from_json(data.get("model_state", data))
        iteration, metric = data.get("iteration"), data.get("checkpoint_metric")
        provenance = ""
        if iteration is not None:
            provenance += f" (iteration {iteration}"
            provenance += f", metric {metric:.4f})" if isinstance(metric, (int, float)) else ")"
        print(f"[warm-start] initialized model from {self.config.warm_start}{provenance}")
        return self.config.warm_start

    def _restore_learning_state(self) -> None:
        """Continue alpha from the warm-start checkpoint when it carried one.

        Reads the ``learning_state`` block the checkpointer writes, but only when
        both the snapshot and the live run run the navigation reward. A checkpoint
        from before this field existed leaves the fresh updater at its config
        initial -- the pre-existing behavior.
        """
        payload = self._warm_start_payload
        if not payload:
            return
        state = payload.get("learning_state") or {}
        alpha_state = state.get("alpha_updater")
        if alpha_state is not None and self.alpha_updater is not None:
            self.alpha_updater.load_state_dict(alpha_state)
            print(f"[warm-start] resumed shaping alpha at {self.alpha_updater.alpha:.4f}")

    def _learning_state(self) -> dict[str, Any]:
        """The adaptive across-episode state to persist in the next checkpoint.

        Empty for a non-navigation run, which carries no alpha to advance, so
        :meth:`_restore_learning_state` restores nothing from it.
        """
        if self.alpha_updater is None:
            return {}
        return {"alpha_updater": self.alpha_updater.state_dict()}

    def _record_iteration_stats(self, mean_return: float, success_rate: float) -> None:
        """Track the latest self-play stats and the EMAs used to pick the best model."""
        self.last_mean_return = mean_return
        self.last_success_rate = success_rate
        self.return_ema = (
            mean_return if self.return_ema is None else 0.7 * self.return_ema + 0.3 * mean_return
        )
        self.success_ema = (
            success_rate
            if self.success_ema is None
            else 0.7 * self.success_ema + 0.3 * success_rate
        )

    def _current_alpha(self) -> float | None:
        """The navigation shaping weight this iteration's episodes run at."""
        return None if self.alpha_updater is None else self.alpha_updater.alpha

    def _dynamic_params(self) -> dict[str, float | int]:
        """The run's dynamic learning parameters, for the iteration line and metrics rows.

        Just the navigation reward's shaping weight ``alpha`` -- the run's one
        adaptive knob -- reported every iteration so its live value is visible on
        the terminal, and folded into every metrics row so the rendered plots show
        how it moved over the run. Empty off navigation, where there is no alpha to
        advance and the alpha figure is therefore skipped.
        """
        alpha = self._current_alpha()
        return {} if alpha is None else {"alpha": alpha}

    def _checkpoint_metric(self) -> float | None:
        """Best-model metric: the success EMA on navigation, the return EMA elsewhere."""
        return self.success_ema if self.alpha_updater is not None else self.return_ema

    def _finalize_navigation(self, iteration: int, episodes: list[EpisodeMetrics]) -> None:
        """Advance alpha from the batch and log per-episode + aggregate nav metrics."""
        if self.alpha_updater is None:
            return
        log_navigation(
            self.manager,
            iteration * 100 + 50,
            iteration,
            self.alpha_updater,
            episodes,
            self._progress_level(iteration),
        )

    def execute(self) -> TrainingPipelineSummary:
        try:
            description = run_description(self.config, self.seed, self.model.architecture)
            self.manager.emit(0, "start", "starting training pipeline", description)
            self.manager.emit(
                1, "dataset", "self-play instance source", dict(self._instance_source.describe())
            )
            _, worker_message, worker_metrics = describe_worker_pool(self.config.workers)
            self.manager.emit(2, "self_play", worker_message, worker_metrics)

            for iteration in range(1, self.config.iterations + 1):
                self._train_iteration(iteration)
                self.completed_iterations = iteration
                if self._budget_spent():
                    self.manager.emit(
                        self._late_event_id(2),
                        "budget",
                        "wall-clock budget spent; stopping at iteration boundary",
                        {"iteration": iteration, "optimizer_step": self.optimizer_step},
                    )
                    break

            checkpoint_path = self._save_checkpoint(self.completed_iterations)
            restored = self.checkpointer.load_latest()
            checkpoint_restored = restored["optimizer_state"]["step"] == self.optimizer_step
            self._write_metrics()
            plot_paths = self._render_plots()
            ReproducibilityManifest.create(
                "training", self.seed, {"pipeline": asdict(self.config)}
            ).write(self.dirs.run / "manifest.json")

            certificate_path, certificate_verified = self._write_certificate()
            self.manager.emit(
                self._late_event_id(4),
                "certificate",
                "solved fixture and verified certificate",
                {
                    "certificate_verified": certificate_verified,
                    "optimizer_updates": self.optimizer_step,
                },
            )

            summary = self._build_summary(
                str(checkpoint_path),
                str(certificate_path),
                checkpoint_restored,
                certificate_verified,
                plot_paths,
            )
            self.manager.emit(
                self._late_event_id(5),
                "completed",
                "training pipeline completed",
                {
                    "optimizer_updates": summary.optimizer_updates,
                    "replay_size": summary.replay_size,
                    "total_loss": summary.final_total_loss,
                },
            )
            (self.dirs.artifacts / "training_summary.json").write_text(
                json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return summary
        except Exception as exc:
            self.manager.emit_error("error", "training pipeline failed", exc)
            raise
        finally:
            self.manager.close()

    def _progress_level(self, count: int) -> LogLevel:
        """INFO on the first and every ``progress_every``-th recurring event, else DEBUG.

        Throttles the terminal progress log on long runs: the DEBUG steps in
        between are still recorded by the JSONL event log and the ASCII graphs
        (which process every event); only the INFO terminal sink is quieted.
        """
        if count == 1 or count % self.config.progress_every == 0:
            return LogLevel.INFO
        return LogLevel.DEBUG

    def _train_iteration(self, iteration: int) -> None:
        """Run one iteration's self-play, then its optimizer updates and checkpoint."""
        if self.ppo is not None:
            self._train_iteration_ppo(iteration)
            return
        episodes = self._collect_iteration(iteration)
        self.total_episodes += len(episodes)
        mean_return, success_rate = batch_return_and_success(episodes)
        self._record_iteration_stats(mean_return, success_rate)
        self._finalize_navigation(iteration, episodes)
        self.manager.emit(
            iteration * 100,
            "self_play",
            "collected search-guided replay",
            {
                "iteration": iteration,
                "episodes": self.total_episodes,
                "mean_return": mean_return,
                "success_rate": success_rate,
                "replay_size": len(self.replay),
                **self._dynamic_params(),
            },
            level=self._progress_level(iteration),
        )
        self._run_optimizer_updates(iteration, mean_return, success_rate)
        self._checkpoint(iteration)

    def _train_iteration_ppo(self, iteration: int) -> None:
        """Run one PPO iteration: on-policy rollouts, clipped updates, checkpoint."""
        assert self.ppo is not None
        result = self.ppo.run_iteration(
            self.model, self.seed, iteration, self.rng, self._current_alpha()
        )
        self.total_episodes += len(result.episodes)
        mean_return, success_rate = batch_return_and_success(result.episodes)
        self._record_iteration_stats(mean_return, success_rate)
        self._finalize_navigation(iteration, result.episodes)
        self.manager.emit(
            iteration * 100,
            "self_play",
            "collected on-policy PPO rollouts",
            {
                "iteration": iteration,
                "episodes": self.total_episodes,
                "mean_return": mean_return,
                "success_rate": success_rate,
                "examples": result.example_count,
                **self._dynamic_params(),
            },
            level=self._progress_level(iteration),
        )
        for stats in result.updates:
            self.optimizer_step += 1
            self.final_loss = PolicyValueLoss(stats.policy_loss, stats.value_loss, stats.total_loss)
            row: MetricsRow = {
                "iteration": iteration,
                "optimizer_step": self.optimizer_step,
                "examples": result.example_count,
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "total_loss": stats.total_loss,
                "entropy": stats.entropy,
                "clip_fraction": stats.clip_fraction,
                "approx_kl": stats.approx_kl,
                "mean_return": mean_return,
                "success_rate": success_rate,
                **self._dynamic_params(),
            }
            self.metrics_rows.append(row)
            self.manager.emit(
                iteration * 100 + self.optimizer_step,
                "optimizer",
                "updated policy via PPO",
                row,
                level=self._progress_level(self.optimizer_step),
            )
        self._checkpoint(iteration)

    def _checkpoint(self, iteration: int) -> None:
        """Save this iteration's checkpoint, then show an episode when one is due.

        The checkpoint event is what the HF uploader listens on, so the showcase
        runs on the same boundary and at the same default cadence: a played-out
        episode lands in the log next to every push (see `EpisodeShowcase`). Its
        seed continues the iteration's own episode-seed block, so the problem it
        draws is a fresh one and no training episode's seed is reused.
        """
        if iteration % self.config.checkpoint_every != 0:
            return
        self._save_checkpoint(iteration)
        self.manager.emit(
            iteration * 100 + self.optimizer_step + 1,
            "checkpoint",
            "saved training checkpoint",
            {"iteration": iteration, "optimizer_step": self.optimizer_step},
        )
        if self.showcase is None or not self.showcase.due():
            return
        self.showcase.show(
            self.manager,
            self.model,
            event_id=iteration * 100 + self.optimizer_step + 2,
            iteration=iteration,
            seed=self.seed + iteration * 10_000 + self.config.episodes_per_iteration,
            alpha=self._current_alpha(),
        )

    def _collect_iteration(self, iteration: int) -> list[EpisodeMetrics]:
        """Collect this iteration's self-play episodes into the replay buffer."""
        episodes: list[EpisodeMetrics] = []
        collected = collect_episodes(
            self.config,
            self.encoder,
            self.model,
            self.seed,
            iteration,
            self._instance_source,
            self._current_alpha(),
        )
        for examples, episode_metrics in collected:
            self.replay.extend(examples)
            episodes.append(episode_metrics)
        return episodes

    def _run_optimizer_updates(
        self, iteration: int, mean_return: float, success_rate: float
    ) -> None:
        """Sample replay batches and update the model, logging one row per step."""
        self.final_loss = PolicyValueLoss(0.0, 0.0, 0.0)
        for _ in range(self.config.optimizer_updates):
            batch = self.replay.sample(self.config.batch_size, self.rng)
            self.final_loss = self.model.train_batch(
                batch,
                learning_rate=self.config.learning_rate,
                value_loss_weight=self.config.value_loss_weight,
                grad_clip=self.config.grad_clip,
            )
            self.optimizer_step += 1
            row: MetricsRow = {
                "iteration": iteration,
                "optimizer_step": self.optimizer_step,
                "batch_size": len(batch),
                "replay_size": len(self.replay),
                "policy_loss": self.final_loss.policy_loss,
                "value_loss": self.final_loss.value_loss,
                "total_loss": self.final_loss.total_loss,
                "mean_return": mean_return,
                "success_rate": success_rate,
                **self._dynamic_params(),
            }
            self.metrics_rows.append(row)
            self.manager.emit(
                iteration * 100 + self.optimizer_step,
                "optimizer",
                "updated policy-value model",
                row,
                level=self._progress_level(self.optimizer_step),
            )

    def _save_checkpoint(self, iteration: int) -> Path:
        # Keep both sinks current: legacy latest.json plus the HF-shaped bundle
        # (latest always, best when the return EMA improves, metrics + provenance).
        return self.checkpointer.save(
            model=self.model,
            iteration=iteration,
            optimizer_step=self.optimizer_step,
            loss=self.final_loss,
            replay_size=len(self.replay),
            metric=self._checkpoint_metric(),
            mean_return=self.last_mean_return,
            success_rate=self.last_success_rate,
            metrics_rows=self.metrics_rows,
            learning_state=self._learning_state(),
        )

    def _write_metrics(self) -> None:
        (self.dirs.run / "metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.metrics_rows),
            encoding="utf-8",
        )

    def _budget_spent(self) -> bool:
        """Whether the optional wall-clock budget has run out."""
        return self.deadline is not None and time.monotonic() >= self.deadline

    def _write_certificate(self) -> tuple[Path, bool]:
        certificate_path = self.dirs.certificates / "example.json"
        verified = write_fixture_certificate(self.config, self.seed, certificate_path)
        return certificate_path, verified

    def _render_plots(self) -> tuple[str, ...]:
        return render_plots(
            self.metrics_rows, self.dirs.artifacts, self.manager, self._late_event_id(3)
        )

    def _build_summary(
        self,
        checkpoint_path: str,
        certificate_path: str,
        checkpoint_restored: bool,
        certificate_verified: bool,
        plot_paths: tuple[str, ...],
    ) -> TrainingPipelineSummary:
        return TrainingPipelineSummary(
            run_directory=str(self.dirs.run),
            checkpoint_path=checkpoint_path,
            certificate_path=certificate_path,
            model_name=self.model.architecture,
            checkpoint_name=self.checkpointer.checkpoint_name,
            checkpoint_bundle_dir=str(self.checkpointer.bundle.directory),
            run_id=self.checkpointer.run_id,
            best_return=self.checkpointer.best_metric,
            iterations=self.completed_iterations,
            episodes=self.total_episodes,
            replay_size=len(self.replay),
            optimizer_updates=self.optimizer_step,
            final_total_loss=self.final_loss.total_loss,
            checkpoint_restored=checkpoint_restored,
            certificate_verified=certificate_verified,
            event_log_path=str(self.dirs.logs / "training_events.jsonl"),
            progress_log_path=str(self.dirs.logs / "progress.log"),
            live_graph_path=str(self.dirs.artifacts / "live_graphs.txt"),
            final_graph_path=str(self.dirs.artifacts / "final_graphs.txt"),
            plot_paths=plot_paths,
        )

    def _late_event_id(self, offset: int) -> int:
        """Monotonic event id for the post-loop artifact/certificate/completion events."""
        return self.completed_iterations * 100 + self.optimizer_step + offset
