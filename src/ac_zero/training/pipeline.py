from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ac_zero.agents.greedy import GreedySolver
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment
from ac_zero.models.registry import create_trainable_model
from ac_zero.system.manifests import ReproducibilityManifest
from ac_zero.system.parallel import describe_worker_pool
from ac_zero.training.callbacks import CallbackManager, default_training_callbacks
from ac_zero.training.checkpointing import CheckpointManager
from ac_zero.training.events import LogLevel
from ac_zero.training.losses import PolicyValueLoss
from ac_zero.training.pipeline_config import TrainingPipelineConfig
from ac_zero.training.pipeline_episodes import (
    EpisodeMetrics,
    ReplayExample,
    build_env_config,
    collect_episodes,
)
from ac_zero.training.plots import PlotsUnavailable, render_training_plots
from ac_zero.training.ppo import PPOTrainer
from ac_zero.training.replay_buffer import ReplayBuffer

MetricsRow = dict[str, float | int | bool | str]


@dataclass(frozen=True, slots=True)
class TrainingPipelineSummary:
    """High-level result of the config-driven training pipeline."""

    run_directory: str
    checkpoint_path: str
    certificate_path: str
    model_name: str
    iterations: int
    episodes: int
    replay_size: int
    optimizer_updates: int
    final_total_loss: float
    checkpoint_restored: bool
    certificate_verified: bool
    event_log_path: str
    progress_log_path: str
    live_graph_path: str
    final_graph_path: str
    plot_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RunDirectories:
    """The output subdirectories of a single training run."""

    run: Path
    checkpoints: Path
    certificates: Path
    artifacts: Path
    logs: Path

    @classmethod
    def create(cls, run_directory: str) -> _RunDirectories:
        run = Path(run_directory)
        dirs = cls(
            run=run,
            checkpoints=run / "checkpoints",
            certificates=run / "certificates",
            artifacts=run / "artifacts",
            logs=run / "logs",
        )
        for directory in (dirs.checkpoints, dirs.certificates, dirs.artifacts, dirs.logs):
            directory.mkdir(parents=True, exist_ok=True)
        return dirs


def run_training_pipeline(
    config: TrainingPipelineConfig,
    seed: int,
    callbacks: CallbackManager | None = None,
) -> TrainingPipelineSummary:
    """Run config-driven data generation, replay training, and artifact writing."""
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
        self.config = config
        self.seed = seed
        self.dirs = _RunDirectories.create(config.run_directory)
        self.manager = callbacks or default_training_callbacks(self.dirs.run)
        self.replay = ReplayBuffer[ReplayExample](config.replay_capacity)
        self.encoder = StateEncoder(config.max_word_length)
        self.rng = random.Random(seed)
        self.model = create_trainable_model(config.model, seed=seed)
        self.checkpoints = CheckpointManager(self.dirs.checkpoints)
        self.metrics_rows: list[MetricsRow] = []
        self.optimizer_step = 0
        self.final_loss = PolicyValueLoss(0.0, 0.0, 0.0)
        self.total_episodes = 0
        self.ppo = PPOTrainer(config, self.encoder) if config.agent == "ppo" else None

    def execute(self) -> TrainingPipelineSummary:
        try:
            self.manager.emit(0, "start", "starting training pipeline", self._run_description())
            _, worker_message, worker_metrics = describe_worker_pool(self.config.workers)
            self.manager.emit(1, "self_play", worker_message, worker_metrics)

            for iteration in range(1, self.config.iterations + 1):
                self._train_iteration(iteration)

            checkpoint_path = self._save_checkpoint(self.config.iterations)
            restored = self.checkpoints.load_json("latest")
            checkpoint_restored = restored["optimizer_state"]["step"] == self.optimizer_step
            self._write_metrics()
            plot_paths = self._render_plots()
            ReproducibilityManifest.create(
                "training", self.seed, {"pipeline": asdict(self.config)}
            ).write(self.dirs.run / "manifest.json")

            certificate_path, certificate_verified = self._write_certificate()
            self.manager.emit(
                self._late_event_id(2),
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
                self._late_event_id(3),
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

    def _train_iteration(self, iteration: int) -> None:
        """Run one iteration's self-play, then its optimizer updates and checkpoint."""
        if self.ppo is not None:
            self._train_iteration_ppo(iteration)
            return
        episodes = self._collect_iteration(iteration)
        self.total_episodes += len(episodes)
        mean_return, success_rate = _episode_stats(episodes)
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
            },
        )
        self._run_optimizer_updates(iteration, mean_return, success_rate)
        if iteration % self.config.checkpoint_every == 0:
            self._save_checkpoint(iteration)
            self.manager.emit(
                iteration * 100 + self.optimizer_step + 1,
                "checkpoint",
                "saved training checkpoint",
                {"iteration": iteration, "optimizer_step": self.optimizer_step},
            )

    def _train_iteration_ppo(self, iteration: int) -> None:
        """Run one PPO iteration: on-policy rollouts, clipped updates, checkpoint."""
        assert self.ppo is not None
        result = self.ppo.run_iteration(self.model, self.seed, iteration, self.rng)
        self.total_episodes += len(result.episodes)
        mean_return, success_rate = _episode_stats(result.episodes)
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
            },
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
            }
            self.metrics_rows.append(row)
            self.manager.emit(
                iteration * 100 + self.optimizer_step, "optimizer", "updated policy via PPO", row
            )
        if iteration % self.config.checkpoint_every == 0:
            self._save_checkpoint(iteration)
            self.manager.emit(
                iteration * 100 + self.optimizer_step + 1,
                "checkpoint",
                "saved training checkpoint",
                {"iteration": iteration, "optimizer_step": self.optimizer_step},
            )

    def _collect_iteration(self, iteration: int) -> list[EpisodeMetrics]:
        """Collect this iteration's self-play episodes into the replay buffer."""
        episodes: list[EpisodeMetrics] = []
        collected = collect_episodes(self.config, self.encoder, self.model, self.seed, iteration)
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
            }
            self.metrics_rows.append(row)
            self.manager.emit(
                iteration * 100 + self.optimizer_step,
                "optimizer",
                "updated policy-value model",
                row,
            )

    def _save_checkpoint(self, iteration: int) -> Path:
        return self.checkpoints.save_json(
            "latest",
            {
                "schema_version": "aczero-training-checkpoint-v1",
                "seed": self.seed,
                "iteration": iteration,
                "config": asdict(self.config),
                "model_state": self.model.to_json(),
                "optimizer_state": {
                    "step": self.optimizer_step,
                    "learning_rate": self.config.learning_rate,
                },
                "replay_size": len(self.replay),
                "loss": asdict(self.final_loss),
            },
        )

    def _write_metrics(self) -> None:
        (self.dirs.run / "metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.metrics_rows),
            encoding="utf-8",
        )

    def _write_certificate(self) -> tuple[Path, bool]:
        """Solve a small fixture with the greedy solver and check its certificate verifies."""
        certificate_path = self.dirs.certificates / "example.json"
        # The "descent" objective (shorten by >=1) is not an AC trivialization, so a
        # solve-to-standard certificate does not apply; skip it for those runs.
        if self.config.reward_mode == "descent":
            return certificate_path, False
        fixture = generate_solvable(self.config.rank, min(self.config.scramble_depth, 2), self.seed)
        solve_env = ACEnvironment(fixture.presentation, build_env_config(self.config))
        result = GreedySolver().solve(
            solve_env,
            certificate_path=certificate_path,
            experiment_id="training",
            seed=self.seed,
        )
        verified = bool(result.success and CertificateVerifier().verify_path(certificate_path).ok)
        return certificate_path, verified

    def _render_plots(self) -> tuple[str, ...]:
        """Render training-progress plots, reporting the outcome through the log.

        Returns the written PNG paths. If matplotlib is not installed the run still
        succeeds — a warning is logged pointing at the always-available ASCII graphs
        and an empty tuple is returned.
        """
        try:
            paths = render_training_plots(self.metrics_rows, self.dirs.artifacts)
        except PlotsUnavailable:
            self.manager.emit(
                self.optimizer_step + 10,
                "plots",
                "matplotlib not installed; skipping image plots (ASCII graphs still written)",
                {"matplotlib": False},
                level=LogLevel.WARNING,
            )
            return ()
        if paths:
            self.manager.emit(
                self.optimizer_step + 10,
                "plots",
                "rendered training-progress plots",
                {"count": len(paths), "directory": str(self.dirs.artifacts)},
            )
        return tuple(str(path) for path in paths)

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
            iterations=self.config.iterations,
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
        return self.config.iterations * 100 + self.optimizer_step + offset

    def _run_description(self) -> MetricsRow:
        """Full description of the run: every parameter that shapes the trained model,
        so the run is reproducible from its opening log entry alone."""
        config = self.config
        return {
            "seed": self.seed,
            "rank": config.rank,
            "agent": config.agent,
            "requested_model": config.model,
            "training_model": self.model.architecture,
            "scramble_depth": config.scramble_depth,
            "iterations": config.iterations,
            "episodes_per_iteration": config.episodes_per_iteration,
            "mcts_simulations": config.mcts_simulations,
            "c_puct": config.c_puct,
            "optimizer_updates": config.optimizer_updates,
            "batch_size": config.batch_size,
            "replay_capacity": config.replay_capacity,
            "learning_rate": config.learning_rate,
            "value_loss_weight": config.value_loss_weight,
            "run_directory": config.run_directory,
        }


def _episode_stats(episodes: list[EpisodeMetrics]) -> tuple[float, float]:
    mean_return = float(np.mean([episode.normalized_return for episode in episodes]))
    success_rate = float(np.mean([1.0 if episode.success else 0.0 for episode in episodes]))
    return mean_return, success_rate
