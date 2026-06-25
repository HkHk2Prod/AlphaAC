from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ac_zero.agents.greedy import GreedySolver
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.base import PolicyValueModel
from ac_zero.models.registry import create_trainable_model, model_from_json
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.search.puct import PUCTMCTS, PUCTConfig
from ac_zero.system.manifests import ReproducibilityManifest
from ac_zero.system.parallel import parallel_map, resolve_worker_count
from ac_zero.training.callbacks import CallbackManager, default_training_callbacks
from ac_zero.training.checkpointing import CheckpointManager
from ac_zero.training.losses import (
    PolicyValueLoss,
    return_to_go,
    visit_count_policy,
)
from ac_zero.training.pipeline_config import TrainingPipelineConfig
from ac_zero.training.replay_buffer import ReplayBuffer


@dataclass(frozen=True, slots=True)
class ReplayExample:
    """One policy/value training target collected from a search state."""

    encoding: PaddedEncoding
    legal_mask: tuple[bool, ...]
    policy_target: NDArray[np.float64]
    value_target: float
    reward: float
    action: int


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Small aggregate metrics from one generated episode."""

    total_return: float
    normalized_return: float
    success: bool
    moves: int


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


def run_training_pipeline(
    config: TrainingPipelineConfig,
    seed: int,
    callbacks: CallbackManager | None = None,
) -> TrainingPipelineSummary:
    """Run config-driven data generation, replay training, and artifact writing."""
    config.validate()
    run = Path(config.run_directory)
    checkpoint_dir = run / "checkpoints"
    certificate_dir = run / "certificates"
    artifact_dir = run / "artifacts"
    log_dir = run / "logs"
    for directory in (checkpoint_dir, certificate_dir, artifact_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manager = callbacks or default_training_callbacks(run)
    replay = ReplayBuffer[ReplayExample](config.replay_capacity)
    encoder = StateEncoder(config.max_word_length)
    rng = random.Random(seed)
    model = create_trainable_model(config.model, seed=seed)
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    metrics_rows: list[dict[str, float | int | bool | str]] = []
    optimizer_step = 0
    final_loss = PolicyValueLoss(0.0, 0.0, 0.0)
    total_episodes = 0

    try:
        manager.emit(
            0,
            "start",
            "starting training pipeline",
            {
                "seed": seed,
                "rank": config.rank,
                "requested_model": config.model,
                "training_model": model.architecture,
                "mcts_simulations": config.mcts_simulations,
            },
        )
        for iteration in range(1, config.iterations + 1):
            episodes = []
            collected = _collect_episodes(config, encoder, model, seed, iteration)
            for examples, episode_metrics in collected:
                replay.extend(examples)
                episodes.append(episode_metrics)
            total_episodes += len(episodes)
            mean_return = float(np.mean([episode.normalized_return for episode in episodes]))
            success_rate = float(np.mean([1.0 if episode.success else 0.0 for episode in episodes]))
            manager.emit(
                iteration * 100,
                "self_play",
                "collected search-guided replay",
                {
                    "iteration": iteration,
                    "episodes": total_episodes,
                    "mean_return": mean_return,
                    "success_rate": success_rate,
                    "replay_size": len(replay),
                },
            )

            for _ in range(config.optimizer_updates):
                batch = replay.sample(config.batch_size, rng)
                final_loss = model.train_batch(
                    batch,
                    learning_rate=config.learning_rate,
                    value_loss_weight=config.value_loss_weight,
                )
                optimizer_step += 1
                row: dict[str, float | int | bool | str] = {
                    "iteration": iteration,
                    "optimizer_step": optimizer_step,
                    "batch_size": len(batch),
                    "replay_size": len(replay),
                    "policy_loss": final_loss.policy_loss,
                    "value_loss": final_loss.value_loss,
                    "total_loss": final_loss.total_loss,
                    "mean_return": mean_return,
                    "success_rate": success_rate,
                }
                metrics_rows.append(row)
                manager.emit(
                    iteration * 100 + optimizer_step,
                    "optimizer",
                    "updated policy-value model",
                    row,
                )

            if iteration % config.checkpoint_every == 0:
                checkpoint_path = _save_checkpoint(
                    checkpoint_manager,
                    config,
                    seed,
                    model,
                    optimizer_step,
                    len(replay),
                    iteration,
                    final_loss,
                )
                manager.emit(
                    iteration * 100 + optimizer_step + 1,
                    "checkpoint",
                    "saved training checkpoint",
                    {"iteration": iteration, "optimizer_step": optimizer_step},
                )

        checkpoint_path = _save_checkpoint(
            checkpoint_manager,
            config,
            seed,
            model,
            optimizer_step,
            len(replay),
            config.iterations,
            final_loss,
        )
        restored = checkpoint_manager.load_json("latest")
        checkpoint_restored = restored["optimizer_state"]["step"] == optimizer_step
        metrics_path = run / "metrics.jsonl"
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics_rows),
            encoding="utf-8",
        )
        ReproducibilityManifest.create("training", seed, {"pipeline": asdict(config)}).write(
            run / "manifest.json"
        )

        certificate_path = certificate_dir / "example.json"
        fixture = generate_solvable(config.rank, min(config.scramble_depth, 2), seed)
        solve_env = ACEnvironment(fixture.presentation, _env_config(config))
        result = GreedySolver().solve(
            solve_env,
            certificate_path=certificate_path,
            experiment_id="training",
            seed=seed,
        )
        certificate_verified = bool(
            result.success and CertificateVerifier().verify_path(certificate_path).ok
        )
        manager.emit(
            config.iterations * 100 + optimizer_step + 2,
            "certificate",
            "solved fixture and verified certificate",
            {
                "certificate_verified": certificate_verified,
                "optimizer_updates": optimizer_step,
            },
        )

        summary = TrainingPipelineSummary(
            run_directory=str(run),
            checkpoint_path=str(checkpoint_path),
            certificate_path=str(certificate_path),
            model_name=model.architecture,
            iterations=config.iterations,
            episodes=total_episodes,
            replay_size=len(replay),
            optimizer_updates=optimizer_step,
            final_total_loss=final_loss.total_loss,
            checkpoint_restored=checkpoint_restored,
            certificate_verified=certificate_verified,
            event_log_path=str(log_dir / "training_events.jsonl"),
            progress_log_path=str(log_dir / "progress.log"),
            live_graph_path=str(artifact_dir / "live_graphs.txt"),
            final_graph_path=str(artifact_dir / "final_graphs.txt"),
        )
        manager.emit(
            config.iterations * 100 + optimizer_step + 3,
            "completed",
            "training pipeline completed",
            {
                "optimizer_updates": summary.optimizer_updates,
                "replay_size": summary.replay_size,
                "total_loss": summary.final_total_loss,
            },
        )
        (artifact_dir / "training_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception as exc:
        manager.emit_error("error", "training pipeline failed", exc)
        raise
    finally:
        manager.close()


def _collect_episodes(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    model: TrainablePolicyValueModel,
    seed: int,
    iteration: int,
) -> list[tuple[list[ReplayExample], EpisodeMetrics]]:
    """Collect one iteration's self-play episodes, fanning out across processes.

    Each episode is fully determined by its seed and the current model, so the
    episodes run independently and are reassembled in order; the result is
    identical whether one or many worker processes are used.
    """
    episode_seeds = [
        seed + iteration * 10_000 + index for index in range(config.episodes_per_iteration)
    ]
    if resolve_worker_count(config.workers) <= 1:
        return [
            _collect_episode(config, encoder, episode_seed, model) for episode_seed in episode_seeds
        ]
    return parallel_map(
        _episode_worker,
        episode_seeds,
        workers=config.workers,
        initializer=_init_episode_worker,
        initargs=(config, model.to_json()),
    )


# Per-worker state, populated once by the process-pool initializer so the model
# is rebuilt from its serialized weights a single time per worker rather than
# pickled with every episode task.
_WORKER_CONFIG: TrainingPipelineConfig | None = None
_WORKER_ENCODER: StateEncoder | None = None
_WORKER_MODEL: PolicyValueModel | None = None


def _init_episode_worker(config: TrainingPipelineConfig, model_state: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_ENCODER, _WORKER_MODEL
    _WORKER_CONFIG = config
    _WORKER_ENCODER = StateEncoder(config.max_word_length)
    _WORKER_MODEL = model_from_json(model_state)


def _episode_worker(episode_seed: int) -> tuple[list[ReplayExample], EpisodeMetrics]:
    assert _WORKER_CONFIG is not None and _WORKER_ENCODER is not None and _WORKER_MODEL is not None
    return _collect_episode(_WORKER_CONFIG, _WORKER_ENCODER, episode_seed, _WORKER_MODEL)


def _collect_episode(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    episode_seed: int,
    model: PolicyValueModel,
) -> tuple[list[ReplayExample], EpisodeMetrics]:
    # A per-episode RNG seeded from the episode seed keeps action sampling
    # independent of execution order, so episodes can run in parallel and still
    # reproduce exactly.
    rng = random.Random(episode_seed)
    instance = generate_solvable(config.rank, config.scramble_depth, episode_seed)
    env = ACEnvironment(instance.presentation, _env_config(config))
    mcts = PUCTMCTS(
        model, encoder, PUCTConfig(simulations=config.mcts_simulations, c_puct=config.c_puct)
    )
    pending: list[tuple[PaddedEncoding, tuple[bool, ...], NDArray[np.float64], int, float]] = []
    rewards: list[float] = []
    terminated = False
    truncated = False
    while not terminated and not truncated:
        encoding = encoder.encode(env.state)
        legal_mask = env.legal_action_mask()
        if not any(legal_mask):
            break
        stats = mcts.search(env)
        policy_target = visit_count_policy(stats.visit_counts, legal_mask)
        action = _sample_action(policy_target, rng)
        _, reward, terminated, truncated, _ = env.step(action)
        normalized_reward = reward / max(1.0, float(env.initial.total_length))
        pending.append((encoding, legal_mask, policy_target, action, normalized_reward))
        rewards.append(normalized_reward)
    returns = return_to_go(rewards)
    examples = [
        ReplayExample(
            encoding=encoding,
            legal_mask=legal_mask,
            policy_target=policy_target,
            value_target=returns[idx],
            reward=reward,
            action=action,
        )
        for idx, (encoding, legal_mask, policy_target, action, reward) in enumerate(pending)
    ]
    total_return = float(sum(rewards))
    return examples, EpisodeMetrics(
        total_return=total_return,
        normalized_return=total_return,
        success=terminated,
        moves=len(pending),
    )


def _save_checkpoint(
    checkpoint_manager: CheckpointManager,
    config: TrainingPipelineConfig,
    seed: int,
    model: TrainablePolicyValueModel,
    optimizer_step: int,
    replay_size: int,
    iteration: int,
    loss: PolicyValueLoss,
) -> Path:
    return checkpoint_manager.save_json(
        "latest",
        {
            "schema_version": "aczero-training-checkpoint-v1",
            "seed": seed,
            "iteration": iteration,
            "config": asdict(config),
            "model_state": model.to_json(),
            "optimizer_state": {
                "step": optimizer_step,
                "learning_rate": config.learning_rate,
            },
            "replay_size": replay_size,
            "loss": asdict(loss),
        },
    )


def _sample_action(policy: NDArray[np.float64], rng: random.Random) -> int:
    total = float(np.sum(policy))
    if total <= 0.0:
        raise RuntimeError("cannot sample from an empty policy")
    threshold = rng.random()
    cumulative = 0.0
    for idx, probability in enumerate(policy):
        cumulative += float(probability) / total
        if threshold <= cumulative:
            return idx
    return int(np.argmax(policy))


def _env_config(config: TrainingPipelineConfig) -> ACEnvironmentConfig:
    return ACEnvironmentConfig(
        max_moves=config.max_moves,
        total_length_cap=config.total_length_cap,
    )
