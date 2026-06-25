from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ac_zero.agents.greedy import GreedySolver
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.base import PolicyValueModel
from ac_zero.models.registry import create_trainable_model
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.search.puct import PUCTMCTS, PUCTConfig
from ac_zero.system.manifests import ReproducibilityManifest
from ac_zero.training.callbacks import CallbackManager, default_training_callbacks
from ac_zero.training.checkpointing import CheckpointManager
from ac_zero.training.losses import (
    PolicyValueLoss,
    return_to_go,
    visit_count_policy,
)
from ac_zero.training.replay_buffer import ReplayBuffer


@dataclass(frozen=True, slots=True)
class TrainingPipelineConfig:
    """Configuration for the CPU policy/value training pipeline."""

    rank: int = 2
    scramble_depth: int = 3
    max_moves: int = 8
    total_length_cap: int = 128
    max_word_length: int = 32
    model: str = "linear_policy_value"
    mcts_simulations: int = 16
    c_puct: float = 1.5
    iterations: int = 2
    episodes_per_iteration: int = 4
    optimizer_updates: int = 4
    batch_size: int = 8
    replay_capacity: int = 512
    learning_rate: float = 0.05
    value_loss_weight: float = 1.0
    checkpoint_every: int = 1
    run_directory: str = "runs/train"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TrainingPipelineConfig:
        """Build a pipeline config from the repository's experiment YAML shape."""
        defaults = cls()
        dataset = _dict_value(data, "dataset")
        training = _dict_value(data, "training")
        return cls(
            rank=int(data.get("rank", defaults.rank)),
            scramble_depth=int(
                dataset.get("depth", data.get("scramble_depth", defaults.scramble_depth))
            ),
            max_moves=int(data.get("max_moves", defaults.max_moves)),
            total_length_cap=int(data.get("total_length_cap", defaults.total_length_cap)),
            max_word_length=int(data.get("max_word_length", defaults.max_word_length)),
            model=str(data.get("model", defaults.model)),
            mcts_simulations=int(
                training.get(
                    "mcts_simulations",
                    data.get("mcts_simulations", defaults.mcts_simulations),
                )
            ),
            c_puct=float(training.get("c_puct", data.get("c_puct", defaults.c_puct))),
            iterations=int(training.get("iterations", data.get("iterations", defaults.iterations))),
            episodes_per_iteration=int(
                training.get(
                    "episodes_per_iteration",
                    dataset.get(
                        "count",
                        data.get(
                            "episodes_per_iteration",
                            defaults.episodes_per_iteration,
                        ),
                    ),
                )
            ),
            optimizer_updates=int(
                training.get(
                    "optimizer_updates",
                    data.get("optimizer_updates", defaults.optimizer_updates),
                )
            ),
            batch_size=int(training.get("batch_size", data.get("batch_size", defaults.batch_size))),
            replay_capacity=int(
                training.get(
                    "replay_capacity",
                    data.get("replay_capacity", defaults.replay_capacity),
                )
            ),
            learning_rate=float(
                training.get("learning_rate", data.get("learning_rate", defaults.learning_rate))
            ),
            value_loss_weight=float(
                training.get(
                    "value_loss_weight",
                    data.get("value_loss_weight", defaults.value_loss_weight),
                )
            ),
            checkpoint_every=int(
                training.get(
                    "checkpoint_every",
                    data.get("checkpoint_every", defaults.checkpoint_every),
                )
            ),
            run_directory=str(
                training.get("run_directory", data.get("run_directory", defaults.run_directory))
            ),
        )

    def validate(self) -> None:
        """Reject impossible training settings before allocating run artifacts."""
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.scramble_depth < 0:
            raise ValueError("scramble_depth must be non-negative")
        if self.max_moves <= 0:
            raise ValueError("max_moves must be positive")
        if self.total_length_cap <= 0:
            raise ValueError("total_length_cap must be positive")
        if self.max_word_length <= 0:
            raise ValueError("max_word_length must be positive")
        if self.mcts_simulations <= 0:
            raise ValueError("mcts_simulations must be positive")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.episodes_per_iteration <= 0:
            raise ValueError("episodes_per_iteration must be positive")
        if self.optimizer_updates <= 0:
            raise ValueError("optimizer_updates must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.value_loss_weight < 0.0:
            raise ValueError("value_loss_weight must be non-negative")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")


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
            for episode_index in range(config.episodes_per_iteration):
                episode_seed = seed + iteration * 10_000 + episode_index
                examples, episode_metrics = _collect_episode(
                    config, encoder, episode_seed, rng, model
                )
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


def _collect_episode(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    episode_seed: int,
    rng: random.Random,
    model: PolicyValueModel,
) -> tuple[list[ReplayExample], EpisodeMetrics]:
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


def _dict_value(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}
