from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


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
    # Self-play episodes are independent, so they fan out across this many worker
    # processes. The default 0 autodetects and uses every CPU core; set 1 to keep
    # the run in-process, or a negative count to leave that many cores free.
    # Results are collected in episode order, so the trained model is identical
    # regardless of the worker count.
    workers: int = 0

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
            workers=int(training.get("workers", data.get("workers", defaults.workers))),
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


def _dict_value(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}
