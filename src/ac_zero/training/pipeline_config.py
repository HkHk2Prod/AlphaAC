from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ac_zero.environment.rewards import REWARD_MODES
from ac_zero.moves.universal import MOVE_SET_NAMES


@dataclass(frozen=True, slots=True)
class TrainingPipelineConfig:
    """Configuration for the CPU policy/value training pipeline."""

    rank: int = 2
    scramble_depth: int = 3
    # Optional grown group dataset to seed self-play from instead of random
    # scrambles. `dataset_path` points at a downloaded ``.groups.json`` file;
    # `dataset_annotations_path` is its companion ``.<moveset>.annotations.json``,
    # which carries the per-group distance to origin the curriculum reads.
    # `dataset_max_difficulty`
    # caps which groups are used by their distance to origin (None = all);
    # `dataset_bucket` names the Hugging Face bucket the CLI/notebook pulls from.
    dataset_path: str | None = None
    dataset_annotations_path: str | None = None
    dataset_max_difficulty: int | None = None
    dataset_bucket: str | None = None
    max_moves: int = 8
    total_length_cap: int = 128
    max_word_length: int = 32
    goal_mode: str = "exact_standard"
    reward_mode: str = "length_reduction_and_goal"
    # Named move set (`ac_zero.moves.universal.MOVE_SET_NAMES`) self-play actually
    # steps with.
    moveset: str = "strict-ac"
    goal_reward: float = 1.0
    # Reward discount applied to every training pipeline: the AlphaZero
    # return-to-go targets and the PPO GAE returns/advantages alike. `gamma < 1`
    # weights nearer rewards more, so shorter paths to the goal are preferred; it
    # is also what makes potential-based shaping (path-length invariant
    # undiscounted) mildly prefer shorter descents to the trivial group.
    gamma: float = 0.99
    model: str = "linear_policy_value"
    # Training backend: "alphazero" (PUCT self-play) or "ppo" (on-policy PPO).
    agent: str = "alphazero"
    mcts_simulations: int = 16
    c_puct: float = 1.5
    # PPO backend hyperparameters (ignored by the AlphaZero backend). Rollout
    # count reuses `episodes_per_iteration`, the minibatch size `batch_size`, the
    # value-loss coefficient `value_loss_weight`, and the discount `gamma`.
    ppo_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    iterations: int = 2
    episodes_per_iteration: int = 4
    optimizer_updates: int = 4
    batch_size: int = 8
    replay_capacity: int = 512
    learning_rate: float = 0.05
    value_loss_weight: float = 1.0
    checkpoint_every: int = 1
    # Emit a terminal progress line at INFO on the first and every
    # ``progress_every``-th recurring event (self-play iteration, optimizer
    # step); the steps in between are logged at DEBUG so the JSONL event log and
    # ASCII graphs still receive every point while the terminal stays readable on
    # long Kaggle/PC runs.
    progress_every: int = 100
    # Optional soft wall-clock budget in seconds. When set, the run stops at the
    # first iteration boundary past the deadline and still writes its checkpoint,
    # plots, and summary -- so a hosted run (Kaggle) ends cleanly and uploads its
    # best model instead of being killed mid-iteration. None runs all `iterations`.
    time_limit_s: float | None = None
    run_directory: str = "runs/train"
    # Optional local checkpoint (``best.json``/``latest.json`` payload) whose model
    # weights initialize this run -- a warm start from a previous run's best model.
    warm_start: str | None = None
    # Override for the Hugging Face checkpoint name; ``None`` derives it from the
    # task/model identity (see ``training.checkpoint_name.derive_checkpoint_name``).
    checkpoint_name: str | None = None
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
            dataset_path=_optional_str(dataset.get("path", data.get("dataset_path"))),
            dataset_annotations_path=_optional_str(dataset.get("annotations")),
            dataset_max_difficulty=_optional_int(dataset.get("max_difficulty")),
            dataset_bucket=_optional_str(dataset.get("bucket")),
            max_moves=int(data.get("max_moves", data.get("horizon", defaults.max_moves))),
            total_length_cap=int(data.get("total_length_cap", defaults.total_length_cap)),
            max_word_length=int(data.get("max_word_length", defaults.max_word_length)),
            goal_mode=str(data.get("goal_mode", defaults.goal_mode)),
            reward_mode=str(
                training.get("reward_mode", data.get("reward_mode", defaults.reward_mode))
            ),
            moveset=str(data.get("moveset", defaults.moveset)),
            goal_reward=float(
                training.get("goal_reward", data.get("goal_reward", defaults.goal_reward))
            ),
            gamma=float(training.get("gamma", data.get("gamma", defaults.gamma))),
            model=str(data.get("model", defaults.model)),
            agent=str(data.get("agent", defaults.agent)),
            mcts_simulations=int(
                training.get(
                    "mcts_simulations",
                    data.get("mcts_simulations", defaults.mcts_simulations),
                )
            ),
            c_puct=float(training.get("c_puct", data.get("c_puct", defaults.c_puct))),
            ppo_lambda=float(
                training.get("ppo_lambda", data.get("ppo_lambda", defaults.ppo_lambda))
            ),
            ppo_clip=float(training.get("ppo_clip", data.get("ppo_clip", defaults.ppo_clip))),
            ppo_epochs=int(training.get("ppo_epochs", data.get("ppo_epochs", defaults.ppo_epochs))),
            entropy_coef=float(
                training.get("entropy_coef", data.get("entropy_coef", defaults.entropy_coef))
            ),
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
            progress_every=int(
                training.get(
                    "progress_every",
                    data.get("progress_every", defaults.progress_every),
                )
            ),
            run_directory=str(
                training.get("run_directory", data.get("run_directory", defaults.run_directory))
            ),
            warm_start=_optional_str(training.get("warm_start", data.get("warm_start"))),
            checkpoint_name=_optional_str(
                training.get("checkpoint_name", data.get("checkpoint_name"))
            ),
            workers=int(training.get("workers", data.get("workers", defaults.workers))),
        )

    def validate(self) -> None:
        """Reject impossible training settings before allocating run artifacts."""
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.scramble_depth < 0:
            raise ValueError("scramble_depth must be non-negative")
        if self.dataset_max_difficulty is not None and self.dataset_max_difficulty < 0:
            raise ValueError("dataset_max_difficulty must be non-negative")
        if self.max_moves <= 0:
            raise ValueError("max_moves must be positive")
        if self.total_length_cap <= 0:
            raise ValueError("total_length_cap must be positive")
        if self.max_word_length <= 0:
            raise ValueError("max_word_length must be positive")
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if self.reward_mode == "potential" and not self.dataset_annotations_path:
            raise ValueError(
                "reward_mode 'potential' needs dataset.annotations for the distance to "
                "the trivial group; without it the potential falls back to length everywhere"
            )
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.moveset not in MOVE_SET_NAMES:
            raise ValueError(f"moveset must be one of {MOVE_SET_NAMES}")
        if self.dataset_max_difficulty is not None and not self.dataset_annotations_path:
            raise ValueError(
                "dataset.max_difficulty filters by distance to origin, which needs "
                "dataset.annotations"
            )
        if self.goal_reward < 0.0:
            raise ValueError("goal_reward must be non-negative")
        if self.agent not in ("alphazero", "ppo"):
            raise ValueError("agent must be 'alphazero' or 'ppo'")
        if self.mcts_simulations <= 0:
            raise ValueError("mcts_simulations must be positive")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive")
        if not 0.0 <= self.ppo_lambda <= 1.0:
            raise ValueError("ppo_lambda must be in [0, 1]")
        if self.ppo_clip <= 0.0:
            raise ValueError("ppo_clip must be positive")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
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
        if self.progress_every <= 0:
            raise ValueError("progress_every must be positive")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive when set")


def _dict_value(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _optional_str(value: Any) -> str | None:
    return str(value) if value else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
