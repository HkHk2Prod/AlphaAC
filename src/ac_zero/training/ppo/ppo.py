from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.env import ACEnvironment
from ac_zero.environment.navigation_reward import RewardComponents
from ac_zero.models.base import PolicyValueModel
from ac_zero.models.registry import model_from_json
from ac_zero.models.torch_utils import use_single_torch_thread
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.system.parallel import parallel_map, resolve_worker_count
from ac_zero.training.pipeline.instance_source import InstanceSource, build_instance_source
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
from ac_zero.training.pipeline.pipeline_episodes import (
    EpisodeMetrics,
    build_env,
    episode_distance_and_moves,
)
from ac_zero.training.ppo.losses import PPOBatchStats, masked_softmax, sample_from_policy


@dataclass(frozen=True, slots=True)
class PPOExample:
    """One PPO training target: an action taken, its old log-prob, and estimates.

    `advantage` is the normalized generalized advantage estimate and
    `return_target` the value-head regression target (advantage plus baseline).
    """

    encoding: PaddedEncoding
    legal_mask: tuple[bool, ...]
    action: int
    old_log_prob: float
    advantage: float
    return_target: float


@dataclass(frozen=True, slots=True)
class _Transition:
    """One sampled step retained for advantage estimation.

    ``components`` keeps the separated navigation-reward parts (item 6 of the
    reward spec) so a stored transition can be re-scored later; ``None`` for the
    scalar-reward modes.
    """

    encoding: PaddedEncoding
    legal_mask: tuple[bool, ...]
    action: int
    log_prob: float
    reward: float
    value: float
    components: RewardComponents | None = None


@dataclass(frozen=True, slots=True)
class _Rollout:
    """One episode's transitions plus the bootstrap value of the final state."""

    transitions: list[_Transition]
    bootstrap_value: float
    metrics: EpisodeMetrics


@dataclass(frozen=True, slots=True)
class PPOIterationResult:
    """What one PPO iteration produced for the training run to log."""

    example_count: int
    episodes: list[EpisodeMetrics] = field(default_factory=list)
    updates: list[PPOBatchStats] = field(default_factory=list)


def _collect_rollout(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    seed: int,
    model: PolicyValueModel,
    source: InstanceSource,
    alpha: float | None = None,
    max_distance: int | None = None,
) -> _Rollout:
    """Play one episode by sampling the current policy and record every step."""
    rng = random.Random(seed)
    presentation = source.sample(seed, max_distance)
    start_distance, max_moves = episode_distance_and_moves(
        source, presentation, config.curriculum_config.unknown_distance_max_moves
    )
    env = build_env(config, presentation, source, alpha, max_moves)
    scale = 1.0 / max(1.0, float(env.initial.total_length))
    action_count = len(env.catalog)
    transitions: list[_Transition] = []
    rewards: list[float] = []
    terminated = truncated = False
    while not terminated and not truncated:
        mask = env.legal_action_mask()
        if not any(mask):
            break
        encoding = encoder.encode(env.state)
        output = model.apply(encoding, action_count)
        probs = masked_softmax(output.logits, mask)
        action = sample_from_policy(probs, rng)
        log_prob = math.log(max(float(probs[action]), 1e-12))
        _, reward, terminated, truncated, info = env.step(action)
        normalized = reward * scale
        components = info.get("reward_components")
        transitions.append(
            _Transition(
                encoding,
                mask,
                action,
                log_prob,
                normalized,
                float(output.value),
                components if isinstance(components, RewardComponents) else None,
            )
        )
        rewards.append(normalized)
    bootstrap = _bootstrap_value(env, encoder, model, action_count, terminated, bool(transitions))
    total = float(sum(rewards))
    nav = env.navigation_episode_stats() if config.reward_mode == "navigation" else None
    metrics = EpisodeMetrics(total, total, terminated, len(transitions), nav, start_distance)
    return _Rollout(transitions, bootstrap, metrics)


def _bootstrap_value(
    env: ACEnvironment,
    encoder: StateEncoder,
    model: PolicyValueModel,
    action_count: int,
    terminated: bool,
    stepped: bool,
) -> float:
    """Value of the state the episode stopped in: zero at a goal or dead end."""
    if terminated or not stepped or not any(env.legal_action_mask()):
        return 0.0
    return float(model.apply(encoder.encode(env.state), action_count).value)


def _generalized_advantages(
    rollout: _Rollout, gamma: float, gae_lambda: float
) -> list[tuple[float, float]]:
    """Return per-step ``(advantage, return_target)`` via GAE(gamma, lambda)."""
    out: list[tuple[float, float]] = []
    advantage = 0.0
    next_value = rollout.bootstrap_value
    for transition in reversed(rollout.transitions):
        delta = transition.reward + gamma * next_value - transition.value
        advantage = delta + gamma * gae_lambda * advantage
        out.append((advantage, advantage + transition.value))
        next_value = transition.value
    out.reverse()
    return out


# Rollouts are independent given the current model, so they fan out across worker
# processes exactly like MCTS self-play; per-worker state is populated once by the
# pool initializer so the model is rebuilt from its weights a single time.
_WORKER_CONFIG: TrainingPipelineConfig | None = None
_WORKER_ENCODER: StateEncoder | None = None
_WORKER_MODEL: PolicyValueModel | None = None
_WORKER_SOURCE: InstanceSource | None = None
_WORKER_ALPHA: float | None = None
_WORKER_MAX_DISTANCE: int | None = None


def _init_rollout_worker(
    config: TrainingPipelineConfig,
    model_state: dict[str, Any],
    alpha: float | None,
    max_distance: int | None,
) -> None:
    global _WORKER_CONFIG, _WORKER_ENCODER, _WORKER_MODEL, _WORKER_SOURCE
    global _WORKER_ALPHA, _WORKER_MAX_DISTANCE
    use_single_torch_thread()
    _WORKER_CONFIG = config
    _WORKER_ENCODER = StateEncoder(config.max_relator_tokens)
    _WORKER_MODEL = model_from_json(model_state)
    _WORKER_SOURCE = build_instance_source(config)
    _WORKER_ALPHA = alpha
    _WORKER_MAX_DISTANCE = max_distance


def _rollout_worker(seed: int) -> _Rollout:
    assert _WORKER_CONFIG is not None and _WORKER_ENCODER is not None and _WORKER_MODEL is not None
    assert _WORKER_SOURCE is not None
    return _collect_rollout(
        _WORKER_CONFIG,
        _WORKER_ENCODER,
        seed,
        _WORKER_MODEL,
        _WORKER_SOURCE,
        _WORKER_ALPHA,
        _WORKER_MAX_DISTANCE,
    )


def collect_rollouts(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    model: TrainablePolicyValueModel,
    seed: int,
    iteration: int,
    source: InstanceSource,
    alpha: float | None = None,
    max_distance: int | None = None,
) -> tuple[list[PPOExample], list[EpisodeMetrics]]:
    """Collect one iteration's rollouts and build advantage-normalized examples.

    ``source`` is the run's instance source, reused across iterations. Workers
    open their own handle on it, which memory-maps the same dataset sidecar rather
    than parsing the dataset again (see :mod:`ac_zero.datasets.instance_store`).
    ``max_distance`` is the distance curriculum's ceiling this batch was sampled
    under (``None`` off the curriculum), held constant across the batch.
    """
    seeds = [seed + iteration * 10_000 + index for index in range(config.episodes_per_iteration)]
    if resolve_worker_count(config.workers) <= 1:
        rollouts = [
            _collect_rollout(config, encoder, s, model, source, alpha, max_distance) for s in seeds
        ]
    else:
        rollouts = parallel_map(
            _rollout_worker,
            seeds,
            workers=config.workers,
            initializer=_init_rollout_worker,
            initargs=(config, model.to_json(), alpha, max_distance),
        )
    scored: list[tuple[_Transition, float, float]] = []
    for rollout in rollouts:
        estimates = _generalized_advantages(rollout, config.gamma, config.ppo_lambda)
        for transition, (advantage, return_target) in zip(
            rollout.transitions, estimates, strict=True
        ):
            scored.append((transition, advantage, return_target))
    examples = _normalize_examples(scored)
    return examples, [rollout.metrics for rollout in rollouts]


def _normalize_examples(
    scored: list[tuple[_Transition, float, float]],
) -> list[PPOExample]:
    """Standardize advantages across the batch and pack them into examples."""
    if not scored:
        return []
    advantages = np.asarray([advantage for _, advantage, _ in scored], dtype=np.float64)
    std = float(advantages.std())
    denominator = std if std > 1e-8 else 1.0
    mean = float(advantages.mean())
    return [
        PPOExample(
            encoding=transition.encoding,
            legal_mask=transition.legal_mask,
            action=transition.action,
            old_log_prob=transition.log_prob,
            advantage=(advantage - mean) / denominator,
            return_target=return_target,
        )
        for transition, advantage, return_target in scored
    ]


class PPOTrainer:
    """On-policy PPO learner over the shared policy-value model.

    Each iteration collects fresh rollouts from the current policy, estimates
    advantages with GAE, then runs several epochs of minibatch clipped-surrogate
    updates over that data. It owns no run state, so the training pipeline drives
    it iteration by iteration and handles logging and checkpoints.
    """

    def __init__(
        self, config: TrainingPipelineConfig, encoder: StateEncoder, source: InstanceSource
    ) -> None:
        """Bind the trainer to the run config, state encoder, and instance source."""
        self.config = config
        self.encoder = encoder
        self.source = source

    def run_iteration(
        self,
        model: TrainablePolicyValueModel,
        seed: int,
        iteration: int,
        rng: random.Random,
        alpha: float | None = None,
        max_distance: int | None = None,
    ) -> PPOIterationResult:
        """Collect rollouts and apply this iteration's PPO updates to `model`."""
        examples, episodes = collect_rollouts(
            self.config, self.encoder, model, seed, iteration, self.source, alpha, max_distance
        )
        updates = self._optimize(model, examples, rng) if examples else []
        return PPOIterationResult(len(examples), episodes, updates)

    def _optimize(
        self,
        model: TrainablePolicyValueModel,
        examples: list[PPOExample],
        rng: random.Random,
    ) -> list[PPOBatchStats]:
        """Run epoch-by-minibatch clipped updates, shuffling deterministically."""
        indices = list(range(len(examples)))
        size = max(1, self.config.batch_size)
        updates: list[PPOBatchStats] = []
        for _ in range(self.config.ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), size):
                batch = [examples[i] for i in indices[start : start + size]]
                updates.append(
                    model.ppo_update(
                        batch,
                        learning_rate=self.config.learning_rate,
                        clip_ratio=self.config.ppo_clip,
                        value_weight=self.config.value_loss_weight,
                        entropy_weight=self.config.entropy_coef,
                    )
                )
        return updates
