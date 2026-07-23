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
    navigation_head_targets,
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
    # Navigation's two alpha-invariant value-head targets (see `ReplayExample`);
    # 0.0 off navigation, where the scalar `return_target` is regressed instead.
    success_target: float = 0.0
    progress_target: float = 0.0


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
    """One episode's transitions, its bootstrap value, and the head targets.

    ``success_targets`` and ``progress_targets`` are the per-transition navigation
    value-head return-to-go targets (empty/zero off navigation); they are computed
    here, where the whole trajectory and its bootstrap heads are in hand, rather
    than in the advantage pass.
    """

    transitions: list[_Transition]
    bootstrap_value: float
    metrics: EpisodeMetrics
    success_targets: list[float]
    progress_targets: list[float]


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
) -> _Rollout:
    """Play one episode by sampling the current policy and record every step."""
    rng = random.Random(seed)
    presentation = source.sample(seed)
    _, max_moves = episode_distance_and_moves(
        source, presentation, config.unknown_distance_max_moves
    )
    env = build_env(config, presentation, source, alpha, max_moves)
    scale = env.reward_scale
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
        scaled = reward * scale
        components = info.get("reward_components")
        transitions.append(
            _Transition(
                encoding,
                mask,
                action,
                log_prob,
                scaled,
                env.leaf_value(output),
                components if isinstance(components, RewardComponents) else None,
            )
        )
        rewards.append(scaled)
    bootstrap, success_tail, progress_tail = _bootstrap(
        env, encoder, model, action_count, terminated, bool(transitions)
    )
    is_navigation = config.reward_mode == "navigation"
    nav = env.navigation_episode_stats() if is_navigation else None
    if is_navigation and transitions:
        success_targets, progress_targets = navigation_head_targets(
            [t.components for t in transitions],
            reached_goal=terminated,
            gamma=config.gamma,
            max_shaping_progress=config.reward_config.max_shaping_progress,
            start_distance=env.navigation_episode_stats().start_distance,
            success_tail=success_tail,
            progress_tail=progress_tail,
        )
    else:
        success_targets = [0.0] * len(transitions)
        progress_targets = [0.0] * len(transitions)
    total = float(sum(rewards))
    metrics = EpisodeMetrics(total, total, terminated, len(transitions), nav)
    return _Rollout(transitions, bootstrap, metrics, success_targets, progress_targets)


def _bootstrap(
    env: ACEnvironment,
    encoder: StateEncoder,
    model: PolicyValueModel,
    action_count: int,
    terminated: bool,
    stepped: bool,
) -> tuple[float, float, float]:
    """Value and head tails of the state the episode stopped in.

    Zero everywhere at a goal or dead end (no future reward); otherwise the
    reconstructed leaf value plus the two head outputs, so the head return-to-go
    targets can be seeded past the truncation the same way GAE bootstraps the
    scalar value.
    """
    if terminated or not stepped or not any(env.legal_action_mask()):
        return 0.0, 0.0, 0.0
    output = model.apply(encoder.encode(env.state), action_count)
    return env.leaf_value(output), output.success, output.progress


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


def _init_rollout_worker(
    config: TrainingPipelineConfig,
    model_state: dict[str, Any],
    alpha: float | None,
) -> None:
    global _WORKER_CONFIG, _WORKER_ENCODER, _WORKER_MODEL, _WORKER_SOURCE, _WORKER_ALPHA
    use_single_torch_thread()
    _WORKER_CONFIG = config
    _WORKER_ENCODER = StateEncoder(config.max_relator_tokens)
    _WORKER_MODEL = model_from_json(model_state)
    _WORKER_SOURCE = build_instance_source(config)
    _WORKER_ALPHA = alpha


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
    )


def collect_rollouts(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    model: TrainablePolicyValueModel,
    seed: int,
    iteration: int,
    source: InstanceSource,
    alpha: float | None = None,
) -> tuple[list[PPOExample], list[EpisodeMetrics]]:
    """Collect one iteration's rollouts and build advantage-normalized examples.

    ``source`` is the run's instance source, reused across iterations. Workers
    open their own handle on it, which memory-maps the same dataset sidecar rather
    than parsing the dataset again (see :mod:`ac_zero.datasets.instance_store`).
    """
    seeds = [seed + iteration * 10_000 + index for index in range(config.episodes_per_iteration)]
    if resolve_worker_count(config.workers) <= 1:
        rollouts = [_collect_rollout(config, encoder, s, model, source, alpha) for s in seeds]
    else:
        rollouts = parallel_map(
            _rollout_worker,
            seeds,
            workers=config.workers,
            initializer=_init_rollout_worker,
            initargs=(config, model.to_json(), alpha),
        )
    scored: list[_Scored] = []
    for rollout in rollouts:
        estimates = _generalized_advantages(rollout, config.gamma, config.ppo_lambda)
        for transition, (advantage, return_target), success_target, progress_target in zip(
            rollout.transitions,
            estimates,
            rollout.success_targets,
            rollout.progress_targets,
            strict=True,
        ):
            scored.append(
                _Scored(transition, advantage, return_target, success_target, progress_target)
            )
    examples = _normalize_examples(scored)
    return examples, [rollout.metrics for rollout in rollouts]


@dataclass(frozen=True, slots=True)
class _Scored:
    """A transition with its advantage and every value-head return target."""

    transition: _Transition
    advantage: float
    return_target: float
    success_target: float
    progress_target: float


def _normalize_examples(scored: list[_Scored]) -> list[PPOExample]:
    """Standardize advantages across the batch and pack them into examples."""
    if not scored:
        return []
    advantages = np.asarray([item.advantage for item in scored], dtype=np.float64)
    std = float(advantages.std())
    denominator = std if std > 1e-8 else 1.0
    mean = float(advantages.mean())
    return [
        PPOExample(
            encoding=item.transition.encoding,
            legal_mask=item.transition.legal_mask,
            action=item.transition.action,
            old_log_prob=item.transition.log_prob,
            advantage=(item.advantage - mean) / denominator,
            return_target=item.return_target,
            success_target=item.success_target,
            progress_target=item.progress_target,
        )
        for item in scored
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
    ) -> PPOIterationResult:
        """Collect rollouts and apply this iteration's PPO updates to `model`."""
        examples, episodes = collect_rollouts(
            self.config, self.encoder, model, seed, iteration, self.source, alpha
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
                        grad_clip=self.config.grad_clip,
                        reward_mode=self.config.reward_mode,
                    )
                )
        return updates
