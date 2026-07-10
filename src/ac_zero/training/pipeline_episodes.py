from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.navigation_reward import EpisodeStats, RewardComponents
from ac_zero.models.base import PolicyValueModel
from ac_zero.models.registry import model_from_json
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.search.puct import PUCTMCTS, PUCTConfig
from ac_zero.system.parallel import parallel_map, resolve_worker_count
from ac_zero.training.instance_source import InstanceSource, build_instance_source
from ac_zero.training.losses import return_to_go, visit_count_policy
from ac_zero.training.pipeline_config import TrainingPipelineConfig


@dataclass(frozen=True, slots=True)
class ReplayExample:
    """One policy/value training target collected from a search state.

    ``components`` retains the separated navigation-reward parts (item 6 of the
    reward spec) so a buffer entry can be re-scored under a different alpha or
    scaling scheme later; ``None`` for the scalar-reward modes.
    """

    encoding: PaddedEncoding
    legal_mask: tuple[bool, ...]
    policy_target: NDArray[np.float64]
    value_target: float
    reward: float
    action: int
    components: RewardComponents | None = None


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Small aggregate metrics from one generated episode.

    ``nav`` carries the navigation reward's per-episode aggregate (progress,
    success, component sums) that feeds the alpha updater and the evaluation
    metrics; ``None`` for other reward modes.
    """

    total_return: float
    normalized_return: float
    success: bool
    moves: int
    nav: EpisodeStats | None = None


def batch_return_and_success(episodes: list[EpisodeMetrics]) -> tuple[float, float]:
    """Mean normalized return and success rate over one iteration's episodes."""
    mean_return = float(np.mean([episode.normalized_return for episode in episodes]))
    success_rate = float(np.mean([1.0 if episode.success else 0.0 for episode in episodes]))
    return mean_return, success_rate


def build_env_config(
    config: TrainingPipelineConfig, alpha: float | None = None, max_moves: int | None = None
) -> ACEnvironmentConfig:
    return ACEnvironmentConfig(
        max_moves=config.max_moves if max_moves is None else max_moves,
        total_length_cap=config.total_length_cap,
        goal_mode=config.goal_mode,
        reward_mode=config.reward_mode,
        goal_reward=config.goal_reward,
        reward_config=config.reward_config,
        alpha=alpha,
        moveset=config.moveset,
    )


def build_env(
    config: TrainingPipelineConfig,
    presentation: BalancedPresentation,
    source: InstanceSource,
    alpha: float | None = None,
    max_moves: int | None = None,
) -> ACEnvironment:
    """Construct the episode env, wiring the potential map for distance-based rewards.

    ``alpha`` is the shaping weight for the "navigation" reward this episode runs
    at (ignored by other modes); the training loop advances it between iterations.
    ``max_moves`` overrides the global horizon with this episode's ``3 * L + 6``
    when the distance curriculum is active (``None`` keeps ``config.max_moves``).
    """
    potentials = source.potentials if config.reward_mode in ("potential", "navigation") else None
    return ACEnvironment(
        presentation, build_env_config(config, alpha, max_moves), potentials=potentials
    )


def episode_distance_and_moves(
    source: InstanceSource, presentation: BalancedPresentation, unknown_max_moves: int
) -> tuple[int | None, int]:
    """Return this problem's distance ``L`` and its horizon ``max_moves``.

    Called under the distance curriculum, where the presentation was usually drawn
    with a ``max_distance`` cap and so carries a known, positive distance to the
    destination -- then the horizon is ``3 * L + 6``, scaling with the *sampled*
    ``L`` and never ``L_max``. A problem off the annotated graph has no known ``L``
    (returned as ``None``); it falls back to the large ``unknown_max_moves`` cutoff
    so it is given room to solve rather than truncated early.
    """
    distance = source.potentials.get(presentation.content_hash)
    if distance is None:
        return None, unknown_max_moves
    L = int(distance)
    return L, 3 * L + 6


def collect_episodes(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    model: TrainablePolicyValueModel,
    seed: int,
    iteration: int,
    source: InstanceSource,
    alpha: float | None = None,
    max_distance: int | None = None,
) -> list[tuple[list[ReplayExample], EpisodeMetrics]]:
    """Collect one iteration's self-play episodes, fanning out across processes.

    Each episode is fully determined by its seed and the current model, so the
    episodes run independently and are reassembled in order; the result is
    identical whether one or many worker processes are used. ``source`` is the
    run's instance source, reused across iterations. Workers open their own handle
    on it, which memory-maps the same dataset sidecar rather than parsing the
    dataset again (see :mod:`ac_zero.datasets.instance_store`). ``max_distance`` is
    the distance curriculum's ceiling this batch was sampled under (``None`` off
    the curriculum), held constant for the batch like ``alpha``.
    """
    episode_seeds = [
        seed + iteration * 10_000 + index for index in range(config.episodes_per_iteration)
    ]
    if resolve_worker_count(config.workers) <= 1:
        return [
            _collect_episode(config, encoder, episode_seed, model, source, alpha, max_distance)
            for episode_seed in episode_seeds
        ]
    return parallel_map(
        _episode_worker,
        episode_seeds,
        workers=config.workers,
        initializer=_init_episode_worker,
        initargs=(config, model.to_json(), alpha, max_distance),
    )


# Per-worker state, populated once by the process-pool initializer so the model
# is rebuilt from its serialized weights a single time per worker rather than
# pickled with every episode task.
_WORKER_CONFIG: TrainingPipelineConfig | None = None
_WORKER_ENCODER: StateEncoder | None = None
_WORKER_MODEL: PolicyValueModel | None = None
_WORKER_SOURCE: InstanceSource | None = None
_WORKER_ALPHA: float | None = None
_WORKER_MAX_DISTANCE: int | None = None


def _init_episode_worker(
    config: TrainingPipelineConfig,
    model_state: dict[str, Any],
    alpha: float | None,
    max_distance: int | None,
) -> None:
    global _WORKER_CONFIG, _WORKER_ENCODER, _WORKER_MODEL, _WORKER_SOURCE
    global _WORKER_ALPHA, _WORKER_MAX_DISTANCE
    _WORKER_CONFIG = config
    _WORKER_ENCODER = StateEncoder(config.max_word_length)
    _WORKER_MODEL = model_from_json(model_state)
    _WORKER_SOURCE = build_instance_source(config)
    _WORKER_ALPHA = alpha
    _WORKER_MAX_DISTANCE = max_distance


def _episode_worker(episode_seed: int) -> tuple[list[ReplayExample], EpisodeMetrics]:
    assert _WORKER_CONFIG is not None and _WORKER_ENCODER is not None and _WORKER_MODEL is not None
    assert _WORKER_SOURCE is not None
    return _collect_episode(
        _WORKER_CONFIG,
        _WORKER_ENCODER,
        episode_seed,
        _WORKER_MODEL,
        _WORKER_SOURCE,
        _WORKER_ALPHA,
        _WORKER_MAX_DISTANCE,
    )


def _collect_episode(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    episode_seed: int,
    model: PolicyValueModel,
    source: InstanceSource,
    alpha: float | None = None,
    max_distance: int | None = None,
) -> tuple[list[ReplayExample], EpisodeMetrics]:
    # A per-episode RNG seeded from the episode seed keeps action sampling
    # independent of execution order, so episodes can run in parallel and still
    # reproduce exactly.
    rng = random.Random(episode_seed)
    presentation = source.sample(episode_seed, max_distance)
    max_moves = None
    if max_distance is not None:
        _, max_moves = episode_distance_and_moves(
            source, presentation, config.curriculum_config.unknown_distance_max_moves
        )
    env = build_env(config, presentation, source, alpha, max_moves)
    mcts = PUCTMCTS(
        model, encoder, PUCTConfig(simulations=config.mcts_simulations, c_puct=config.c_puct)
    )
    pending: list[_PendingStep] = []
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
        _, reward, terminated, truncated, info = env.step(action)
        normalized_reward = reward / max(1.0, float(env.initial.total_length))
        pending.append(
            (encoding, legal_mask, policy_target, action, normalized_reward, _components(info))
        )
        rewards.append(normalized_reward)
    returns = return_to_go(rewards, config.gamma)
    examples = [
        ReplayExample(
            encoding=encoding,
            legal_mask=legal_mask,
            policy_target=policy_target,
            value_target=returns[idx],
            reward=reward,
            action=action,
            components=components,
        )
        for idx, (encoding, legal_mask, policy_target, action, reward, components) in enumerate(
            pending
        )
    ]
    total_return = float(sum(rewards))
    nav = env.navigation_episode_stats() if config.reward_mode == "navigation" else None
    return examples, EpisodeMetrics(
        total_return=total_return,
        normalized_return=total_return,
        success=terminated,
        moves=len(pending),
        nav=nav,
    )


# One collected step awaiting its return-to-go target: encoding, legal mask,
# policy target, action, normalized reward, and the navigation reward components.
_PendingStep = tuple[
    PaddedEncoding, tuple[bool, ...], NDArray[np.float64], int, float, RewardComponents | None
]


def _components(info: dict[str, Any]) -> RewardComponents | None:
    """Pull the navigation reward components out of a step's info, if present."""
    components = info.get("reward_components")
    return components if isinstance(components, RewardComponents) else None


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
