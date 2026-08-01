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
from ac_zero.models.torch_utils import use_single_torch_thread
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.search.puct import PUCTMCTS, PUCTConfig
from ac_zero.system.parallel import parallel_map, resolve_worker_count
from ac_zero.training.pipeline.instance_source import InstanceSource, build_instance_source
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
from ac_zero.training.ppo.losses import return_to_go, visit_count_policy


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
    # Navigation's two alpha-invariant value-head targets (0.0 off navigation, where
    # the scalar `value_target` is regressed instead): `success_target` is the
    # discounted-success return-to-go gamma**(steps-to-goal), and `progress_target`
    # the clipped shaping return-to-go normalized by the start distance L0.
    success_target: float = 0.0
    progress_target: float = 0.0


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Small aggregate metrics from one generated episode.

    ``normalized_return`` is the episode's return in difficulty-invariant units
    (see :func:`normalize_return`); the raw sum it came from stays available on
    ``nav`` as ``total_reward``.

    ``nav`` carries the navigation reward's per-episode aggregate (progress,
    success, component sums) that feeds the alpha updater and the evaluation
    metrics; ``None`` for other reward modes.
    """

    normalized_return: float
    success: bool
    moves: int
    nav: EpisodeStats | None = None


def normalize_return(total_return: float, nav: EpisodeStats | None) -> float:
    """Scale one episode's return so instances of different difficulty compare.

    A navigation return is dominated by the terminal bonus, which *is* the
    episode's start distance ``L0`` (:meth:`RewardComputer.step`), so a solved
    ``L0 = 20`` instance scores ten times a solved ``L0 = 2`` one. An iteration's
    mean over a difficulty-mixed batch then swings with which groups the seeds
    happened to draw rather than with the policy -- the run's reward curve
    oscillates even while nothing is learning or unlearning. Dividing by ``L0``
    makes a solve worth about 1 whatever it started from, so the mean moves only
    when the policy does.

    The other reward modes already carry their own ``1/initial_length`` scale
    (:attr:`ACEnvironment.reward_scale`), so their return passes through.
    """
    if nav is None:
        return total_return
    return total_return / max(1, nav.start_distance)


@dataclass(frozen=True, slots=True)
class BatchMetrics:
    """One iteration's self-play summary, in difficulty-invariant units.

    ``progress_rate`` -- the mean fraction of the start distance the episodes
    closed at their closest approach -- and ``start_distance`` are ``None`` off
    navigation, where no episode carries the distance stats they come from.

    ``start_distance`` and ``moves`` are recorded together because the horizon is
    a function of the first (``3 * L + 6``), so the pair is what says whether the
    episodes ran the length their instances called for. A stored run whose mean
    ``moves`` exceeds ``3 * max_distance + 6`` is drawing a horizon from somewhere
    other than its instance's distance, and neither number was previously written
    anywhere a finished run could be audited from.

    ``no_start_distance`` is the share of navigation episodes that began from a
    node with no usable distance to the destination. It should be exactly zero:
    a navigation run filters its instances to annotated groups
    (``require_potential``), and an episode without a start distance silently
    loses both its destination bonus (which *is* that distance) and its horizon,
    falling back to ``unknown_distance_max_moves`` -- an order-of-magnitude change
    that nothing previously reported.
    """

    mean_return: float
    success_rate: float
    progress_rate: float | None
    start_distance: float | None
    no_start_distance: float | None
    moves: float


def batch_episode_metrics(episodes: list[EpisodeMetrics]) -> BatchMetrics:
    """Summarize one iteration's episodes into the series the run records."""
    nav = [episode.nav for episode in episodes if episode.nav is not None]
    return BatchMetrics(
        mean_return=float(np.mean([episode.normalized_return for episode in episodes])),
        success_rate=float(np.mean([1.0 if episode.success else 0.0 for episode in episodes])),
        progress_rate=float(np.mean([stats.progress_rate for stats in nav])) if nav else None,
        start_distance=float(np.mean([stats.start_distance for stats in nav])) if nav else None,
        no_start_distance=(
            float(np.mean([1.0 if stats.start_distance <= 0 else 0.0 for stats in nav]))
            if nav
            else None
        ),
        moves=float(np.mean([episode.moves for episode in episodes])),
    )


def moves_for_distance(distance: int) -> int:
    """Self-play horizon for a problem at distance ``distance`` from the target.

    The default (and only) horizon rule: ``3 * L + 6`` moves for a problem whose
    shortest-path distance to the destination is ``L``. Enough slack to recover
    from a few wrong turns while still bounding an episode tightly to its problem's
    difficulty rather than a single global cap.
    """
    return 3 * distance + 6


def navigation_head_targets(
    components: list[RewardComponents | None],
    *,
    reached_goal: bool,
    gamma: float,
    max_shaping_progress: int,
    start_distance: int,
    success_tail: float = 0.0,
    progress_tail: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Per-step return-to-go targets for the two alpha-invariant navigation heads.

    ``success`` regresses ``gamma**(steps-to-goal)`` -- the return-to-go of a reward
    that is 1 only on the goal-reaching step -- so it estimates the discounted
    probability this state reaches the destination. ``progress`` regresses the
    clipped shaping return-to-go divided by the start distance ``L0``, the exact
    ``B~`` the reconstruction multiplies by ``alpha``. The clip mirrors the reward
    (:class:`RewardConfig.max_shaping_progress`), so target and reward price an
    off-graph re-entry the same way.

    The ``*_tail`` seeds carry a bootstrap value past the last step, in each head's
    own units -- ``success_tail`` a success probability, ``progress_tail`` a ``B~``
    -- so PPO seeds them from the model's heads at the final state; AlphaZero leaves
    them zero, treating truncation as a dead end.
    """
    cap = max_shaping_progress
    denom = max(1, start_distance)
    last = len(components) - 1
    success = 0.0 if reached_goal else success_tail
    progress = progress_tail
    success_out = [0.0] * len(components)
    progress_out = [0.0] * len(components)
    for idx in range(last, -1, -1):
        goal_here = 1.0 if (reached_goal and idx == last) else 0.0
        success = goal_here + gamma * success
        component = components[idx]
        clipped = 0.0 if component is None else _clip(component.distance_progress, cap)
        progress = clipped / denom + gamma * progress
        success_out[idx] = success
        progress_out[idx] = progress
    return success_out, progress_out


def _clip(value: int, cap: int) -> float:
    return float(max(-cap, min(cap, value)))


def build_env_config(
    config: TrainingPipelineConfig, alpha: float | None, max_moves: int
) -> ACEnvironmentConfig:
    return ACEnvironmentConfig(
        max_moves=max_moves,
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
    alpha: float | None,
    max_moves: int,
) -> ACEnvironment:
    """Construct the episode env, wiring the potential map for distance-based rewards.

    ``alpha`` is the shaping weight for the "navigation" reward this episode runs
    at (ignored by other modes); the training loop advances it between iterations.
    ``max_moves`` is this episode's horizon -- ``3 * L + 6`` for its distance ``L``
    (see :func:`episode_distance_and_moves`).

    The env is given the *run's* encoder, not a default one: its legal-move mask is what
    keeps the episode inside the states the model can represent, so a mask computed
    against some other capacity would let the episode walk into a presentation the
    training encoder then refuses.
    """
    potentials = source.potentials if config.reward_mode in ("potential", "navigation") else None
    return ACEnvironment(
        presentation,
        build_env_config(config, alpha, max_moves),
        StateEncoder(config.max_relator_tokens),
        potentials=potentials,
    )


def episode_distance_and_moves(
    source: InstanceSource, presentation: BalancedPresentation, unknown_max_moves: int
) -> tuple[int | None, int]:
    """Return this problem's distance ``L`` and its horizon ``max_moves``.

    The horizon is always ``3 * L + 6`` for a problem whose shortest-path distance
    to the destination is known -- the default rule for every training run, scaling
    each episode to its own difficulty rather than a fixed global cap. A problem off
    the annotated graph (a scramble, or an unannotated dataset group) has no known
    ``L`` (returned as ``None``); it falls back to the large ``unknown_max_moves``
    cutoff so it is given room to solve rather than truncated early.
    """
    distance = source.potentials.get(presentation.content_hash)
    if distance is None:
        return None, unknown_max_moves
    L = int(distance)
    return L, moves_for_distance(L)


def collect_episodes(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    model: TrainablePolicyValueModel,
    seed: int,
    iteration: int,
    source: InstanceSource,
    alpha: float | None = None,
) -> list[tuple[list[ReplayExample], EpisodeMetrics]]:
    """Collect one iteration's self-play episodes, fanning out across processes.

    Each episode is fully determined by its seed and the current model, so the
    episodes run independently and are reassembled in order; the result is
    identical whether one or many worker processes are used. ``source`` is the
    run's instance source, reused across iterations. Workers open their own handle
    on it, which memory-maps the same dataset sidecar rather than parsing the
    dataset again (see :mod:`ac_zero.datasets.instance_store`).
    """
    episode_seeds = [
        seed + iteration * 10_000 + index for index in range(config.episodes_per_iteration)
    ]
    if resolve_worker_count(config.workers) <= 1:
        return [
            _collect_episode(config, encoder, episode_seed, model, source, alpha)
            for episode_seed in episode_seeds
        ]
    return parallel_map(
        _episode_worker,
        episode_seeds,
        workers=config.workers,
        initializer=_init_episode_worker,
        initargs=(config, model.to_json(), alpha),
    )


# Per-worker state, populated once by the process-pool initializer so the model
# is rebuilt from its serialized weights a single time per worker rather than
# pickled with every episode task.
_WORKER_CONFIG: TrainingPipelineConfig | None = None
_WORKER_ENCODER: StateEncoder | None = None
_WORKER_MODEL: PolicyValueModel | None = None
_WORKER_SOURCE: InstanceSource | None = None
_WORKER_ALPHA: float | None = None


def _init_episode_worker(
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
    )


def _collect_episode(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    episode_seed: int,
    model: PolicyValueModel,
    source: InstanceSource,
    alpha: float | None = None,
) -> tuple[list[ReplayExample], EpisodeMetrics]:
    # A per-episode RNG seeded from the episode seed keeps action sampling
    # independent of execution order, so episodes can run in parallel and still
    # reproduce exactly.
    rng = random.Random(episode_seed)
    presentation = source.sample(episode_seed)
    _, max_moves = episode_distance_and_moves(
        source, presentation, config.unknown_distance_max_moves
    )
    env = build_env(config, presentation, source, alpha, max_moves)
    mcts = PUCTMCTS(
        model, encoder, PUCTConfig(simulations=config.mcts_simulations, c_puct=config.c_puct)
    )
    scale = env.reward_scale
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
        scaled_reward = reward * scale
        pending.append(
            (encoding, legal_mask, policy_target, action, scaled_reward, _components(info))
        )
        rewards.append(scaled_reward)
    returns = return_to_go(rewards, config.gamma)
    is_navigation = config.reward_mode == "navigation"
    success_targets, progress_targets = _episode_head_targets(
        config, env, [components for *_, components in pending], terminated, is_navigation
    )
    examples = [
        ReplayExample(
            encoding=encoding,
            legal_mask=legal_mask,
            policy_target=policy_target,
            value_target=returns[idx],
            reward=reward,
            action=action,
            components=components,
            success_target=success_targets[idx],
            progress_target=progress_targets[idx],
        )
        for idx, (encoding, legal_mask, policy_target, action, reward, components) in enumerate(
            pending
        )
    ]
    total_return = float(sum(rewards))
    nav = env.navigation_episode_stats() if is_navigation else None
    return examples, EpisodeMetrics(
        normalized_return=normalize_return(total_return, nav),
        success=terminated,
        moves=len(pending),
        nav=nav,
    )


def _episode_head_targets(
    config: TrainingPipelineConfig,
    env: ACEnvironment,
    components: list[RewardComponents | None],
    reached_goal: bool,
    is_navigation: bool,
) -> tuple[list[float], list[float]]:
    """The two navigation head targets for one episode; zeros off navigation."""
    if not is_navigation or not components:
        return [0.0] * len(components), [0.0] * len(components)
    return navigation_head_targets(
        components,
        reached_goal=reached_goal,
        gamma=config.gamma,
        max_shaping_progress=config.reward_config.max_shaping_progress,
        start_distance=env.navigation_episode_stats().start_distance,
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
