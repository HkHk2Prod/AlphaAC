from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
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


def build_env_config(config: TrainingPipelineConfig) -> ACEnvironmentConfig:
    return ACEnvironmentConfig(
        max_moves=config.max_moves,
        total_length_cap=config.total_length_cap,
        goal_mode=config.goal_mode,
        reward_mode=config.reward_mode,
        goal_reward=config.goal_reward,
        moveset=config.moveset,
    )


def collect_episodes(
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
        source = build_instance_source(config)
        return [
            _collect_episode(config, encoder, episode_seed, model, source)
            for episode_seed in episode_seeds
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
_WORKER_SOURCE: InstanceSource | None = None


def _init_episode_worker(config: TrainingPipelineConfig, model_state: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_ENCODER, _WORKER_MODEL, _WORKER_SOURCE
    _WORKER_CONFIG = config
    _WORKER_ENCODER = StateEncoder(config.max_word_length)
    _WORKER_MODEL = model_from_json(model_state)
    _WORKER_SOURCE = build_instance_source(config)


def _episode_worker(episode_seed: int) -> tuple[list[ReplayExample], EpisodeMetrics]:
    assert _WORKER_CONFIG is not None and _WORKER_ENCODER is not None and _WORKER_MODEL is not None
    assert _WORKER_SOURCE is not None
    return _collect_episode(
        _WORKER_CONFIG, _WORKER_ENCODER, episode_seed, _WORKER_MODEL, _WORKER_SOURCE
    )


def _collect_episode(
    config: TrainingPipelineConfig,
    encoder: StateEncoder,
    episode_seed: int,
    model: PolicyValueModel,
    source: InstanceSource,
) -> tuple[list[ReplayExample], EpisodeMetrics]:
    # A per-episode RNG seeded from the episode seed keeps action sampling
    # independent of execution order, so episodes can run in parallel and still
    # reproduce exactly.
    rng = random.Random(episode_seed)
    presentation = source.sample(episode_seed)
    env = ACEnvironment(presentation, build_env_config(config))
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
