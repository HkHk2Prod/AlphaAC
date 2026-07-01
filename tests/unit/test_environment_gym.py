from __future__ import annotations

import gymnasium
import numpy as np

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ENV_ID, ACEnvironment, ACEnvironmentConfig


def _env(max_moves: int = 4) -> ACEnvironment:
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    return ACEnvironment(pres, ACEnvironmentConfig(max_moves=max_moves, mask_noops=False))


def test_env_is_gymnasium_subclass_with_spaces() -> None:
    env = _env()
    assert isinstance(env, gymnasium.Env)
    assert env.action_space.n == len(env.catalog)
    assert set(env.observation_space.spaces) == {"tokens", "mask", "scalar_features"}


def test_reset_returns_observation_and_info() -> None:
    env = _env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["state"] is env.state
    assert info["action_mask"].shape == (len(env.catalog),)


def test_step_returns_conformant_five_tuple() -> None:
    env = _env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["state"] is env.state
    assert info["state"].moves_used == 1


def test_as_observation_matches_observation_space() -> None:
    env = _env()
    encoding = StateEncoder().encode(env.state)
    obs = encoding.as_observation()
    assert obs["mask"].dtype == np.int8
    assert np.array_equal(obs["tokens"], encoding.tokens)
    assert env.observation_space.contains(obs)


def test_registered_and_constructible_via_gym_make() -> None:
    assert ENV_ID in gymnasium.registry
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    env = gymnasium.make(ENV_ID, presentation=pres, config=ACEnvironmentConfig(max_moves=3))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.step(env.action_space.sample())
    env.close()
