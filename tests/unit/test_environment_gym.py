from __future__ import annotations

import random
from collections.abc import Iterator

import gymnasium
import numpy as np

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ENV_ID, ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.state import ACSearchState


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


def _content_hash_mask(env: ACEnvironment, state: ACSearchState) -> tuple[bool, ...]:
    """The mask as it was computed before: no-ops detected by SHA-256 content hash."""
    mask: list[bool] = []
    for move in env.catalog.moves:
        nxt = move.apply(state.presentation)
        legal = nxt.total_length <= env.config.total_length_cap and all(
            len(relator.letters) <= env.encoder.max_relator_tokens for relator in nxt.relators
        )
        if env.config.mask_noops and nxt.content_hash == state.presentation.content_hash:
            legal = False
        mask.append(legal)
    return tuple(mask)


def _walk(seed: int, steps: int = 12) -> Iterator[ACEnvironment]:
    """Yield the env at each state of a random legal walk from a scrambled start."""
    rng = random.Random(seed)
    pres = generate_solvable(2, 6, seed).presentation
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=steps, mask_noops=True))
    env.reset(seed=seed)
    for _ in range(steps):
        yield env
        legal = [i for i, ok in enumerate(env.legal_action_mask()) if ok]
        if not legal:
            return
        _, _, terminated, truncated, _ = env.step(rng.choice(legal))
        if terminated or truncated:
            return


def test_legal_action_mask_matches_content_hash_reference() -> None:
    """Relator-tuple no-op detection agrees exactly with the hash test it replaced."""
    for seed in range(25):
        for env in _walk(seed):
            assert env.legal_action_mask() == _content_hash_mask(env, env.state)


def test_legal_action_mask_flags_noops_only_when_enabled() -> None:
    """A move that leaves the relators untouched is illegal iff `mask_noops` is set."""
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    masked = ACEnvironment(pres, ACEnvironmentConfig(mask_noops=True)).legal_action_mask()
    unmasked = ACEnvironment(pres, ACEnvironmentConfig(mask_noops=False)).legal_action_mask()
    noops = [
        i
        for i, move in enumerate(ACEnvironment(pres, ACEnvironmentConfig()).catalog.moves)
        if move.apply(pres).relators == pres.relators
    ]
    assert noops, "the fixture must exercise at least one no-op move"
    assert all(not masked[i] for i in noops)
    assert all(unmasked[i] for i in noops)


def test_legal_action_mask_rejects_moves_over_the_length_cap() -> None:
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    env = ACEnvironment(pres, ACEnvironmentConfig(total_length_cap=pres.total_length))
    for action, legal in enumerate(env.legal_action_mask()):
        grown = env.catalog.moves[action].apply(pres).total_length > pres.total_length
        assert not (grown and legal)


def test_step_info_action_mask_matches_the_resulting_state() -> None:
    """`step` threads its mask into `info`; it must still describe the new state."""
    for seed in range(15):
        for env in _walk(seed, steps=6):
            legal = [i for i, ok in enumerate(env.legal_action_mask()) if ok]
            if not legal:
                continue
            _, _, _, _, info = env.step(legal[0])
            expected = np.asarray(env.legal_action_mask(env.state), dtype=np.int8)
            assert np.array_equal(info["action_mask"], expected)
            break


def test_registered_and_constructible_via_gym_make() -> None:
    assert ENV_ID in gymnasium.registry
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    env = gymnasium.make(ENV_ID, presentation=pres, config=ACEnvironmentConfig(max_moves=3))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.step(env.action_space.sample())
    env.close()
