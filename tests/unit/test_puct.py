import numpy as np
import pytest

from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.navigation_reward import RewardConfig
from ac_zero.models.base import PolicyValueOutput
from ac_zero.models.registry import create_trainable_model
from ac_zero.moves.universal import moveset_catalog
from ac_zero.search.puct import PUCTMCTS, PUCTConfig, _MinMaxStats


def _mcts(simulations=48):
    model = create_trainable_model("residual_mlp", seed=0)
    return PUCTMCTS(model, StateEncoder(16), PUCTConfig(simulations=simulations))


def test_search_distributes_simulations_and_restores_root() -> None:
    instance = generate_solvable(rank=2, depth=2, seed=1)
    config = ACEnvironmentConfig(max_moves=8)
    env = ACEnvironment(instance.presentation, config)
    root_key = env.state.key
    stats = _mcts(simulations=48).search(env)
    assert env.state.key == root_key  # caller's state is untouched
    assert sum(stats.visit_counts) == 48
    assert len(stats.visit_counts) == len(env.catalog)
    assert stats.model_evaluations > 0
    # visits only ever land on legal root actions
    mask = env.legal_action_mask()
    assert all(visits == 0 or mask[action] for action, visits in enumerate(stats.visit_counts))


def test_greedy_rollout_solves_easy_instances() -> None:
    config = ACEnvironmentConfig(max_moves=8)
    solved = 0
    for seed in range(6):
        instance = generate_solvable(rank=2, depth=2, seed=seed)
        env = ACEnvironment(instance.presentation, config)
        mcts = _mcts(simulations=64)
        terminated = False
        for _ in range(8):
            _, _, terminated, truncated, _ = env.step(mcts.select_action(env))
            if terminated or truncated:
                break
        solved += terminated
    assert solved >= 5


# --- The search must not be scored as if the agent had played its simulations ---


def _navigation_env() -> ACEnvironment:
    """A navigation-mode env whose start (distance 2) and neighbours (1) are annotated."""
    start = generate_solvable(rank=2, depth=2, seed=3).presentation
    potentials = {start.content_hash: 2}
    for move in moveset_catalog("strict-ac", 2).moves:
        potentials.setdefault(move.apply(start).content_hash, 1)
    return ACEnvironment(
        start,
        ACEnvironmentConfig(
            max_moves=8,
            reward_mode="navigation",
            alpha=0.5,
            reward_config=RewardConfig(alpha_initial=0.5),
        ),
        StateEncoder(16),
        potentials=potentials,
    )


def test_search_leaves_the_navigation_reward_untouched() -> None:
    env = _navigation_env()
    _mcts(simulations=32).search(env)
    stats = env.navigation_episode_stats()
    # The search stepped the env dozens of times; the episode played none of them.
    assert (stats.length, stats.revisit_count) == (0, 0)
    assert stats.min_distance_reached == stats.start_distance
    assert not stats.success


def test_the_move_after_a_search_is_scored_from_the_agents_own_position() -> None:
    env = _navigation_env()
    _, _, _, _, info = env.step(_mcts(simulations=32).select_action(env))
    components = info["reward_components"]
    # Distances are read at the agent's node, not wherever the deepest simulation
    # ended -- otherwise a step toward the goal can be shaped as a step away from it.
    assert components.distance_before == 2
    assert components.reward_shaping == pytest.approx(0.5 * (2 - components.distance_after))
    # And the revisit fee is charged for nodes the agent visited, not the search.
    assert components.reward_revisit_fee == 0.0


class _PeakedModel:
    """A stub model that puts almost all prior mass on one action."""

    def __init__(self, favoured: int, actions: int) -> None:
        self._favoured = favoured
        self._actions = actions

    def apply(self, encoding, action_count):  # type: ignore[no-untyped-def]
        logits = np.full(action_count, -10.0)
        logits[self._favoured] = 10.0
        return PolicyValueOutput(logits=logits, value=0.0, success=0.0, progress=0.0)


def test_min_max_stats_normalizes_only_once_a_range_exists() -> None:
    bounds = _MinMaxStats()
    assert bounds.normalize(7.0) == 7.0  # nothing seen: identity, not a divide by zero
    bounds.update(3.0)
    assert bounds.normalize(3.0) == 3.0  # a single value is still no range
    bounds.update(-1.0)
    assert bounds.normalize(-1.0) == 0.0
    assert bounds.normalize(3.0) == 1.0
    assert bounds.normalize(1.0) == pytest.approx(0.5)


def test_the_first_simulation_follows_the_prior_not_the_action_order() -> None:
    """With `sqrt(total)` every score is 0 before any visit, so index order decided it.

    That threw away the policy on the first simulation of every search -- and on a
    warm-started run the policy is the most reliable thing the search has.
    """
    instance = generate_solvable(rank=2, depth=2, seed=2)
    env = ACEnvironment(instance.presentation, ACEnvironmentConfig(max_moves=8), StateEncoder(16))
    mask = env.legal_action_mask()
    favoured = max(idx for idx, legal in enumerate(mask) if legal)  # not index 0

    stats = PUCTMCTS(
        _PeakedModel(favoured, len(env.catalog)), StateEncoder(16), PUCTConfig(simulations=1)
    ).search(env)

    assert sum(stats.visit_counts) == 1
    assert stats.visit_counts[favoured] == 1


def test_selection_is_invariant_to_an_affine_rescale_of_the_rewards() -> None:
    """Normalized values make `c_puct` mean the same thing at any reward scale.

    Unnormalized, a mean action value spanning tens swamps a prior term bounded by
    `c_puct`, and the search degenerates to greedy on one-sample estimates.
    """
    instance = generate_solvable(rank=2, depth=2, seed=2)

    def counts(goal_reward: float) -> tuple[int, ...]:
        env = ACEnvironment(
            instance.presentation,
            ACEnvironmentConfig(max_moves=8, goal_reward=goal_reward),
            StateEncoder(16),
        )
        return _mcts(simulations=32).search(env).visit_counts

    assert counts(1.0) == counts(100.0)


def test_search_is_deterministic() -> None:
    instance = generate_solvable(rank=2, depth=2, seed=2)
    config = ACEnvironmentConfig(max_moves=8)
    first = _mcts(simulations=32).search(ACEnvironment(instance.presentation, config))
    second = _mcts(simulations=32).search(ACEnvironment(instance.presentation, config))
    assert first.visit_counts == second.visit_counts
