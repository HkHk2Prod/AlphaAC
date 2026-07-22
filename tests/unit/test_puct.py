import pytest

from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.navigation_reward import RewardConfig
from ac_zero.models.registry import create_trainable_model
from ac_zero.moves.universal import moveset_catalog
from ac_zero.search.puct import PUCTMCTS, PUCTConfig


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


def test_search_is_deterministic() -> None:
    instance = generate_solvable(rank=2, depth=2, seed=2)
    config = ACEnvironmentConfig(max_moves=8)
    first = _mcts(simulations=32).search(ACEnvironment(instance.presentation, config))
    second = _mcts(simulations=32).search(ACEnvironment(instance.presentation, config))
    assert first.visit_counts == second.visit_counts
