from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.registry import create_trainable_model
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


def test_search_is_deterministic() -> None:
    instance = generate_solvable(rank=2, depth=2, seed=2)
    config = ACEnvironmentConfig(max_moves=8)
    first = _mcts(simulations=32).search(ACEnvironment(instance.presentation, config))
    second = _mcts(simulations=32).search(ACEnvironment(instance.presentation, config))
    assert first.visit_counts == second.visit_counts
