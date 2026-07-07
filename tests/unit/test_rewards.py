import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.rewards import RewardSignal, step_reward
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import InvertRelatorMove, MultiplyRelatorsMove


def _signal(reduction: int, *, goal: bool) -> RewardSignal:
    return RewardSignal(
        previous_best_length=reduction,
        new_best_length=0,
        goal_reached=goal,
        goal_reward=5.0,
    )


def test_length_reduction_ignores_goal_bonus() -> None:
    assert step_reward("length_reduction", _signal(3, goal=True)) == 3.0
    assert step_reward("length_reduction", _signal(3, goal=False)) == 3.0


def test_sparse_goal_only_pays_on_goal() -> None:
    assert step_reward("sparse_goal", _signal(3, goal=True)) == 5.0
    assert step_reward("sparse_goal", _signal(3, goal=False)) == 0.0


def test_combined_adds_bonus_only_at_goal() -> None:
    assert step_reward("length_reduction_and_goal", _signal(3, goal=True)) == 8.0
    assert step_reward("length_reduction_and_goal", _signal(3, goal=False)) == 3.0


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reward mode"):
        step_reward("bogus", _signal(0, goal=False))


def _reach_signed_basis(reward_mode: str) -> float:
    # InvertRelator then MultiplyRelators turns (x1 x2, x2) into (x1, x2^-1),
    # a signed permuted basis reached at the minimum total length.
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    env = ACEnvironment(
        pres,
        ACEnvironmentConfig(
            max_moves=2,
            mask_noops=False,
            goal_mode="signed_permuted_basis",
            reward_mode=reward_mode,
            goal_reward=1.0,
        ),
    )
    catalog = ActionCatalog(2)
    env.step(catalog.action_id(InvertRelatorMove(1)))
    _, reward, terminated, _, _ = env.step(catalog.action_id(MultiplyRelatorsMove(0, 1)))
    assert terminated
    return reward


def test_goal_step_rewards_more_than_length_alone() -> None:
    length_only = _reach_signed_basis("length_reduction")
    with_goal = _reach_signed_basis("length_reduction_and_goal")
    assert with_goal == pytest.approx(length_only + 1.0)
