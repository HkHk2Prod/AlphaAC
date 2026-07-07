import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.rewards import RewardSignal, step_reward
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import InvertRelatorMove, MultiplyRelatorsMove
from ac_zero.moves.universal import UniversalCatalog


def _signal(reduction: int, *, goal: bool) -> RewardSignal:
    return RewardSignal(
        previous_best_length=reduction,
        new_best_length=0,
        goal_reached=goal,
        goal_reward=5.0,
    )


def _descent_signal(*, goal: bool, moves: int = 12, distance: int | None = 3) -> RewardSignal:
    return RewardSignal(
        previous_best_length=0,
        new_best_length=0,
        goal_reached=goal,
        goal_reward=0.0,
        available_moves=moves,
        descent_distance=distance,
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


def test_descent_penalizes_every_non_goal_step() -> None:
    assert step_reward("descent", _descent_signal(goal=False)) == -1.0


def test_descent_goal_pays_branching_to_the_power_of_distance() -> None:
    reward = step_reward("descent", _descent_signal(goal=True, moves=12, distance=3))
    assert reward == (12 - 1) ** 3


def test_descent_goal_needs_a_known_distance() -> None:
    with pytest.raises(ValueError, match="descent_distance"):
        step_reward("descent", _descent_signal(goal=True, distance=None))


def _descent_env(moveset: str = "strict-ac") -> ACEnvironment:
    # (x1 x2, x2^-1): multiplying relator 0 by relator 1 freely reduces x1 x2 x2^-1
    # to x1, dropping the total length from 3 to 2 -- the descent goal in one move.
    pres = BalancedPresentation.from_letters(2, [[1, 2], [-2]], provenance={"descent_distance": 1})
    return ACEnvironment(
        pres, ACEnvironmentConfig(max_moves=4, reward_mode="descent", moveset=moveset)
    )


def test_descent_env_pays_the_goal_move_and_terminates() -> None:
    env = _descent_env()
    catalog = ActionCatalog(2)
    _, reward, terminated, _, _ = env.step(catalog.action_id(MultiplyRelatorsMove(0, 1)))
    assert terminated
    assert reward == (len(catalog) - 1) ** 1


def test_descent_env_penalizes_a_move_that_does_not_shorten() -> None:
    env = _descent_env()
    catalog = ActionCatalog(2)
    # Inverting x2^-1 to x2 keeps the total length at 3, so it is off-goal.
    _, reward, terminated, _, _ = env.step(catalog.action_id(InvertRelatorMove(1)))
    assert not terminated
    assert reward == -1.0


def test_descent_env_pays_by_universal_moveset_size_when_configured() -> None:
    # D in (D - 1)**N is the *configured* move set's size, not always strict-AC's.
    env = _descent_env(moveset="universal")
    universal = UniversalCatalog(2)
    assert len(env.catalog) == len(universal) != len(ActionCatalog(2))
    _, reward, terminated, _, _ = env.step(universal.move_id(MultiplyRelatorsMove(0, 1)))
    assert terminated
    assert reward == (len(universal) - 1) ** 1


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
