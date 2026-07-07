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


def _potential_signal(delta: float, *, goal: bool) -> RewardSignal:
    return RewardSignal(
        previous_best_length=0,
        new_best_length=0,
        goal_reached=goal,
        goal_reward=5.0,
        potential_delta=delta,
    )


def test_potential_scores_the_credited_delta_plus_goal_bonus() -> None:
    # Off-goal the reward is just the credited potential change...
    assert step_reward("potential", _potential_signal(3.0, goal=False)) == 3.0
    assert step_reward("potential", _potential_signal(-3.0, goal=False)) == -3.0
    # ...and the goal step adds the bonus on top of the final descent.
    assert step_reward("potential", _potential_signal(4.0, goal=True)) == 9.0


# Two moves from (x1 x2, x2): InvertRelator(1) then MultiplyRelators(0, 1) reach a
# signed permuted basis at minimum length -- the fixture the reward tests step over.
_P0 = BalancedPresentation.from_letters(2, [[1, 2], [2]])
_MOVE_A, _MOVE_B = InvertRelatorMove(1), MultiplyRelatorsMove(0, 1)
_P1 = _MOVE_A.apply(_P0)
_P2 = _MOVE_B.apply(_P1)


def _potential_env(
    potentials: dict[str, int], *, goal_mode: str = "exact_standard"
) -> ACEnvironment:
    return ACEnvironment(
        _P0,
        ACEnvironmentConfig(
            max_moves=4, mask_noops=False, reward_mode="potential", goal_mode=goal_mode
        ),
        potentials=potentials,
    )


def test_potential_scores_the_drop_between_annotated_states() -> None:
    catalog = ActionCatalog(2)
    env = _potential_env({_P0.content_hash: 5, _P1.content_hash: 3, _P2.content_hash: 1})
    _, reward_a, _, _, _ = env.step(catalog.action_id(_MOVE_A))
    _, reward_b, _, _, _ = env.step(catalog.action_id(_MOVE_B))
    assert reward_a == pytest.approx(5.0 - 3.0)
    assert reward_b == pytest.approx(3.0 - 1.0)


def test_potential_defers_credit_across_an_unannotated_excursion() -> None:
    catalog = ActionCatalog(2)
    # P1 is off the annotated graph, so leaving earns nothing and the exit
    # potential (5) is held until P2 re-enters the known region, crediting 5 - 1.
    env = _potential_env({_P0.content_hash: 5, _P2.content_hash: 1})
    _, reward_a, _, _, _ = env.step(catalog.action_id(_MOVE_A))
    _, reward_b, _, _, _ = env.step(catalog.action_id(_MOVE_B))
    assert reward_a == 0.0
    assert reward_b == pytest.approx(5.0 - 1.0)


def test_potential_credits_full_descent_on_reaching_the_goal() -> None:
    catalog = ActionCatalog(2)
    # P2 is the signed-permuted-basis goal (known potential 0) even though it is not
    # in `potentials`; the off-graph exit at 5 is credited in full, plus the bonus.
    env = _potential_env({_P0.content_hash: 5}, goal_mode="signed_permuted_basis")
    env.step(catalog.action_id(_MOVE_A))
    _, reward_b, terminated, _, _ = env.step(catalog.action_id(_MOVE_B))
    assert terminated
    assert reward_b == pytest.approx(5.0 + 1.0)  # 5 - 0 descent + default goal_reward


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
