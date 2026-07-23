"""`ACEnvironment` end to end under the "navigation" reward mode.

Two moves from (x1 x2, x2): InvertRelator(1) then MultiplyRelators(0, 1) reach a
signed permuted basis (the destination). The goal P2 is at distance 0 by
definition, so the only distances consistent with that path are P1 at 1 and P0 at
2: shortest-path distance is 1-Lipschitz over an invertible moveset, and a fixture
that jumps further per move would exercise shaping the real graph never pays.
"""

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.navigation_reward import RewardComponents, RewardConfig
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import InvertRelatorMove, MultiplyRelatorsMove

_P0 = BalancedPresentation.from_letters(2, [[1, 2], [2]])
_MOVE_A, _MOVE_B = InvertRelatorMove(1), MultiplyRelatorsMove(0, 1)
_P1 = _MOVE_A.apply(_P0)


def _navigation_env(alpha: float = 0.5, potentials: dict[str, int] | None = None) -> ACEnvironment:
    return ACEnvironment(
        _P0,
        ACEnvironmentConfig(
            max_moves=4,
            mask_noops=False,
            reward_mode="navigation",
            goal_mode="signed_permuted_basis",
            alpha=alpha,
            reward_config=RewardConfig(
                destination_reward_scale=1.0, move_fee_scale=0.01, revisit_fee_scale=0.02
            ),
        ),
        potentials=potentials
        if potentials is not None
        else {_P0.content_hash: 2, _P1.content_hash: 1},
    )


def test_navigation_env_scores_destination_by_start_distance() -> None:
    catalog = ActionCatalog(2)
    env = _navigation_env(alpha=0.5)
    _, reward_a, term_a, _, info_a = env.step(catalog.action_id(_MOVE_A))
    _, reward_b, term_b, _, info_b = env.step(catalog.action_id(_MOVE_B))
    assert not term_a and term_b
    comps_b: RewardComponents = info_b["reward_components"]
    # Destination bonus is scale * L0 = 1.0 * 2, independent of the local step, and
    # is never clipped: reaching the goal is what the episode is paid for.
    assert comps_b.reward_destination == pytest.approx(2.0)
    # Shaping is alpha * distance_progress: 0.5*(2-1) leaving, 0.5*(1-0) arriving.
    assert info_a["reward_components"].reward_shaping == pytest.approx(0.5)
    assert comps_b.reward_shaping == pytest.approx(0.5)
    assert reward_a == pytest.approx(0.5 - 0.01)
    assert reward_b == pytest.approx(2.0 + 0.5 - 0.01)


def test_navigation_env_prices_a_step_off_the_annotated_graph() -> None:
    catalog = ActionCatalog(2)
    # Only P0 is annotated, so P1 is off the graph and is scored as one step further
    # out (3) rather than free -- leaving must not be cheaper than climbing inside.
    env = _navigation_env(potentials={_P0.content_hash: 2})
    _, _, _, _, info_a = env.step(catalog.action_id(_MOVE_A))
    _, _, terminated, _, info_b = env.step(catalog.action_id(_MOVE_B))
    assert terminated
    leaving: RewardComponents = info_a["reward_components"]
    arriving: RewardComponents = info_b["reward_components"]
    assert leaving.distance_after == 3
    assert leaving.reward_shaping == pytest.approx(-0.5)
    # Re-entry is scored against the inflated anchor (3 -> 0), so it hands the fee
    # back -- but only one step's worth. Paying the full 0.5*3 would make this single
    # move worth three descents purely because it was the one that came back, and the
    # network has no way to see why. The excursion nets zero shaping: it cost a step
    # out and earned a step back.
    assert arriving.distance_progress == 3
    assert arriving.reward_shaping == pytest.approx(0.5)
    assert leaving.reward_shaping + arriving.reward_shaping == pytest.approx(0.0)
    # Reaching the goal is still paid in full by the destination bonus, which the
    # shaping cap does not touch.
    assert arriving.reward_destination == pytest.approx(2.0)
    # Closest known approach is the goal, so progress is complete.
    assert env.navigation_episode_stats().progress_rate == pytest.approx(1.0)


def test_navigation_env_holds_alpha_constant_across_the_episode() -> None:
    catalog = ActionCatalog(2)
    env = _navigation_env(alpha=0.7)
    _, _, _, _, info_a = env.step(catalog.action_id(_MOVE_A))
    _, _, _, _, info_b = env.step(catalog.action_id(_MOVE_B))
    assert info_a["reward_components"].alpha == 0.7
    assert info_b["reward_components"].alpha == 0.7


def test_navigation_env_reports_episode_stats() -> None:
    catalog = ActionCatalog(2)
    env = _navigation_env()
    env.step(catalog.action_id(_MOVE_A))
    env.step(catalog.action_id(_MOVE_B))
    stats = env.navigation_episode_stats()
    assert stats.start_distance == 2
    assert stats.min_distance_reached == 0
    assert stats.success is True
    assert stats.length == 2
    assert stats.revisit_count == 0
    assert stats.progress_rate == pytest.approx(1.0)


def test_navigation_env_penalizes_stepping_back_to_the_start() -> None:
    catalog = ActionCatalog(2)
    env = _navigation_env()
    # InvertRelator(1) is its own inverse: applying it twice returns to P0 (start).
    env.step(catalog.action_id(_MOVE_A))
    _, _, _, _, info_back = env.step(catalog.action_id(_MOVE_A))
    assert info_back["reward_components"].reward_revisit_fee == pytest.approx(-0.02)


def test_navigation_env_resets_visited_set_on_reset() -> None:
    catalog = ActionCatalog(2)
    env = _navigation_env()
    env.step(catalog.action_id(_MOVE_A))
    env.step(catalog.action_id(_MOVE_A))  # revisits P0
    env.reset()
    stats = env.navigation_episode_stats()
    assert stats.revisit_count == 0
    _, _, _, _, info = env.step(catalog.action_id(_MOVE_A))
    assert info["reward_components"].reward_revisit_fee == 0.0
