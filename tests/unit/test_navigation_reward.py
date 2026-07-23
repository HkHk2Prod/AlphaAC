"""How `RewardComputer` scores one transition, and how it rewinds for a search.

The `AlphaUpdater` rule lives in `test_navigation_alpha.py`, and the environment's
end-to-end navigation episodes in `test_navigation_env.py`.
"""

import pytest

from ac_zero.environment.navigation_reward import (
    AlphaUpdater,
    EpisodeStats,
    RewardComputer,
    RewardConfig,
)


def _computer(**overrides: float) -> RewardComputer:
    return RewardComputer(RewardConfig(**overrides))


# --- Sanity check 1 & 2: alpha is constant within an episode, updates only after ---


def test_alpha_is_constant_within_an_episode() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.5, start_node="a", start_distance=4)
    first = computer.step("b", distance_before=4, distance_after=3, reached_destination=False)
    second = computer.step("c", distance_before=3, distance_after=2, reached_destination=False)
    assert first.alpha == 0.5
    assert second.alpha == 0.5
    assert computer.alpha == 0.5


def test_alpha_updates_only_after_the_episode_not_during() -> None:
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    computer = _computer()
    computer.start_episode(alpha=updater.alpha, start_node="a", start_distance=4)
    # Stepping the computer never touches the updater's alpha.
    computer.step("b", distance_before=4, distance_after=3, reached_destination=False)
    assert updater.alpha == 0.3
    # Observing folds the EMAs but still holds alpha; only advancing moves it.
    updater.observe(computer.episode_stats())
    assert updater.alpha == 0.3
    updater.advance()
    assert updater.alpha != 0.3


# --- Sanity check 3: farther destination -> larger terminal reward ---


def test_farther_destination_gives_larger_terminal_reward() -> None:
    near = _computer()
    near.start_episode(alpha=0.3, start_node="s", start_distance=2)
    near_reward = near.step("g", distance_before=1, distance_after=0, reached_destination=True)

    far = _computer()
    far.start_episode(alpha=0.3, start_node="s", start_distance=10)
    far_reward = far.step("g", distance_before=1, distance_after=0, reached_destination=True)

    assert far_reward.reward_destination > near_reward.reward_destination
    assert far_reward.reward_destination == pytest.approx(10.0)
    assert near_reward.reward_destination == pytest.approx(2.0)


def test_destination_reward_zero_when_not_reached() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.3, start_node="s", start_distance=5)
    components = computer.step("m", distance_before=5, distance_after=4, reached_destination=False)
    assert components.reward_destination == 0.0


# --- Sanity check 4 & 5: shaping sign follows distance change ---


def test_moving_closer_gives_positive_shaping() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.5, start_node="s", start_distance=5)
    components = computer.step("n", distance_before=5, distance_after=4, reached_destination=False)
    assert components.reward_shaping == pytest.approx(0.5)
    assert components.reward_shaping > 0.0


def test_moving_farther_gives_negative_shaping() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.5, start_node="s", start_distance=5)
    components = computer.step("n", distance_before=4, distance_after=5, reached_destination=False)
    assert components.reward_shaping == pytest.approx(-0.5)
    assert components.reward_shaping < 0.0


def test_shaping_is_capped_at_one_step_of_progress() -> None:
    # A move cannot change true distance by more than one (shortest-path distance is
    # 1-Lipschitz over an invertible moveset), so a larger jump only ever comes from
    # an estimated distance -- re-entering the annotated graph against an anchor the
    # off-graph detour inflated. Paying it in full hands one move the worth of the
    # whole detour.
    computer = _computer()
    computer.start_episode(alpha=0.5, start_node="s", start_distance=5)
    forward = computer.step("n", distance_before=12, distance_after=4, reached_destination=False)
    assert forward.reward_shaping == pytest.approx(0.5)
    # The unclipped change is still reported: shaping is capped, history is not.
    assert forward.distance_progress == 8
    backward = computer.step("m", distance_before=4, distance_after=12, reached_destination=False)
    assert backward.reward_shaping == pytest.approx(-0.5)
    assert backward.distance_progress == -8


def test_max_shaping_progress_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="max_shaping_progress"):
        RewardConfig(max_shaping_progress=0).validate()


def test_unchanged_distance_gives_zero_shaping() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.7, start_node="s", start_distance=5)
    components = computer.step("n", distance_before=4, distance_after=4, reached_destination=False)
    assert components.reward_shaping == 0.0


# --- Sanity check 6: revisiting a node adds the revisit penalty ---


def test_revisiting_a_node_adds_the_revisit_fee() -> None:
    computer = _computer(move_fee_scale=0.01, revisit_fee_scale=0.02)
    computer.start_episode(alpha=0.3, start_node="s", start_distance=3)
    fresh = computer.step("a", distance_before=3, distance_after=2, reached_destination=False)
    revisit = computer.step("a", distance_before=2, distance_after=3, reached_destination=False)
    assert fresh.reward_revisit_fee == 0.0
    assert revisit.reward_revisit_fee == pytest.approx(-0.02)


def test_stepping_back_onto_the_start_node_is_a_revisit() -> None:
    computer = _computer(revisit_fee_scale=0.02)
    computer.start_episode(alpha=0.3, start_node="s", start_distance=1)
    back = computer.step("s", distance_before=1, distance_after=0, reached_destination=False)
    assert back.reward_revisit_fee == pytest.approx(-0.02)


def test_move_fee_applies_to_every_step() -> None:
    computer = _computer(move_fee_scale=0.01)
    computer.start_episode(alpha=0.3, start_node="s", start_distance=3)
    components = computer.step("a", distance_before=3, distance_after=2, reached_destination=False)
    assert components.reward_move_fee == pytest.approx(-0.01)


# --- Sanity check 7: the visited set resets between episodes ---


def test_visited_set_resets_between_episodes() -> None:
    computer = _computer(revisit_fee_scale=0.02)
    computer.start_episode(alpha=0.3, start_node="s", start_distance=3)
    computer.step("a", distance_before=3, distance_after=2, reached_destination=False)
    # New episode: "a" has not been visited *this* episode, so no revisit fee.
    computer.start_episode(alpha=0.3, start_node="s", start_distance=3)
    components = computer.step("a", distance_before=3, distance_after=2, reached_destination=False)
    assert components.reward_revisit_fee == 0.0


# --- Sanity check 8: components sum to the total ---


def test_reward_components_sum_to_total() -> None:
    computer = _computer(move_fee_scale=0.01, revisit_fee_scale=0.02, destination_reward_scale=1.0)
    computer.start_episode(alpha=0.4, start_node="s", start_distance=6)
    computer.step("a", distance_before=6, distance_after=4, reached_destination=False)
    # Revisit "a" and reach the destination in the same step to exercise all four parts.
    components = computer.step("a", distance_before=4, distance_after=0, reached_destination=True)
    parts = (
        components.reward_destination
        + components.reward_shaping
        + components.reward_move_fee
        + components.reward_revisit_fee
    )
    assert parts == pytest.approx(components.reward_total)
    assert components.reward_revisit_fee == pytest.approx(-0.02)
    assert components.reward_destination == pytest.approx(6.0)


# --- The episode aggregate the alpha updater consumes ---


def test_episode_stats_track_progress_and_revisits() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.3, start_node="s", start_distance=10)
    computer.step("a", distance_before=10, distance_after=4, reached_destination=False)
    computer.step("a", distance_before=4, distance_after=7, reached_destination=False)
    stats = computer.episode_stats()
    assert stats.start_distance == 10
    assert stats.min_distance_reached == 4
    assert stats.revisit_count == 1
    assert stats.length == 2
    assert stats.success is False
    assert stats.progress_rate == pytest.approx((10 - 4) / 10)


def test_progress_rate_is_zero_when_starting_at_the_destination() -> None:
    stats = EpisodeStats(
        start_distance=0,
        min_distance_reached=0,
        final_distance=0,
        success=True,
        length=0,
        revisit_count=0,
        alpha=0.3,
        destination_reward=0.0,
        shaping_reward=0.0,
        move_fee=0.0,
        revisit_fee=0.0,
        total_reward=0.0,
    )
    assert stats.progress_rate == 0.0


# --- Search safety: the within-episode state rewinds to a snapshot ---


def test_reward_snapshot_rewinds_the_within_episode_state() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.5, start_node="a", start_distance=4)
    computer.step("b", distance_before=4, distance_after=3, reached_destination=False)
    snapshot = computer.snapshot()
    # Stand in for a search: hypothetical moves that must leave no trace behind.
    computer.step("c", distance_before=3, distance_after=1, reached_destination=True)
    computer.step("b", distance_before=1, distance_after=3, reached_destination=False)
    computer.restore(snapshot)

    stats = computer.episode_stats()
    assert (stats.length, stats.revisit_count) == (1, 0)
    assert stats.min_distance_reached == 3  # not the 1 the rewound steps reached
    assert not stats.success
    assert stats.total_reward == pytest.approx(0.5 * 1 - 0.01)
    # "c" was visited only by the rewound steps, so stepping onto it now is new.
    assert computer.step("c", 3, 2, False).reward_revisit_fee == 0.0
