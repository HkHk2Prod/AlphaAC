"""Tests for the distance curriculum: L_max control and frontier success tracking.

Covers the spec's sanity checks 1-16 that are not already exercised by
``test_navigation_reward`` (alpha within-episode constancy) or
``test_instance_source`` (the sampler's L <= L_max restriction and lack of a
lower bound).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ac_zero.environment.navigation_reward import AlphaUpdater, RewardConfig
from ac_zero.training.navigation_curriculum import (
    DistanceCurriculum,
    DistanceCurriculumConfig,
)
from ac_zero.training.pipeline_episodes import episode_distance_and_moves


def _curriculum(**overrides: float) -> DistanceCurriculum:
    return DistanceCurriculum(DistanceCurriculumConfig(**overrides))


# --- Sanity checks 1 & 2: initialization and the lower bound on L_max ---


def test_l_max_defaults_to_two() -> None:
    assert _curriculum().current_L_max() == 2
    assert DistanceCurriculumConfig().L_max_initial == 2


def test_l_max_never_drops_below_two() -> None:
    curriculum = _curriculum(min_frontier_episodes_before_update=3)
    # At L_max = 2 the frontier band is just {2}; feed failures that would lower it.
    for _ in range(10):
        update = curriculum.update(L=2, success=False, L_max_episode=2)
    assert curriculum.current_L_max() == 2
    assert not update.L_max_changed
    # No actual change means the estimator is *not* reset (spec item 11 corollary):
    assert curriculum.frontier_success_count == 10


# --- Sanity check 5: the frontier band is ceil(0.7 * L_max) <= L <= L_max ---


def test_frontier_band_endpoints() -> None:
    curriculum = _curriculum(frontier_fraction=0.7)
    assert curriculum.frontier_lower(10) == 7  # ceil(7.0)
    assert curriculum.frontier_lower(4) == 3  # ceil(2.8)
    assert curriculum.is_frontier(7, 10)
    assert curriculum.is_frontier(10, 10)
    assert not curriculum.is_frontier(6, 10)
    assert not curriculum.is_frontier(11, 10)


# --- Sanity checks 6 & 7: only frontier episodes move the rolling estimate ---


def test_non_frontier_episode_does_not_update_estimator() -> None:
    curriculum = _curriculum()
    update = curriculum.update(L=3, success=True, L_max_episode=10)  # below the band
    assert not update.is_frontier_episode
    assert curriculum.frontier_success_ema is None
    assert curriculum.frontier_success_count == 0


def test_frontier_episode_updates_estimator() -> None:
    curriculum = _curriculum()
    update = curriculum.update(L=8, success=True, L_max_episode=10)  # inside the band
    assert update.is_frontier_episode
    assert curriculum.frontier_success_ema == pytest.approx(1.0)
    assert curriculum.frontier_success_count == 1


def test_frontier_estimator_uses_the_sampled_lmax_not_the_current_one() -> None:
    # L_max_episode is the ceiling active when the episode was sampled; a later
    # change must not retroactively alter frontier membership.
    curriculum = _curriculum(min_frontier_episodes_before_update=1, frontier_success_high=0.75)
    curriculum.update(L=2, success=True, L_max_episode=2)  # bumps L_max to 3
    assert curriculum.current_L_max() == 3
    # An episode sampled under the *old* L_max=2 is still judged against 2.
    update = curriculum.update(L=2, success=True, L_max_episode=2)
    assert update.frontier_lower == 2
    assert update.is_frontier_episode


# --- Sanity checks 8, 9 & 10: hysteresis moves L_max and resets the estimator ---


def test_l_max_increases_after_enough_high_success_frontier_episodes() -> None:
    curriculum = _curriculum(min_frontier_episodes_before_update=3, frontier_success_high=0.75)
    for _ in range(2):
        assert curriculum.update(L=2, success=True, L_max_episode=2).L_max == 2
    final = curriculum.update(L=2, success=True, L_max_episode=2)
    assert final.L_max == 3
    assert final.L_max_changed
    assert final.L_max_change_direction == "increase"
    # Reset after an actual change (sanity check 10).
    assert curriculum.frontier_success_ema is None
    assert curriculum.frontier_success_count == 0
    assert curriculum.episodes_since_lmax_change == 0


def test_l_max_decreases_after_enough_low_success_but_not_below_min() -> None:
    curriculum = _curriculum(
        L_max_initial=5,
        L_max_min=2,
        min_frontier_episodes_before_update=3,
        frontier_success_low=0.25,
        allow_L_max_decrease=True,
    )
    for _ in range(2):
        assert curriculum.update(L=5, success=False, L_max_episode=5).L_max == 5
    final = curriculum.update(L=5, success=False, L_max_episode=5)
    assert final.L_max == 4
    assert final.L_max_change_direction == "decrease"
    assert curriculum.frontier_success_ema is None


def test_l_max_never_decreases_by_default() -> None:
    curriculum = _curriculum(
        L_max_initial=5,
        L_max_min=2,
        min_frontier_episodes_before_update=3,
        frontier_success_low=0.25,
    )
    for _ in range(5):
        result = curriculum.update(L=5, success=False, L_max_episode=5)
        assert result.L_max == 5
        assert result.L_max_change_direction == "none"


# --- Sanity check 11: no reset when L_max is unchanged ---


def test_estimator_not_reset_when_l_max_unchanged() -> None:
    curriculum = _curriculum(min_frontier_episodes_before_update=100)
    for _ in range(5):
        curriculum.update(L=2, success=True, L_max_episode=2)
    assert curriculum.current_L_max() == 2  # min not reached, no change
    assert curriculum.frontier_success_ema == pytest.approx(1.0)
    assert curriculum.frontier_success_count == 5


# --- Sanity checks 12 & 13: max_moves = 3 * L + 6, driven by the sampled L ---


def _source_with_distances(distances: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(potentials=distances)


@pytest.mark.parametrize(("distance", "expected"), [(2, 12), (5, 21), (10, 36)])
def test_max_moves_is_three_l_plus_six(distance: int, expected: int) -> None:
    source = _source_with_distances({"h": distance})
    presentation = SimpleNamespace(content_hash="h")
    L, max_moves = episode_distance_and_moves(source, presentation, unknown_max_moves=512)
    assert L == distance
    assert max_moves == expected


def test_max_moves_depends_on_sampled_distance_not_l_max() -> None:
    # Two problems drawn under the same L_max ceiling but with different sampled L
    # get different horizons -- the horizon tracks L, never L_max.
    source = _source_with_distances({"near": 3, "far": 9})
    near = episode_distance_and_moves(source, SimpleNamespace(content_hash="near"), 512)
    far = episode_distance_and_moves(source, SimpleNamespace(content_hash="far"), 512)
    assert near == (3, 15)
    assert far == (9, 33)


def test_max_moves_falls_back_to_large_cutoff_when_distance_unknown() -> None:
    # A problem off the annotated graph has no known L, so the horizon is the
    # configured large cutoff rather than 3 * L + 6.
    source = _source_with_distances({})  # no distance for this hash
    L, max_moves = episode_distance_and_moves(
        source, SimpleNamespace(content_hash="off_graph"), unknown_max_moves=999
    )
    assert L is None
    assert max_moves == 999


# --- Sanity checks 14 & 16: alpha and L_max are independent and stable ---


def test_alpha_updater_and_curriculum_are_independent() -> None:
    alpha_updater = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    curriculum = _curriculum(min_frontier_episodes_before_update=1)
    # Stepping the curriculum never touches alpha...
    curriculum.update(L=2, success=True, L_max_episode=2)
    assert alpha_updater.alpha == 0.3
    # ...and the curriculum takes a config with no alpha state at all.
    assert not hasattr(curriculum.config, "alpha_initial")
    assert not hasattr(curriculum, "_alpha")


def test_current_l_max_is_stable_between_updates() -> None:
    curriculum = _curriculum(min_frontier_episodes_before_update=1000)
    curriculum.update(L=2, success=True, L_max_episode=2)
    before = curriculum.current_L_max()
    # Read-only queries must not advance the ceiling (constant within an episode).
    curriculum.frontier_lower(2)
    curriculum.is_frontier(2, 2)
    assert curriculum.current_L_max() == before


# --- Config validation ---


def test_config_rejects_l_max_min_below_two() -> None:
    with pytest.raises(ValueError, match="L_max_min"):
        DistanceCurriculumConfig(L_max_min=1).validate()


def test_config_rejects_inverted_success_band() -> None:
    with pytest.raises(ValueError, match="frontier_success"):
        DistanceCurriculumConfig(frontier_success_low=0.9, frontier_success_high=0.2).validate()


# --- Cross-run continuity: L_max state survives a checkpoint round-trip ---


def test_curriculum_state_round_trips_through_state_dict() -> None:
    curriculum = _curriculum(min_frontier_episodes_before_update=1, frontier_success_high=0.75)
    curriculum.update(L=2, success=True, L_max_episode=2)  # advances L_max, tracks frontier
    snapshot = curriculum.state_dict()

    resumed = _curriculum(min_frontier_episodes_before_update=1)
    resumed.load_state_dict(snapshot)
    assert resumed.current_L_max() == curriculum.current_L_max()
    assert resumed.frontier_success_ema == curriculum.frontier_success_ema
    assert resumed.frontier_success_count == curriculum.frontier_success_count
    assert resumed.episodes_since_lmax_change == curriculum.episodes_since_lmax_change


def test_load_state_dict_floors_l_max_at_min() -> None:
    # A snapshot from a run with a lower floor must not resume below this run's min.
    curriculum = _curriculum(L_max_initial=5, L_max_min=5)
    curriculum.load_state_dict(
        {
            "L_max": 2,
            "frontier_success_ema": None,
            "frontier_success_count": 0,
            "episodes_since_lmax_change": 0,
        }
    )
    assert curriculum.current_L_max() == 5
