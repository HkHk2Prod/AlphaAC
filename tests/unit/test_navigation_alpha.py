"""The `AlphaUpdater` rule: how the shaping weight moves, and what pins it.

`RewardComputer` scoring lives in `test_navigation_reward.py`.
"""

import pytest

from ac_zero.environment.navigation_reward import AlphaUpdater, EpisodeStats, RewardConfig


def _stats(progress: float, success: bool, start_distance: int = 10) -> EpisodeStats:
    min_reached = round(start_distance * (1.0 - progress))
    return EpisodeStats(
        start_distance=start_distance,
        min_distance_reached=min_reached,
        final_distance=min_reached,
        success=success,
        length=1,
        revisit_count=0,
        alpha=0.3,
        destination_reward=0.0,
        shaping_reward=0.0,
        move_fee=0.0,
        revisit_fee=0.0,
        total_reward=0.0,
    )


# --- The four-way update rule and its bounds ---


def test_alpha_increases_when_progress_is_low() -> None:
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3, progress_low=0.3, increase_factor=1.1))
    updater.observe(_stats(progress=0.1, success=False))
    updater.advance()
    assert updater.alpha == pytest.approx(0.3 * 1.1)


def test_alpha_decreases_when_progress_high_but_success_low() -> None:
    config = RewardConfig(
        alpha_initial=0.3, progress_high=0.8, success_target=0.4, decrease_factor=0.8
    )
    updater = AlphaUpdater(config)
    updater.observe(_stats(progress=0.9, success=False))
    updater.advance()
    assert updater.alpha == pytest.approx(0.3 * 0.8)


def test_alpha_anneals_once_success_is_high() -> None:
    config = RewardConfig(alpha_initial=0.3, success_good=0.7, anneal_factor=0.95)
    updater = AlphaUpdater(config)
    # First (seeding) episode is a success -> success_ema = 1.0 > success_good.
    updater.observe(_stats(progress=0.85, success=True))
    updater.advance()
    assert updater.alpha == pytest.approx(0.3 * 0.95)


def test_alpha_updater_exposes_the_running_emas() -> None:
    updater = AlphaUpdater(RewardConfig(ema_rate=0.5))
    updater.observe(_stats(progress=0.5, success=True))
    assert updater.progress_ema == pytest.approx(0.5)
    assert updater.success_ema == pytest.approx(1.0)
    updater.observe(_stats(progress=0.1, success=False))
    assert updater.progress_ema == pytest.approx(0.5 * 0.5 + 0.5 * 0.1)
    assert updater.success_ema == pytest.approx(0.5 * 1.0 + 0.5 * 0.0)


def test_alpha_is_capped_at_alpha_max_under_a_persistent_stall() -> None:
    # A cold-start policy that never makes progress used to ramp alpha as
    # 0.3 * 1.1^n without bound, blowing up the shaping reward and the PPO loss.
    config = RewardConfig(alpha_initial=0.3, alpha_max=2.0, progress_low=0.3, increase_factor=1.1)
    updater = AlphaUpdater(config)
    for _ in range(500):
        updater.observe(_stats(progress=0.0, success=False))
        updater.advance()
    assert updater.alpha == pytest.approx(2.0)


def test_alpha_is_floored_at_alpha_min_under_a_persistent_anneal() -> None:
    config = RewardConfig(alpha_initial=0.3, alpha_min=0.05, success_good=0.7, anneal_factor=0.95)
    updater = AlphaUpdater(config)
    for _ in range(500):
        updater.observe(_stats(progress=0.9, success=True))
        updater.advance()
    assert updater.alpha == pytest.approx(0.05)


# --- Rate limiting: the actuator must not outrun the sensor ---


def test_advance_moves_alpha_only_every_nth_iteration() -> None:
    config = RewardConfig(
        alpha_initial=0.3, progress_low=0.3, increase_factor=1.1, alpha_update_every_iterations=3
    )
    updater = AlphaUpdater(config)
    updater.observe(_stats(progress=0.0, success=False))
    updater.advance()
    updater.advance()
    assert updater.alpha == pytest.approx(0.3)  # held for the first two iterations
    updater.advance()
    assert updater.alpha == pytest.approx(0.3 * 1.1)  # moved once on the third


def test_the_iteration_counter_resets_after_each_move() -> None:
    config = RewardConfig(
        alpha_initial=0.3, progress_low=0.3, increase_factor=1.1, alpha_update_every_iterations=2
    )
    updater = AlphaUpdater(config)
    updater.observe(_stats(progress=0.0, success=False))
    for _ in range(6):
        updater.advance()
    # Six iterations at one move per two iterations -> exactly three moves.
    assert updater.alpha == pytest.approx(0.3 * 1.1**3)


# --- The recovery branch: a regression is not a stall ---


def test_a_regression_from_a_working_policy_holds_alpha_instead_of_ramping() -> None:
    """The bug this guards: a dip used to ramp alpha 1000x and destroy the policy.

    A run that has been succeeding drives ``recovery_ema`` high. When progress
    then collapses, the increase branch must not fire -- ramping the shaping back
    up mid-collapse changes the reward scale under a value function fit to the
    annealed one, and the resulting zero-success state pins alpha at its ceiling
    permanently.
    """
    config = RewardConfig(
        alpha_initial=0.3,
        alpha_min=0.05,
        progress_low=0.3,
        increase_factor=1.1,
        success_target=0.4,
        recovery_ema_rate=0.01,
    )
    updater = AlphaUpdater(config)
    for _ in range(400):  # a long run of success lifts recovery_ema well above target
        updater.observe(_stats(progress=1.0, success=True))
    assert updater.recovery_ema > config.success_target

    # Then it collapses. Alpha may still fall (progress_ema decays through the
    # "progress but no success" branch), but it must never be ramped back up --
    # that ramp is what invalidates the value function and makes the dip terminal.
    trace = []
    for _ in range(60):
        updater.observe(_stats(progress=0.0, success=False))
        trace.append(updater.advance())
    assert updater.progress_ema < config.progress_low  # squarely in the stall branch
    assert updater.recovery_ema > config.success_target  # but still credited as recovering
    assert trace == sorted(trace, reverse=True)  # monotone non-increasing: no ramp
    assert updater.alpha == pytest.approx(min(trace))


def test_ramping_resumes_once_the_recovery_ema_decays() -> None:
    """The hold is self-limiting: a policy that never comes back gets shaping again."""
    config = RewardConfig(
        alpha_initial=0.3,
        progress_low=0.3,
        increase_factor=1.1,
        success_target=0.4,
        recovery_ema_rate=0.1,  # short memory so the decay is quick to exercise
    )
    updater = AlphaUpdater(config)
    for _ in range(100):
        updater.observe(_stats(progress=1.0, success=True))
    for _ in range(200):
        updater.observe(_stats(progress=0.0, success=False))
        updater.advance()
    assert updater.recovery_ema < config.success_target
    assert updater.alpha > 0.3  # ramping resumed on its own


def test_a_cold_start_stall_still_ramps_immediately() -> None:
    """A policy that never succeeded has no recovery credit, so shaping ramps at once."""
    config = RewardConfig(alpha_initial=0.3, progress_low=0.3, increase_factor=1.1)
    updater = AlphaUpdater(config)
    updater.observe(_stats(progress=0.0, success=False))
    updater.advance()
    assert updater.alpha == pytest.approx(0.3 * 1.1)


# --- Cross-run continuity: alpha state survives a checkpoint round-trip ---


def test_alpha_state_round_trips_through_state_dict() -> None:
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3, increase_factor=1.1))
    updater.observe(_stats(progress=0.1, success=False))  # seeds EMAs
    updater.advance()  # raises alpha
    snapshot = updater.state_dict()

    resumed = AlphaUpdater(RewardConfig(alpha_initial=0.3, increase_factor=1.1))
    resumed.load_state_dict(snapshot)
    assert resumed.alpha == pytest.approx(updater.alpha)
    assert resumed.progress_ema == pytest.approx(updater.progress_ema)
    assert resumed.success_ema == pytest.approx(updater.success_ema)
    assert resumed.recovery_ema == pytest.approx(updater.recovery_ema)


def test_resumed_alpha_continues_instead_of_reseeding() -> None:
    # A fresh updater re-seeds its EMAs from the first episode it sees; a resumed
    # one must instead blend that episode into the restored EMAs.
    original = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    original.observe(_stats(progress=0.5, success=True))
    resumed = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    resumed.load_state_dict(original.state_dict())

    fresh = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    resumed.observe(_stats(progress=0.9, success=True))
    fresh.observe(_stats(progress=0.9, success=True))
    assert resumed.progress_ema != pytest.approx(fresh.progress_ema)


def test_recovery_ema_falls_back_to_success_ema_on_a_pre_existing_snapshot() -> None:
    """Checkpoints written before recovery_ema existed must not read as never-succeeded."""
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    updater.observe(_stats(progress=0.9, success=True))
    snapshot = updater.state_dict()
    del snapshot["recovery_ema"]

    resumed = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    resumed.load_state_dict(snapshot)
    assert resumed.recovery_ema == pytest.approx(resumed.success_ema)


def test_load_state_dict_clamps_alpha_to_bounds() -> None:
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3, alpha_max=0.5))
    # A snapshot taken under looser bounds must not reintroduce an out-of-range alpha.
    updater.load_state_dict({"alpha": 2.0, "progress_ema": 0.4, "success_ema": 0.4, "seeded": True})
    assert updater.alpha == pytest.approx(0.5)


def test_load_state_dict_lifts_an_alpha_below_the_current_floor() -> None:
    """The floor rose from 1e-3 to 0.05; live checkpoints hold the old annealed value."""
    resumed = AlphaUpdater(RewardConfig(alpha_min=0.05))
    resumed.load_state_dict(
        {"alpha": 0.001, "progress_ema": 0.9, "success_ema": 0.8, "seeded": True}
    )
    assert resumed.alpha == pytest.approx(0.05)


# --- Config validation ---


def test_reward_config_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError, match="progress_low"):
        RewardConfig(progress_low=0.9, progress_high=0.2).validate()
    with pytest.raises(ValueError, match="ema_rate"):
        RewardConfig(ema_rate=0.0).validate()
    with pytest.raises(ValueError, match="recovery_ema_rate"):
        RewardConfig(recovery_ema_rate=0.0).validate()
    with pytest.raises(ValueError, match="alpha_update_every_iterations"):
        RewardConfig(alpha_update_every_iterations=0).validate()


def test_reward_config_rejects_bad_alpha_bounds() -> None:
    with pytest.raises(ValueError, match="alpha_min"):
        RewardConfig(alpha_min=1.0, alpha_max=0.5).validate()
    with pytest.raises(ValueError, match="alpha_initial must lie"):
        RewardConfig(alpha_initial=5.0, alpha_min=0.1, alpha_max=1.0).validate()
