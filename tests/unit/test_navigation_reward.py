import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.environment.navigation_reward import (
    AlphaUpdater,
    EpisodeStats,
    RewardComponents,
    RewardComputer,
    RewardConfig,
)
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import InvertRelatorMove, MultiplyRelatorsMove


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
    components = computer.step("n", distance_before=5, distance_after=3, reached_destination=False)
    assert components.reward_shaping == pytest.approx(0.5 * 2)
    assert components.reward_shaping > 0.0


def test_moving_farther_gives_negative_shaping() -> None:
    computer = _computer()
    computer.start_episode(alpha=0.5, start_node="s", start_distance=5)
    components = computer.step("n", distance_before=3, distance_after=5, reached_destination=False)
    assert components.reward_shaping == pytest.approx(0.5 * -2)
    assert components.reward_shaping < 0.0


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


# --- EpisodeStats + AlphaUpdater behavior ---


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


def test_recovery_ema_falls_back_to_success_ema_on_a_pre_existing_snapshot() -> None:
    """Checkpoints written before recovery_ema existed must not read as never-succeeded."""
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    updater.observe(_stats(progress=0.9, success=True))
    snapshot = updater.state_dict()
    del snapshot["recovery_ema"]

    resumed = AlphaUpdater(RewardConfig(alpha_initial=0.3))
    resumed.load_state_dict(snapshot)
    assert resumed.recovery_ema == pytest.approx(resumed.success_ema)


def test_load_state_dict_lifts_an_alpha_below_the_current_floor() -> None:
    """The floor rose from 1e-3 to 0.05; live checkpoints hold the old annealed value."""
    resumed = AlphaUpdater(RewardConfig(alpha_min=0.05))
    resumed.load_state_dict(
        {"alpha": 0.001, "progress_ema": 0.9, "success_ema": 0.8, "seeded": True}
    )
    assert resumed.alpha == pytest.approx(0.05)


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


# --- ACEnvironment integration for the "navigation" reward mode ---
#
# Two moves from (x1 x2, x2): InvertRelator(1) then MultiplyRelators(0, 1) reach a
# signed permuted basis (the destination). Annotate P0 at distance 5, P1 at 3; the
# goal P2 is at distance 0 by definition.
_P0 = BalancedPresentation.from_letters(2, [[1, 2], [2]])
_MOVE_A, _MOVE_B = InvertRelatorMove(1), MultiplyRelatorsMove(0, 1)
_P1 = _MOVE_A.apply(_P0)


def _navigation_env(alpha: float = 0.5) -> ACEnvironment:
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
        potentials={_P0.content_hash: 5, _P1.content_hash: 3},
    )


def test_navigation_env_scores_destination_by_start_distance() -> None:
    catalog = ActionCatalog(2)
    env = _navigation_env(alpha=0.5)
    _, reward_a, term_a, _, info_a = env.step(catalog.action_id(_MOVE_A))
    _, reward_b, term_b, _, info_b = env.step(catalog.action_id(_MOVE_B))
    assert not term_a and term_b
    comps_b: RewardComponents = info_b["reward_components"]
    # Destination bonus is scale * L0 = 1.0 * 5, independent of the local step.
    assert comps_b.reward_destination == pytest.approx(5.0)
    # Shaping is alpha * distance_progress: 0.5*(5-3) leaving, 0.5*(3-0) arriving.
    assert info_a["reward_components"].reward_shaping == pytest.approx(0.5 * 2)
    assert comps_b.reward_shaping == pytest.approx(0.5 * 3)
    assert reward_a == pytest.approx(0.5 * 2 - 0.01)
    assert reward_b == pytest.approx(5.0 + 0.5 * 3 - 0.01)


def test_navigation_env_defers_shaping_across_off_graph_nodes() -> None:
    catalog = ActionCatalog(2)
    # P1 is off the annotated graph, so leaving P0 earns zero shaping; the descent
    # from the exit distance (5) is credited in full only when the goal (distance 0)
    # is reached -- the same deferred crediting the potential reward uses, and no
    # length proxy is invented for the off-graph node.
    env = ACEnvironment(
        _P0,
        ACEnvironmentConfig(
            max_moves=4,
            mask_noops=False,
            reward_mode="navigation",
            goal_mode="signed_permuted_basis",
            alpha=0.5,
            reward_config=RewardConfig(destination_reward_scale=1.0, move_fee_scale=0.01),
        ),
        potentials={_P0.content_hash: 5},
    )
    _, _, _, _, info_a = env.step(catalog.action_id(_MOVE_A))
    _, _, terminated, _, info_b = env.step(catalog.action_id(_MOVE_B))
    assert terminated
    assert info_a["reward_components"].reward_shaping == 0.0
    assert info_b["reward_components"].reward_shaping == pytest.approx(0.5 * 5)
    assert info_b["reward_components"].reward_destination == pytest.approx(5.0)
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
    assert stats.start_distance == 5
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


def test_load_state_dict_clamps_alpha_to_bounds() -> None:
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3, alpha_max=0.5))
    # A snapshot taken under looser bounds must not reintroduce an out-of-range alpha.
    updater.load_state_dict({"alpha": 2.0, "progress_ema": 0.4, "success_ema": 0.4, "seeded": True})
    assert updater.alpha == pytest.approx(0.5)
