from __future__ import annotations

from dataclasses import dataclass

# Reward shaping is decoupled from `step` so an episode's objective can be
# chosen per run instead of being hardwired to dense length reduction.
#
# - "length_reduction": dense telescoping signal `prev_best - new_best`. The
#   per-episode return equals the total length reduction achieved. This signal
#   saturates at the minimum total length, which a non-goal presentation can
#   also reach (e.g. duplicate or wrongly-ordered single generators), so it does
#   not, on its own, distinguish a solved instance from a merely short one.
# - "sparse_goal": `goal_reward` on the terminal goal step and nothing else.
# - "length_reduction_and_goal": dense reduction plus a `goal_reward` bonus on
#   the goal step, so reaching the goal is strictly the unique optimum.
# - "potential": potential-based shaping toward the trivial group. The potential
#   Phi(s) is the presentation's distance to origin (its `distance_to_origin`
#   annotation). While both endpoints of a step are in the annotated region the
#   step scores `Phi(prev) - Phi(next)` -- positive when it steps closer to the
#   trivial group. Steps that land in the unannotated region score zero; the
#   environment remembers the potential at the exit point and, on re-entry, credits
#   the whole `Phi(exit) - Phi(entry)` change at once (the goal counts as a known
#   `Phi = 0` entry). Deferring that credit to the later re-entry step is what makes
#   the discounted return account for the time spent off-graph. A `goal_reward`
#   bonus is added on the goal step. Undiscounted the potential telescopes to the
#   start potential regardless of path length, so shorter paths are preferred only
#   once the return is discounted (see `TrainingPipelineConfig.gamma`).
# "navigation": adaptive distance-shaping toward the destination, scored by the
#   stateful `ac_zero.environment.navigation_reward` subsystem rather than the
#   pure `step_reward` below. A terminal bonus proportional to the start-to-goal
#   distance, an alpha-weighted distance-reduction shaping term (alpha retuned
#   between episodes), a flat move fee, and a per-episode revisit fee. Because it
#   carries within-episode state (the visited set, the running minimum distance),
#   the environment drives a `RewardComputer` directly for this mode.
REWARD_MODES = (
    "length_reduction",
    "sparse_goal",
    "length_reduction_and_goal",
    "potential",
    "navigation",
)


@dataclass(frozen=True, slots=True)
class RewardSignal:
    """Quantities available to a reward strategy after one environment step."""

    previous_best_length: int
    new_best_length: int
    goal_reached: bool
    goal_reward: float
    # Change in potential credited by this step (see the "potential" mode above);
    # the environment computes it, tracking the off-graph excursion. Zero for
    # other modes.
    potential_delta: float = 0.0


def step_reward(mode: str, signal: RewardSignal) -> float:
    """Score one transition under the configured reward `mode`."""
    reduction = float(signal.previous_best_length - signal.new_best_length)
    bonus = signal.goal_reward if signal.goal_reached else 0.0
    if mode == "length_reduction":
        return reduction
    if mode == "sparse_goal":
        return bonus
    if mode == "length_reduction_and_goal":
        return reduction + bonus
    if mode == "potential":
        return signal.potential_delta + bonus
    raise ValueError(f"unknown reward mode {mode!r}")
