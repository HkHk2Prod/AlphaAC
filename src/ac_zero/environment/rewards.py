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
# - "descent": the objective is to shorten the presentation by at least one, and
#   the goal is that first length reduction (see env `_is_goal`). Every non-goal
#   step scores `-1`; the goal step scores `(D - 1) ** N`, where `D` is the number
#   of available moves (the full action catalog) and `N` is the presentation's
#   known minimal length-descent distance (its `descent_distance` annotation).
#   The exponential goal payoff offsets the `-1` a random policy accrues while
#   searching `N` deep with branching factor `D`, keeping the optimal descent the
#   uniquely rational target rather than an early bail-out.
REWARD_MODES = ("length_reduction", "sparse_goal", "length_reduction_and_goal", "descent")


@dataclass(frozen=True, slots=True)
class RewardSignal:
    """Quantities available to a reward strategy after one environment step."""

    previous_best_length: int
    new_best_length: int
    goal_reached: bool
    goal_reward: float
    # Populated only for the "descent" mode: `available_moves` is D (the catalog
    # size) and `descent_distance` is N (the known fewest moves that shorten the
    # start presentation). They are unused by the length/goal modes.
    available_moves: int = 0
    descent_distance: int | None = None


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
    if mode == "descent":
        return _descent_reward(signal)
    raise ValueError(f"unknown reward mode {mode!r}")


def _descent_reward(signal: RewardSignal) -> float:
    """Score the descent objective: `-1` off-goal, `(D - 1) ** N` on the descent."""
    if not signal.goal_reached:
        return -1.0
    if signal.descent_distance is None:
        raise ValueError("descent reward requires a known descent_distance (N)")
    return float((signal.available_moves - 1) ** signal.descent_distance)
