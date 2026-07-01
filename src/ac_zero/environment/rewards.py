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
REWARD_MODES = ("length_reduction", "sparse_goal", "length_reduction_and_goal")


@dataclass(frozen=True, slots=True)
class RewardSignal:
    """Quantities available to a reward strategy after one environment step."""

    previous_best_length: int
    new_best_length: int
    goal_reached: bool
    goal_reward: float


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
    raise ValueError(f"unknown reward mode {mode!r}")
