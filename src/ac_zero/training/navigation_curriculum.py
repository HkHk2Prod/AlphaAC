"""Distance curriculum: a dynamic cap on the problems self-play samples.

The agent is trained to navigate from a start node to the destination in a
graph; a problem's difficulty is its shortest-path distance ``L`` between the
two. This module raises a ceiling ``L_max`` on that difficulty as the policy
starts solving the hardest problems it is currently allowed, and lowers it when
the frontier collapses -- a curriculum over *which problems are sampled*.

It is deliberately independent of the shaping-weight ``AlphaUpdater`` in
:mod:`ac_zero.environment.navigation_reward`: alpha shapes the reward, this
shapes the sampling distribution, and neither reads the other's state. The two
are stepped from the same :class:`EpisodeStats` but by separate calls.

:class:`DistanceCurriculum` holds the across-episode state -- the live ``L_max``
and a rolling success estimate over *frontier* episodes only (those whose ``L``
sits in the top ``frontier_fraction`` band of the ``L_max`` that was active when
they were sampled). It is stepped once per finished episode via :meth:`update`,
which returns a :class:`CurriculumUpdate` snapshot for logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DistanceCurriculumConfig:
    """Every knob the distance curriculum reads; no constant is hard-coded elsewhere."""

    # The ceiling starts at ``L_max_initial`` and is never lowered below
    # ``L_max_min``; each change moves it by ``L_max_step``.
    L_max_initial: int = 2
    L_max_min: int = 2
    L_max_step: int = 1
    # An episode counts toward the frontier estimate when its distance is in the
    # top ``frontier_fraction`` band of the active ``L_max``.
    frontier_fraction: float = 0.7
    # Rolling frontier success EMA smoothing and the hysteresis band that moves
    # ``L_max`` up (>= high) or down (<= low).
    frontier_success_ema_rate: float = 0.05
    frontier_success_high: float = 0.75
    frontier_success_low: float = 0.25
    # ``L_max`` is frozen until this many frontier episodes have been observed
    # since the last change, so a change rests on a settled estimate.
    min_frontier_episodes_before_update: int = 100
    # Fallback horizon for a sampled problem whose distance to the destination is
    # unknown (off the annotated graph), where ``3 * L + 6`` cannot be formed. A
    # deliberately large cutoff so an unmeasured problem is still given room to
    # solve rather than truncated early.
    unknown_distance_max_moves: int = 512

    def validate(self) -> None:
        """Reject configs that would make the curriculum ill-defined."""
        if self.L_max_min < 2:
            raise ValueError("L_max_min must be at least 2")
        if self.L_max_initial < self.L_max_min:
            raise ValueError("L_max_initial must be >= L_max_min")
        if self.L_max_step < 1:
            raise ValueError("L_max_step must be at least 1")
        if not 0.0 < self.frontier_fraction <= 1.0:
            raise ValueError("frontier_fraction must be in (0, 1]")
        if not 0.0 < self.frontier_success_ema_rate <= 1.0:
            raise ValueError("frontier_success_ema_rate must be in (0, 1]")
        if not 0.0 <= self.frontier_success_low <= self.frontier_success_high <= 1.0:
            raise ValueError("require 0 <= frontier_success_low <= frontier_success_high <= 1")
        if self.min_frontier_episodes_before_update < 1:
            raise ValueError("min_frontier_episodes_before_update must be at least 1")
        if self.unknown_distance_max_moves < 1:
            raise ValueError("unknown_distance_max_moves must be at least 1")


@dataclass(frozen=True, slots=True)
class CurriculumUpdate:
    """Immutable snapshot the curriculum returns after folding one episode.

    Carries exactly the per-episode fields the spec asks to log so the caller can
    emit them without reaching into the curriculum's mutable state.
    """

    L: int
    L_max: int
    L_max_episode: int
    frontier_lower: int
    is_frontier_episode: bool
    frontier_success_ema: float | None
    frontier_success_count: int
    episodes_since_lmax_change: int
    L_max_changed: bool
    L_max_change_direction: str  # "increase", "decrease", or "none"


class DistanceCurriculum:
    """Tracks ``L_max`` and a frontier success estimate across training episodes.

    ``L_max`` bounds the sampled problem distance; :meth:`current_L_max` is read
    before an episode is sampled and :meth:`update` is stepped once after it
    finishes. The rolling estimate only moves on *frontier* episodes -- those in
    the top ``frontier_fraction`` band of the ``L_max`` active when they were
    sampled -- and after enough of them a hysteresis rule nudges ``L_max`` and
    resets the estimate so old-frontier statistics never leak into the new one.
    """

    def __init__(self, config: DistanceCurriculumConfig) -> None:
        self.config = config
        self.L_max = config.L_max_initial
        self.frontier_success_ema: float | None = None
        self.frontier_success_count = 0
        self.episodes_since_lmax_change = 0

    def current_L_max(self) -> int:
        """The ceiling the next episode should be sampled under (``L <= L_max``)."""
        return self.L_max

    def frontier_lower(self, L_max_episode: int) -> int:
        """Lowest distance still in the frontier band for a given active ``L_max``."""
        return math.ceil(self.config.frontier_fraction * L_max_episode)

    def is_frontier(self, L: int, L_max_episode: int) -> bool:
        """Whether distance ``L`` sits in the frontier band of ``L_max_episode``."""
        return self.frontier_lower(L_max_episode) <= L <= L_max_episode

    def _reset_frontier(self) -> None:
        """Drop the frontier estimate so the new ``L_max``'s band starts fresh."""
        self.frontier_success_ema = None
        self.frontier_success_count = 0
        self.episodes_since_lmax_change = 0

    def update(self, L: int, success: bool, L_max_episode: int) -> CurriculumUpdate:
        """Fold one finished episode into the frontier estimate and maybe move ``L_max``.

        ``L_max_episode`` is the ceiling that was active when the episode was
        *sampled*, not the possibly-already-changed live ``L_max`` -- it alone
        decides whether the episode belonged to the frontier. Episodes outside
        the band leave the estimate untouched. ``L_max`` moves only once enough
        frontier episodes have accumulated, and any actual move resets the
        estimate.
        """
        cfg = self.config
        self.episodes_since_lmax_change += 1
        lower = self.frontier_lower(L_max_episode)
        is_frontier = lower <= L <= L_max_episode
        if is_frontier:
            value = 1.0 if success else 0.0
            if self.frontier_success_ema is None:
                self.frontier_success_ema = value
            else:
                rate = cfg.frontier_success_ema_rate
                self.frontier_success_ema = (1.0 - rate) * self.frontier_success_ema + rate * value
            self.frontier_success_count += 1

        changed = False
        direction = "none"
        ready = (
            self.frontier_success_count >= cfg.min_frontier_episodes_before_update
            and self.frontier_success_ema is not None
        )
        if ready:
            assert self.frontier_success_ema is not None
            if self.frontier_success_ema >= cfg.frontier_success_high:
                self.L_max += cfg.L_max_step
                changed, direction = True, "increase"
            elif self.frontier_success_ema <= cfg.frontier_success_low:
                lowered = max(cfg.L_max_min, self.L_max - cfg.L_max_step)
                if lowered != self.L_max:
                    self.L_max = lowered
                    changed, direction = True, "decrease"
        if changed:
            self._reset_frontier()

        return CurriculumUpdate(
            L=L,
            L_max=self.L_max,
            L_max_episode=L_max_episode,
            frontier_lower=lower,
            is_frontier_episode=is_frontier,
            frontier_success_ema=self.frontier_success_ema,
            frontier_success_count=self.frontier_success_count,
            episodes_since_lmax_change=self.episodes_since_lmax_change,
            L_max_changed=changed,
            L_max_change_direction=direction,
        )
