"""Navigation reward: distance-shaping with a per-episode adaptive weight.

The agent navigates a graph from a start node to a destination node. Each step
is scored by four config-driven components -- a terminal destination bonus that
grows with the start-to-destination distance, a distance-reduction shaping term
weighted by the episode's ``alpha``, a flat move fee, and a revisit fee -- and
``alpha`` itself is retuned after every episode from running success/progress
EMAs so the shaping pressure tracks how well the policy is currently solving the
task.

The three collaborators are deliberately decoupled so each is trivially
testable in isolation:

- :class:`RewardComputer` holds the *within-episode* state (the visited set, the
  start distance ``L0``, the minimum distance reached) and turns one transition
  into a :class:`RewardComponents`. It never mutates ``alpha``.
- :class:`AlphaUpdater` holds the *across-episode* state (the EMAs and the live
  ``alpha``) and is stepped once per finished episode from an
  :class:`EpisodeStats`.
- :class:`EpisodeStats` is the immutable hand-off between the two: the aggregate
  an episode produces and the updater consumes.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Every scale and threshold the navigation reward and alpha rule read.

    No constant in :class:`RewardComputer` or :class:`AlphaUpdater` is
    hard-coded; they all live here so a run is fully described by its config.
    """

    # Reward component scales.
    destination_reward_scale: float = 1.0
    move_fee_scale: float = 0.01
    revisit_fee_scale: float = 0.02
    # Ceiling on the distance change one move may be *paid* for. Shortest-path
    # distance is 1-Lipschitz over an invertible moveset -- a single move changes
    # the true distance by at most one -- so on the annotated graph this clips
    # nothing. It binds only where the distance is an estimate: re-entering the
    # graph after an off-graph excursion scores against an anchor inflated by the
    # whole detour, and would otherwise pay one move the worth of a dozen descents.
    # Measured on the rank-2 ball, re-entry steps claimed up to +9 where a genuine
    # descent earns +1. The network cannot see *why* that move was special, so it
    # generalizes the preference to states that merely encode similarly.
    max_shaping_progress: int = 1
    # Alpha updater: initial weight, bounds, and EMA smoothing.
    alpha_initial: float = 0.3
    # The floor is deliberately well above zero. Annealing alpha to ~0 turns the
    # shaping off entirely, and the value function then converges against a
    # near-sparse reward; the next ramp back up changes the reward scale by the
    # whole alpha_max/alpha_min ratio at once, invalidating that value function
    # and destroying the policy through it. Keeping the span inside one order of
    # magnitude bounds how much the reward can move under a converged critic.
    alpha_min: float = 0.05
    alpha_max: float = 1.0
    ema_rate: float = 0.05
    # Long-memory success EMA, used only to tell a regression from a stall (see
    # `advance`). Much slower than `ema_rate` on purpose: the fast EMA decays to
    # ~0 within a few iterations of a collapse, so it cannot distinguish a policy
    # that was working moments ago from one that never worked at all. This rate
    # sets how long a collapsed run is given to recover before shaping ramps
    # again -- ~460 episodes, which is ~20 iterations at 24 episodes/iteration
    # and ~58 at 8. It folds per episode, so the window in *iterations* scales
    # with `episodes_per_iteration`.
    recovery_ema_rate: float = 0.002
    # Alpha update-rule thresholds.
    progress_low: float = 0.3
    progress_high: float = 0.8
    success_target: float = 0.4
    success_good: float = 0.7
    # Alpha multiplicative moves.
    increase_factor: float = 1.05
    decrease_factor: float = 0.8
    anneal_factor: float = 0.95
    # Iterations that must elapse between two alpha moves. The EMAs still fold
    # every episode -- only the multiplicative move is rate-limited, so the
    # controller's actuator cannot outrun its own sensor. 1 applies a move once
    # per iteration; raising it slows alpha further.
    alpha_update_every_iterations: int = 1

    def validate(self) -> None:
        """Reject configs that would make the reward or alpha rule ill-defined."""
        if self.destination_reward_scale < 0.0:
            raise ValueError("destination_reward_scale must be non-negative")
        if self.move_fee_scale < 0.0:
            raise ValueError("move_fee_scale must be non-negative")
        if self.revisit_fee_scale < 0.0:
            raise ValueError("revisit_fee_scale must be non-negative")
        if self.max_shaping_progress < 1:
            raise ValueError("max_shaping_progress must be at least 1")
        if self.alpha_initial <= 0.0:
            raise ValueError("alpha_initial must be positive")
        if not 0.0 < self.alpha_min <= self.alpha_max:
            raise ValueError("require 0 < alpha_min <= alpha_max")
        if not self.alpha_min <= self.alpha_initial <= self.alpha_max:
            raise ValueError("alpha_initial must lie in [alpha_min, alpha_max]")
        if not 0.0 < self.ema_rate <= 1.0:
            raise ValueError("ema_rate must be in (0, 1]")
        if not 0.0 < self.recovery_ema_rate <= 1.0:
            raise ValueError("recovery_ema_rate must be in (0, 1]")
        if self.alpha_update_every_iterations < 1:
            raise ValueError("alpha_update_every_iterations must be at least 1")
        if not 0.0 <= self.progress_low <= self.progress_high <= 1.0:
            raise ValueError("require 0 <= progress_low <= progress_high <= 1")
        if not 0.0 <= self.success_target <= 1.0:
            raise ValueError("success_target must be in [0, 1]")
        if not 0.0 <= self.success_good <= 1.0:
            raise ValueError("success_good must be in [0, 1]")
        for name in ("increase_factor", "decrease_factor", "anneal_factor"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RewardComponents:
    """The scored parts of one transition, kept separate for later re-tuning.

    Storing the pieces (not just ``reward_total``) lets a replay buffer recompute
    the reward under a different ``alpha`` or scaling scheme without replaying the
    episode. ``reward_move_fee`` and ``reward_revisit_fee`` are already signed
    (non-positive), so the four component fields sum to ``reward_total``.
    """

    reward_total: float
    reward_destination: float
    reward_shaping: float
    reward_move_fee: float
    reward_revisit_fee: float
    alpha: float
    distance_before: int
    distance_after: int
    distance_progress: int


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    """Aggregate an episode produces and the alpha updater consumes.

    ``progress_rate`` is the fraction of the start-to-destination distance the
    episode closed at its closest approach; ``success`` is whether it reached the
    destination. The reward-component sums and counts ride along so a training
    loop can log per-episode averages (item 7 of the reward spec).
    """

    start_distance: int
    min_distance_reached: int
    final_distance: int
    success: bool
    length: int
    revisit_count: int
    alpha: float
    destination_reward: float
    shaping_reward: float
    move_fee: float
    revisit_fee: float
    total_reward: float

    @property
    def progress_rate(self) -> float:
        """Closed fraction of the start distance; 0 when already at the goal."""
        if self.start_distance <= 0:
            return 0.0
        return (self.start_distance - self.min_distance_reached) / self.start_distance


@dataclass(frozen=True, slots=True)
class RewardSnapshot:
    """A :class:`RewardComputer`'s within-episode state, captured mid-episode.

    A tree search plays hypothetical moves through the same environment the real
    episode steps, and every one of those calls folds into the visited set, the
    distance trackers, and the component sums. Handing this snapshot back after
    the search puts the episode where it was, so the move the agent actually plays
    is scored against its own path rather than the search's.

    ``alpha`` and the start distance are fixed for the episode's span, so they are
    neither captured nor restored.
    """

    visited: frozenset[Hashable]
    min_distance: int
    final_distance: int
    revisit_count: int
    steps: int
    reached_destination: bool
    sums: _ComponentSums


class RewardComputer:
    """Scores transitions within one episode at a fixed ``alpha``.

    An episode is bracketed by :meth:`start_episode` (which resets the visited
    set and distance trackers) and any number of :meth:`step` calls. ``alpha`` is
    captured at :meth:`start_episode` and never changes mid-episode, so every
    step of one episode shares the same shaping weight.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._alpha = config.alpha_initial
        self._start_distance = 0
        self._min_distance = 0
        self._final_distance = 0
        self._revisit_count = 0
        self._steps = 0
        self._reached_destination = False
        self._visited: set[Hashable] = set()
        self._sums = _ComponentSums()

    def start_episode(self, alpha: float, start_node: Hashable, start_distance: int) -> None:
        """Begin a fresh episode at ``start_node``, fixing ``alpha`` for its span.

        Resets the visited set (item 4/7 of the spec: revisits are per-episode)
        so a node revisited only in a *previous* episode is not penalized here.
        The start node counts as visited, so stepping back onto it is a revisit.
        """
        self._alpha = alpha
        self._start_distance = start_distance
        self._min_distance = start_distance
        self._final_distance = start_distance
        self._revisit_count = 0
        self._steps = 0
        self._reached_destination = False
        self._visited = {start_node}
        self._sums = _ComponentSums()

    @property
    def alpha(self) -> float:
        """The (constant within an episode) shaping weight currently in force."""
        return self._alpha

    @property
    def start_distance(self) -> int:
        """The episode's start-to-destination distance ``L0`` (fixed for its span).

        The value reconstruction reads it to turn the ``alpha``-free heads back into
        a scalar: ``L0 * (destination_scale * success + alpha * progress)``.
        """
        return self._start_distance

    def step(
        self,
        next_node: Hashable,
        distance_before: int,
        distance_after: int,
        reached_destination: bool,
    ) -> RewardComponents:
        """Score the transition into ``next_node`` and fold it into episode stats.

        ``next_node`` is checked against the visited set *before* being added, so
        the revisit fee fires only when the node was reached earlier this episode.
        """
        cfg = self.config
        distance_progress = distance_before - distance_after
        # `distance_progress` is reported unclipped -- it is what happened -- while
        # shaping pays the clipped value, which is what one move can be worth (see
        # `max_shaping_progress`).
        cap = cfg.max_shaping_progress
        shaping = self._alpha * max(-cap, min(cap, distance_progress))
        destination = (
            cfg.destination_reward_scale * self._start_distance if reached_destination else 0.0
        )
        move_fee = -cfg.move_fee_scale
        revisited = next_node in self._visited
        revisit_fee = -cfg.revisit_fee_scale if revisited else 0.0
        total = destination + shaping + move_fee + revisit_fee

        # Update episode state only after reading the pre-step visited set.
        if revisited:
            self._revisit_count += 1
        self._visited.add(next_node)
        self._min_distance = min(self._min_distance, distance_after)
        self._final_distance = distance_after
        self._reached_destination = self._reached_destination or reached_destination
        self._steps += 1
        self._sums.add(destination, shaping, move_fee, revisit_fee, total)

        return RewardComponents(
            reward_total=total,
            reward_destination=destination,
            reward_shaping=shaping,
            reward_move_fee=move_fee,
            reward_revisit_fee=revisit_fee,
            alpha=self._alpha,
            distance_before=distance_before,
            distance_after=distance_after,
            distance_progress=distance_progress,
        )

    def snapshot(self) -> RewardSnapshot:
        """Capture the within-episode state so a search can hand it back later."""
        return RewardSnapshot(
            visited=frozenset(self._visited),
            min_distance=self._min_distance,
            final_distance=self._final_distance,
            revisit_count=self._revisit_count,
            steps=self._steps,
            reached_destination=self._reached_destination,
            sums=replace(self._sums),
        )

    def restore(self, snapshot: RewardSnapshot) -> None:
        """Roll the within-episode state back to ``snapshot``."""
        self._visited = set(snapshot.visited)
        self._min_distance = snapshot.min_distance
        self._final_distance = snapshot.final_distance
        self._revisit_count = snapshot.revisit_count
        self._steps = snapshot.steps
        self._reached_destination = snapshot.reached_destination
        self._sums = replace(snapshot.sums)

    def episode_stats(self) -> EpisodeStats:
        """Snapshot the finished episode for the alpha updater and metrics."""
        return EpisodeStats(
            start_distance=self._start_distance,
            min_distance_reached=self._min_distance,
            final_distance=self._final_distance,
            success=self._reached_destination,
            length=self._steps,
            revisit_count=self._revisit_count,
            alpha=self._alpha,
            destination_reward=self._sums.destination,
            shaping_reward=self._sums.shaping,
            move_fee=self._sums.move_fee,
            revisit_fee=self._sums.revisit_fee,
            total_reward=self._sums.total,
        )


@dataclass(slots=True)
class _ComponentSums:
    """Running per-episode totals of each reward component."""

    destination: float = 0.0
    shaping: float = 0.0
    move_fee: float = 0.0
    revisit_fee: float = 0.0
    total: float = 0.0

    def add(
        self, destination: float, shaping: float, move_fee: float, revisit_fee: float, total: float
    ) -> None:
        self.destination += destination
        self.shaping += shaping
        self.move_fee += move_fee
        self.revisit_fee += revisit_fee
        self.total += total


class AlphaUpdater:
    """Retunes the shaping weight ``alpha`` from success/progress.

    Holds the EMAs and the live ``alpha``. Sensor and actuator run at different
    rates on purpose: :meth:`observe` folds one finished episode into the EMAs
    and is called for every episode, while :meth:`advance` applies the
    multiplicative move and is called once per *iteration* (and then only every
    ``alpha_update_every_iterations``-th time). Moving alpha per episode let the
    actuator outrun the EMA that steers it -- alpha could traverse its whole
    range in a couple of iterations, changing the reward scale by orders of
    magnitude under a value function fit to the old one.

    The EMAs seed from the first episode observed to avoid a cold-start bias
    toward zero.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._alpha = config.alpha_initial
        self._progress_ema = 0.0
        self._success_ema = 0.0
        self._recovery_ema = 0.0
        self._seeded = False
        self._iterations_since_move = 0

    @property
    def alpha(self) -> float:
        """The weight the next episode should use."""
        return self._alpha

    @property
    def progress_ema(self) -> float:
        return self._progress_ema

    @property
    def success_ema(self) -> float:
        return self._success_ema

    @property
    def recovery_ema(self) -> float:
        """The long-memory success EMA that gates the increase branch."""
        return self._recovery_ema

    def state_dict(self) -> dict[str, float | bool]:
        """Snapshot the across-episode state so a resumed run continues from it.

        Captures the live ``alpha`` and every EMA (plus whether they are seeded)
        so restoring skips the cold-start re-seed and the shaping weight does not
        snap back to ``alpha_initial`` when a checkpoint is warm-started.
        """
        return {
            "alpha": self._alpha,
            "progress_ema": self._progress_ema,
            "success_ema": self._success_ema,
            "recovery_ema": self._recovery_ema,
            "seeded": self._seeded,
        }

    def load_state_dict(self, state: dict[str, float | bool]) -> None:
        """Restore a snapshot from :meth:`state_dict`, clamping ``alpha`` to bounds.

        ``alpha`` is re-clamped to ``[alpha_min, alpha_max]`` so a snapshot taken
        under looser bounds cannot reintroduce an out-of-range weight after a
        config change between runs -- which is also what pulls a checkpoint
        written under the old ``alpha_min`` of 1e-3 up to the current floor.
        ``recovery_ema`` falls back to ``success_ema`` when absent, so a snapshot
        written before it existed resumes with a plausible value rather than a
        zero that would read as "never succeeded".
        """
        cfg = self.config
        self._alpha = min(max(float(state["alpha"]), cfg.alpha_min), cfg.alpha_max)
        self._progress_ema = float(state["progress_ema"])
        self._success_ema = float(state["success_ema"])
        self._recovery_ema = float(state.get("recovery_ema", self._success_ema))
        self._seeded = bool(state["seeded"])

    def observe(self, stats: EpisodeStats) -> None:
        """Fold one finished episode into the EMAs, leaving ``alpha`` untouched.

        Called for every episode. The move itself is deferred to :meth:`advance`
        so the shaping weight cannot race ahead of the averages that steer it.
        """
        cfg = self.config
        progress_rate = stats.progress_rate
        success = 1.0 if stats.success else 0.0
        if not self._seeded:
            self._progress_ema = progress_rate
            self._success_ema = success
            self._recovery_ema = success
            self._seeded = True
            return
        rate = cfg.ema_rate
        self._progress_ema = (1.0 - rate) * self._progress_ema + rate * progress_rate
        self._success_ema = (1.0 - rate) * self._success_ema + rate * success
        slow = cfg.recovery_ema_rate
        self._recovery_ema = (1.0 - slow) * self._recovery_ema + slow * success

    def advance(self) -> float:
        """Retune ``alpha`` from the current EMAs, at most once every N iterations.

        Called once per iteration; applies a move only on every
        ``alpha_update_every_iterations``-th call, and returns the weight the
        next iteration runs at either way.

        The rule is four-way. Raise ``alpha`` when progress is stalling, lower it
        when the agent makes progress but still fails to reach the goal, anneal it
        once success is reliably high -- and, the fourth case, *hold* it when
        progress has collapsed but ``recovery_ema`` says the policy was recently
        succeeding. That last branch is what stops a transient regression from
        becoming permanent: ramping the shaping back up mid-collapse changes the
        reward scale under a value function fit to the annealed one, which breaks
        the policy far more thoroughly than the dip that triggered it, and the
        resulting zero-success state then keeps ``progress_ema`` low enough to
        pin ``alpha`` at its ceiling forever. Holding gives the policy room to
        recover; if it genuinely never does, ``recovery_ema`` decays and normal
        ramping resumes on its own.

        The weight is clamped to ``[alpha_min, alpha_max]`` so a persistent stall
        cannot ramp it geometrically without bound and blow up the shaping reward
        (and, through it, the value targets and PPO loss).
        """
        cfg = self.config
        self._iterations_since_move += 1
        if self._iterations_since_move < cfg.alpha_update_every_iterations:
            return self._alpha
        self._iterations_since_move = 0

        if self._progress_ema < cfg.progress_low:
            if self._recovery_ema < cfg.success_target:
                self._alpha = self._alpha * cfg.increase_factor
        elif self._progress_ema > cfg.progress_high and self._success_ema < cfg.success_target:
            self._alpha = self._alpha * cfg.decrease_factor
        elif self._success_ema > cfg.success_good:
            self._alpha = self._alpha * cfg.anneal_factor
        self._alpha = min(max(self._alpha, cfg.alpha_min), cfg.alpha_max)
        return self._alpha
