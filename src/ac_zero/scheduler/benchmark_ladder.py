"""When an improved checkpoint has improved *enough* to be worth benchmarking again.

A benchmark run is expensive, so a model does not earn one every time a training
run beats its predecessor -- it earns one when it has closed a fixed fraction of
its remaining error. From a first evaluation at 0.35, a 25% reduction puts the
next rung at 0.51, then 0.63, 0.73, 0.79. That is geometric in error, i.e.
uniform in ``log(1 - accuracy)``, so every evaluation costs the same and buys the
same amount of evidence -- unlike a fixed absolute step, which fires constantly
at 0.35 and never again past 0.95.

Three things keep the ladder from silently ending the evaluations:

* It only applies to a metric that *is* an accuracy. The training pipeline's
  checkpoint metric is the self-play success EMA on navigation runs but the
  return EMA elsewhere, and ``1 - return_ema`` is not an error rate, so those
  runs keep the old one-evaluation-per-run behaviour.
* A rung is only comparable within one model format. After a format bump the
  ladder resets, exactly as ``best.json`` promotion does.
* A metric that plateaus a hair under the next rung would otherwise freeze the
  evaluations forever, while the thing actually being measured -- how many
  benchmark entries the model solves -- may well still be moving. So a rung goes
  stale: once a name has gone ``staleness_days`` without an evaluation, its next
  new best model earns one wherever it sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ac_zero.scheduler.models import parse_iso

DEFAULT_ERROR_REDUCTION = 0.25
DEFAULT_STALENESS_DAYS = 14.0


@dataclass(slots=True)
class LadderRung:
    """The last best-model of one checkpoint name that was committed to an evaluation.

    Recorded at enqueue rather than at dispatch: a checkpoint waiting in the queue
    has already claimed its rung, so a string of small improvements cannot pile up
    behind one pending evaluation.
    """

    run_id: str
    metric: float
    format_version: int | None = None
    at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LadderRung:
        fmt = data.get("format_version")
        return cls(
            run_id=str(data.get("run_id", "")),
            metric=float(data.get("metric", 0.0)),
            format_version=None if fmt is None else int(fmt),
            at=str(data.get("at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "metric": self.metric,
            "format_version": self.format_version,
            "at": self.at,
        }

    def next_metric(self, error_reduction: float) -> float | None:
        """The accuracy the next evaluation must reach, or ``None`` if this is no accuracy."""
        if not 0.0 <= self.metric < 1.0:
            return None
        return 1.0 - (1.0 - self.metric) * (1.0 - error_reduction)


def evaluation_decision(
    rung: LadderRung | None,
    *,
    metric: float,
    run_id: str,
    format_version: int | None,
    error_reduction: float,
    staleness_days: float,
    now: str,
) -> tuple[bool, str]:
    """Whether this best model earns an evaluation, and why (or why not).

    An empty reason means there is nothing to say: the best model on offer is the
    very one the rung was set from, which is the steady state while a run trains.
    """
    if rung is None:
        return True, "first evaluation"
    if rung.run_id == run_id:
        return False, ""
    if rung.format_version != format_version:
        return True, f"model format {rung.format_version} -> {format_version}"
    target = rung.next_metric(error_reduction)
    if target is None:
        return True, f"metric {rung.metric:.3f} is not an accuracy; no ladder applies"
    if metric >= target:
        return True, f"metric {metric:.3f} cleared rung {target:.3f}"
    if _days_since(rung.at, now) >= staleness_days:
        return True, f"{staleness_days:g}d since the last evaluation"
    return False, f"metric {metric:.3f} below rung {target:.3f}"


def _days_since(stamp: str, now: str) -> float:
    """Days between ``stamp`` and ``now``, unbounded when either cannot be read.

    Failing open matters more than precision: an unreadable timestamp should let an
    evaluation through, never quietly stop them.
    """
    then, reference = parse_iso(stamp), parse_iso(now)
    if then is None or reference is None:
        return float("inf")
    return (reference - then).total_seconds() / 86_400
