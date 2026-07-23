from __future__ import annotations

import pytest

from ac_zero.scheduler.benchmark_ladder import LadderRung, evaluation_decision

_NOW = "2026-07-23T12:00:00Z"


def _rung(
    metric: float, *, run_id: str = "run-1", fmt: int | None = 1, at: str = _NOW
) -> LadderRung:
    return LadderRung(run_id=run_id, metric=metric, format_version=fmt, at=at)


def _decide(
    rung: LadderRung | None,
    metric: float,
    *,
    run_id: str = "run-2",
    fmt: int | None = 1,
    error_reduction: float = 0.25,
    staleness_days: float = 14.0,
    now: str = _NOW,
) -> tuple[bool, str]:
    return evaluation_decision(
        rung,
        metric=metric,
        run_id=run_id,
        format_version=fmt,
        error_reduction=error_reduction,
        staleness_days=staleness_days,
        now=now,
    )


def test_a_quarter_of_the_remaining_error_sets_the_next_rung() -> None:
    assert _rung(0.35).next_metric(0.25) == pytest.approx(0.5125)


def test_the_ladder_is_geometric_in_error() -> None:
    metric, rungs = 0.35, []
    for _ in range(4):
        metric = _rung(metric).next_metric(0.25)  # type: ignore[assignment]
        rungs.append(round(metric, 4))
    assert rungs == [0.5125, 0.6344, 0.7258, 0.7943]


def test_a_metric_outside_the_unit_interval_has_no_rung() -> None:
    assert _rung(1.7).next_metric(0.25) is None
    assert _rung(-0.4).next_metric(0.25) is None
    assert _rung(1.0).next_metric(0.25) is None


def test_a_model_that_has_never_been_evaluated_is_evaluated() -> None:
    evaluate, reason = _decide(None, 0.31)
    assert evaluate
    assert "first" in reason


def test_clearing_the_rung_earns_an_evaluation() -> None:
    assert _decide(_rung(0.35), 0.52)[0]


def test_landing_exactly_on_the_rung_earns_an_evaluation() -> None:
    assert _decide(_rung(0.35), 0.5125)[0]


def test_an_improvement_short_of_the_rung_does_not() -> None:
    evaluate, reason = _decide(_rung(0.35), 0.49)
    assert not evaluate
    assert "below rung" in reason


def test_the_rung_the_model_is_measured_against_is_reported() -> None:
    assert "0.512" in _decide(_rung(0.35), 0.49)[1]


def test_the_best_model_that_set_the_rung_is_not_re_evaluated() -> None:
    evaluate, reason = _decide(_rung(0.35, run_id="run-1"), 0.35, run_id="run-1")
    assert not evaluate
    assert reason == ""  # the steady state while a run trains; nothing to report


def test_a_return_ema_run_keeps_the_one_evaluation_per_run_behaviour() -> None:
    # ``1 - return_ema`` is not an error rate, so no ladder can apply to it.
    evaluate, reason = _decide(_rung(4.2), 4.3)
    assert evaluate
    assert "not an accuracy" in reason


def test_a_format_bump_resets_the_ladder() -> None:
    evaluate, reason = _decide(_rung(0.90, fmt=2), 0.40, fmt=3)
    assert evaluate
    assert "format" in reason


def test_a_plateaued_metric_is_evaluated_once_the_rung_goes_stale() -> None:
    evaluate, reason = _decide(_rung(0.35, at="2026-07-01T12:00:00Z"), 0.36, staleness_days=14.0)
    assert evaluate
    assert "since the last evaluation" in reason


def test_a_rung_short_of_stale_still_holds_the_model_back() -> None:
    assert not _decide(_rung(0.35, at="2026-07-20T12:00:00Z"), 0.36, staleness_days=14.0)[0]


def test_an_unreadable_rung_timestamp_fails_open() -> None:
    # Losing the ability to tell the time must never quietly end the evaluations.
    assert _decide(_rung(0.35, at="not-a-time"), 0.36)[0]


def test_a_larger_reduction_demands_more_before_the_next_evaluation() -> None:
    assert _decide(_rung(0.35), 0.60, error_reduction=0.25)[0]
    assert not _decide(_rung(0.35), 0.60, error_reduction=0.50)[0]
