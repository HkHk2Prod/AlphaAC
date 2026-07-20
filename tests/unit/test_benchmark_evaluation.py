from __future__ import annotations

from pathlib import Path

import pytest

from ac_zero.benchmarks.catalog import BenchmarkCatalog
from ac_zero.benchmarks.config import DEEP_AGENT, SCAN_AGENT, BenchmarkConfig
from ac_zero.benchmarks.evaluation import BenchmarkEvaluator, BenchmarkReport, EntryResult

_SMALL = BenchmarkConfig(max_relator_tokens=16, scan_expansions=32, scan_generated=500)


def _catalog() -> BenchmarkCatalog:
    return BenchmarkCatalog.build(max_relator_length=8, max_w_length=3)


def _entry(presentation_id: str, *, solved: bool, agent: str = SCAN_AGENT) -> EntryResult:
    return EntryResult(
        presentation_id=presentation_id,
        family="miller_schupp",
        total_length=10,
        solved=solved,
        agent=agent,
        moves=1,
        path=(0,),
        best_reduction=1,
        termination_reason="goal" if solved else "budget_exhausted",
        seconds=0.1,
        expanded_nodes=3,
    )


def test_scan_attempts_every_entry_without_a_model() -> None:
    catalog = _catalog()
    report = BenchmarkEvaluator(catalog, _SMALL).run(log=lambda _: None)
    assert report.attempted == len(catalog.entries)
    assert not report.deep_pass_ran
    assert not report.stopped_early
    assert all(result.agent == SCAN_AGENT for result in report.results)


def test_scan_solves_some_of_the_easy_catalog() -> None:
    report = BenchmarkEvaluator(_catalog(), _SMALL).run(log=lambda _: None)
    assert 0 < len(report.solved) < report.attempted
    assert 0.0 < report.solve_rate < 1.0


def test_results_carry_a_replayable_path_for_every_solve() -> None:
    report = BenchmarkEvaluator(_catalog(), _SMALL).run(log=lambda _: None)
    for result in report.solved:
        assert result.path
        assert result.moves == len(result.path)
        assert result.termination_reason == "goal"


def test_a_zero_budget_stops_before_scanning_anything() -> None:
    config = BenchmarkConfig(max_relator_tokens=16, scan_expansions=32, max_total_minutes=1e-9)
    report = BenchmarkEvaluator(_catalog(), config).run(log=lambda _: None)
    assert report.stopped_early
    assert report.attempted == 0
    assert report.solve_rate == 0.0


def test_the_deep_pass_needs_a_model() -> None:
    evaluator = BenchmarkEvaluator(_catalog(), _SMALL)
    with pytest.raises(RuntimeError):
        evaluator._deep(_catalog().entries[0])


def test_report_counts_are_grouped_by_family() -> None:
    report = BenchmarkReport("cat", "ckpt")
    report.results = [
        _entry("miller-schupp-1-Y", solved=True),
        _entry("miller-schupp-1-y", solved=False),
    ]
    report.results[1].family = "akbulut_kirby"
    report.attempted = 2
    counts = report.counts_by_family()
    assert counts["miller_schupp"] == {"attempted": 1, "solved": 1}
    assert counts["akbulut_kirby"] == {"attempted": 1, "solved": 0}


def test_solve_rate_is_zero_when_nothing_was_attempted() -> None:
    assert BenchmarkReport("cat", "ckpt").solve_rate == 0.0


def test_entry_result_json_is_serializable() -> None:
    payload = _entry("miller-schupp-1-Y", solved=True, agent=DEEP_AGENT).to_json()
    assert payload["solved"] is True
    assert payload["agent"] == DEEP_AGENT
    assert payload["path"] == [0]


def test_config_reads_nested_benchmark_budgets() -> None:
    config = BenchmarkConfig.from_mapping(
        {"moveset": "strict-ac", "dataset": {"bucket": "x"}, "benchmark": {"scan_expansions": 7}}
    )
    assert config.scan_expansions == 7
    assert config.moveset == "strict-ac"


def test_config_deadline_is_none_without_a_cap() -> None:
    assert BenchmarkConfig().deadline_seconds is None
    assert BenchmarkConfig(max_total_minutes=2).deadline_seconds == 120


def test_loading_a_checkpoint_model_rejects_a_missing_file(tmp_path: Path) -> None:
    from ac_zero.benchmarks.evaluation import load_checkpoint_model

    with pytest.raises(FileNotFoundError):
        load_checkpoint_model(tmp_path / "absent.json")


def test_entries_beyond_the_encoder_capacity_are_reported_not_scored() -> None:
    catalog = BenchmarkCatalog.build(max_relator_length=12, max_w_length=3)
    tight = BenchmarkConfig(max_relator_tokens=8, scan_expansions=8, scan_generated=100)
    report = BenchmarkEvaluator(catalog, tight).run(log=lambda _: None)
    over_long = sum(
        1 for entry in catalog.entries if max(len(relator) for relator in entry.relators) > 8
    )
    assert over_long > 0
    assert report.out_of_capacity == over_long
    assert report.attempted == len(catalog.entries) - over_long


def test_a_matching_capacity_skips_nothing() -> None:
    catalog = BenchmarkCatalog.build(max_relator_length=8, max_w_length=3)
    config = BenchmarkConfig(max_relator_tokens=8, scan_expansions=8, scan_generated=100)
    report = BenchmarkEvaluator(catalog, config).run(log=lambda _: None)
    assert report.out_of_capacity == 0
    assert report.attempted == len(catalog.entries)
