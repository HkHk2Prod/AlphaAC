from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ac_zero.benchmarks import results as results_module
from ac_zero.benchmarks.catalog import BenchmarkCatalog
from ac_zero.benchmarks.config import BenchmarkConfig
from ac_zero.benchmarks.evaluation import BenchmarkReport, EntryResult
from ac_zero.benchmarks.results import (
    catalog_remote_path,
    detail_path,
    detail_payload,
    merge_summary,
    publish_benchmark,
    summary_path,
)


def _result(presentation_id: str, *, solved: bool) -> EntryResult:
    return EntryResult(
        presentation_id=presentation_id,
        family="miller_schupp",
        total_length=9,
        solved=solved,
        agent="greedy-best-first",
        moves=2,
        path=(1, 2),
        best_reduction=3,
        termination_reason="goal" if solved else "budget_exhausted",
        seconds=0.5,
        expanded_nodes=12,
    )


def _report(*solved_ids: str, attempted: int = 3) -> BenchmarkReport:
    report = BenchmarkReport("ak-ms-rel8-w3", "rank2-ppo-x")
    report.results = [_result(f"ms-{i}", solved=False) for i in range(attempted)]
    for index, name in enumerate(solved_ids):
        report.results[index] = _result(name, solved=True)
    report.attempted = attempted
    report.seconds = 12.0
    return report


def test_paths_put_summaries_at_the_top_and_catalogs_in_their_own_prefix() -> None:
    assert summary_path("model-a") == "benchmarks/model-a.json"
    assert detail_path("model-a", "run-1") == "benchmarks/runs/model-a/run-1.json"
    assert catalog_remote_path("ak-ms-rel48-w7") == "benchmark_datasets/ak-ms-rel48-w7.json"


def test_summary_records_the_first_run() -> None:
    summary = merge_summary(None, _report("ms-a"), run_id="run-1")
    assert summary["checkpoint_name"] == "rank2-ppo-x"
    assert summary["best_solved"] == 1
    assert summary["ever_solved"] == ["ms-a"]
    assert summary["latest"]["run_id"] == "run-1"
    assert [entry["run_id"] for entry in summary["runs"]] == ["run-1"]


def test_ever_solved_is_the_union_across_runs() -> None:
    first = merge_summary(None, _report("ms-a"), run_id="run-1")
    second = merge_summary(first, _report("ms-b"), run_id="run-2")
    assert second["ever_solved"] == ["ms-a", "ms-b"]
    assert second["ever_solved_count"] == 2


def test_best_solved_is_a_high_water_mark_a_weaker_run_cannot_lower() -> None:
    strong = merge_summary(None, _report("ms-a", "ms-b"), run_id="run-1")
    weak = merge_summary(strong, _report("ms-a"), run_id="run-2")
    assert weak["best_solved"] == 2
    assert weak["latest"]["solved"] == 1


def test_rerunning_the_same_run_id_replaces_rather_than_duplicates() -> None:
    first = merge_summary(None, _report("ms-a"), run_id="run-1")
    again = merge_summary(first, _report("ms-a", "ms-b"), run_id="run-1")
    assert [entry["run_id"] for entry in again["runs"]] == ["run-1"]
    assert again["runs"][0]["solved"] == 2


def test_run_history_is_trimmed_to_the_recent_tail() -> None:
    summary: dict[str, Any] | None = None
    for index in range(results_module.MAX_SUMMARY_RUNS + 5):
        summary = merge_summary(summary, _report("ms-a"), run_id=f"run-{index:03d}")
    assert summary is not None
    assert len(summary["runs"]) == results_module.MAX_SUMMARY_RUNS


def test_detail_records_every_attempted_entry_and_the_budget() -> None:
    config = BenchmarkConfig(scan_expansions=99, deep_simulations=7)
    payload = detail_payload(_report("ms-a"), config, run_id="run-1")
    assert len(payload["results"]) == 3
    assert payload["solved"] == 1
    assert payload["budget"]["scan_expansions"] == 99
    assert payload["budget"]["deep_simulations"] == 7


def test_publish_writes_summary_detail_and_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: dict[str, str] = {}

    def fake_upload(pairs: list[tuple[str | Path, str]], *, bucket: str) -> None:
        for local, remote in pairs:
            uploaded[remote] = Path(local).read_text()

    monkeypatch.setattr(results_module, "upload_files", fake_upload)
    monkeypatch.setattr(
        results_module, "download_file", lambda *a, **k: None
    )  # no previous summary

    catalog = BenchmarkCatalog.build(max_relator_length=8, max_w_length=2)
    paths = publish_benchmark(
        _report("ms-a"), BenchmarkConfig(), catalog, run_id="run-1", bucket="b/c"
    )

    assert set(paths) == {"summary", "detail", "catalog"}
    assert set(uploaded) == set(paths.values())
    assert json.loads(uploaded[paths["summary"]])["best_solved"] == 1
    assert json.loads(uploaded[paths["detail"]])["run_id"] == "run-1"
    assert json.loads(uploaded[paths["catalog"]])["name"] == catalog.name


def test_publish_merges_into_an_existing_remote_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = merge_summary(None, _report("ms-old"), run_id="run-0")

    def fake_download(remote: str, local: Path, *, bucket: str, missing_ok: bool) -> Path | None:
        if remote != summary_path("rank2-ppo-x"):
            return None
        Path(local).write_text(json.dumps(previous))
        return Path(local)

    uploaded: dict[str, str] = {}
    monkeypatch.setattr(results_module, "download_file", fake_download)
    monkeypatch.setattr(
        results_module,
        "upload_files",
        lambda pairs, *, bucket: uploaded.update(
            {remote: Path(local).read_text() for local, remote in pairs}
        ),
    )

    catalog = BenchmarkCatalog.build(max_relator_length=8, max_w_length=2)
    paths = publish_benchmark(
        _report("ms-new"), BenchmarkConfig(), catalog, run_id="run-1", bucket="b/c"
    )
    summary = json.loads(uploaded[paths["summary"]])
    assert summary["ever_solved"] == ["ms-new", "ms-old"]
    assert len(summary["runs"]) == 2


def test_upload_catalog_publishes_under_the_dataset_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ac_zero.benchmarks.results import upload_catalog

    uploaded: dict[str, str] = {}
    monkeypatch.setattr(
        results_module,
        "upload_files",
        lambda pairs, *, bucket: uploaded.update(
            {remote: Path(local).read_text() for local, remote in pairs}
        ),
    )
    catalog = BenchmarkCatalog.build(max_relator_length=8, max_w_length=2)
    remote = upload_catalog(catalog, bucket="b/c")

    assert remote == f"benchmark_datasets/{catalog.name}.json"
    assert json.loads(uploaded[remote])["count"] == len(catalog.entries)
