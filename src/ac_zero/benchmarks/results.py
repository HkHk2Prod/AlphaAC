"""Shape a benchmark run into its published documents and push them.

Results live under one bucket prefix, the catalogs they score against under
another::

    benchmarks/
        <checkpoint_name>.json                  # summary: this run + the lineage's history
        runs/<checkpoint_name>/<run_id>.json    # detail: every entry's outcome

    benchmark_datasets/
        <catalog_name>.json                     # the entry set, e.g. ak-ms-rel48-w7

Summaries sit at the top of ``benchmarks/`` so the folder listing *is* the
leaderboard -- one file per model, named after it. The per-entry detail is bulky
and only read when something interesting turned up, so it lives a level down.

Catalogs get their own prefix rather than a subfolder of results because they are
inputs, not outputs: one catalog is scored by every model, it is written by
``benchmark create`` before any run exists, and it is immutable for a given name
(a pure function of its two bounds). Keeping it out of ``benchmarks/`` also keeps
that folder's listing exactly one line per model.

The summary accumulates across runs of a checkpoint name. ``best_solved`` is the
high-water mark and ``ever_solved`` is the union of every presentation the
lineage has ever solved, because a solve is a permanent fact about a
presentation: a later run that happens to get less budget does not un-solve it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ac_zero.benchmarks.catalog import BenchmarkCatalog
from ac_zero.benchmarks.config import BenchmarkConfig
from ac_zero.benchmarks.evaluation import BenchmarkReport
from ac_zero.datasets.hub import DEFAULT_BUCKET, download_file, upload_files
from ac_zero.scheduler.models import utc_now

BENCHMARK_PREFIX = "benchmarks"
BENCHMARK_DATASET_PREFIX = "benchmark_datasets"
MAX_SUMMARY_RUNS = 50


def summary_path(checkpoint_name: str) -> str:
    """Bucket path of ``checkpoint_name``'s rolling summary."""
    return f"{BENCHMARK_PREFIX}/{checkpoint_name}.json"


def detail_path(checkpoint_name: str, run_id: str) -> str:
    """Bucket path of one run's per-entry detail."""
    return f"{BENCHMARK_PREFIX}/runs/{checkpoint_name}/{run_id}.json"


def catalog_remote_path(catalog_name: str) -> str:
    """Bucket path of the catalog itself -- the shared entry set, not a run's output."""
    return f"{BENCHMARK_DATASET_PREFIX}/{catalog_name}.json"


def upload_catalog(catalog: BenchmarkCatalog, *, bucket: str = DEFAULT_BUCKET) -> str:
    """Publish ``catalog`` under ``benchmark_datasets/`` and return its bucket path.

    Overwriting an existing file is harmless: a catalog name pins both bounds and
    the enumeration is deterministic, so the same name always means the same set.
    """
    remote = catalog_remote_path(catalog.name)
    with tempfile.TemporaryDirectory() as tmp:
        local = catalog.write(Path(tmp) / "catalog.json")
        upload_files([(local, remote)], bucket=bucket)
    return remote


def detail_payload(
    report: BenchmarkReport, config: BenchmarkConfig, *, run_id: str
) -> dict[str, Any]:
    """The full record: every attempted entry, plus the budget that produced it."""
    return {
        "run_id": run_id,
        "checkpoint_name": report.checkpoint_name,
        "catalog": report.catalog_name,
        "finished_at": utc_now(),
        "attempted": report.attempted,
        "solved": len(report.solved),
        "solve_rate": report.solve_rate,
        "out_of_capacity": report.out_of_capacity,
        "seconds": round(report.seconds, 1),
        "deep_pass_ran": report.deep_pass_ran,
        "stopped_early": report.stopped_early,
        "budget": {
            "scan_expansions": config.scan_expansions,
            "scan_generated": config.scan_generated,
            "deep_simulations": config.deep_simulations,
            "deep_moves": config.deep_moves,
            "max_moves": config.max_moves,
            "max_total_minutes": config.max_total_minutes,
        },
        "results": [result.to_json() for result in report.results],
    }


def _run_entry(report: BenchmarkReport, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "catalog": report.catalog_name,
        "attempted": report.attempted,
        "solved": len(report.solved),
        "solve_rate": round(report.solve_rate, 6),
        "seconds": round(report.seconds, 1),
        "stopped_early": report.stopped_early,
        "finished_at": utc_now(),
    }


def merge_summary(
    previous: dict[str, Any] | None, report: BenchmarkReport, *, run_id: str
) -> dict[str, Any]:
    """Fold one run into the checkpoint's rolling summary."""
    prior = previous or {}
    runs = [r for r in prior.get("runs", []) if r.get("run_id") != run_id]
    runs.append(_run_entry(report, run_id))
    runs.sort(key=lambda r: str(r.get("run_id")))

    ever = set(prior.get("ever_solved") or [])
    ever.update(result.presentation_id for result in report.solved)
    return {
        "checkpoint_name": report.checkpoint_name,
        "catalog": report.catalog_name,
        "updated_at": utc_now(),
        "latest": _run_entry(report, run_id),
        "best_solved": max(int(prior.get("best_solved", 0)), len(report.solved)),
        "ever_solved": sorted(ever),
        "ever_solved_count": len(ever),
        "by_family": report.counts_by_family(),
        "runs": runs[-MAX_SUMMARY_RUNS:],
    }


def publish_benchmark(
    report: BenchmarkReport,
    config: BenchmarkConfig,
    catalog: BenchmarkCatalog,
    *,
    run_id: str,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, str]:
    """Upload the detail, the merged summary, and the catalog. Returns the paths.

    The catalog is uploaded alongside so a published solve rate can always be
    traced back to the exact entry set it was computed over -- the bounds alone
    would not pin it down if the enumerator ever changes.
    """
    name = report.checkpoint_name
    remote_summary = summary_path(name)
    remote_detail = detail_path(name, run_id)
    remote_catalog = catalog_remote_path(catalog.name)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        previous = _read_remote_json(remote_summary, work / "previous.json", bucket=bucket)

        detail_file = work / "detail.json"
        detail_file.write_text(
            json.dumps(detail_payload(report, config, run_id=run_id), indent=2, sort_keys=True)
            + "\n"
        )
        summary_file = work / "summary.json"
        summary_file.write_text(
            json.dumps(merge_summary(previous, report, run_id=run_id), indent=2, sort_keys=True)
            + "\n"
        )
        catalog_file = catalog.write(work / "catalog.json")

        upload_files(
            [
                (detail_file, remote_detail),
                (summary_file, remote_summary),
                (catalog_file, remote_catalog),
            ],
            bucket=bucket,
        )
    print(f"[benchmark-upload] pushed {remote_summary} and {remote_detail} to {bucket}")
    return {"summary": remote_summary, "detail": remote_detail, "catalog": remote_catalog}


def _read_remote_json(remote: str, local: Path, *, bucket: str) -> dict[str, Any] | None:
    path = download_file(remote, local, bucket=bucket, missing_ok=True)
    if path is None or not Path(path).is_file():
        return None
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else None
