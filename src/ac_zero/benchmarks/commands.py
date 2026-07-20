"""The work behind ``aczero benchmark create`` and ``aczero benchmark run``.

Kept out of ``ac_zero.cli`` so the CLI stays a parser and a dispatch table, and
so these two flows are unit-testable without going through argparse. Each
returns the JSON payload the CLI prints; neither knows about the reporter.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ac_zero.benchmarks.catalog import DEFAULT_MAX_W_LENGTH, BenchmarkCatalog
from ac_zero.benchmarks.config import BenchmarkConfig
from ac_zero.benchmarks.evaluation import BenchmarkEvaluator, load_checkpoint_model
from ac_zero.benchmarks.results import publish_benchmark
from ac_zero.datasets.hub import DEFAULT_BUCKET

Logger = Callable[[str], None]
DEFAULT_CATALOG_DIR = "data/benchmarks"


def default_catalog_path(catalog: BenchmarkCatalog) -> Path:
    return Path(DEFAULT_CATALOG_DIR) / f"{catalog.name}.json"


def create_catalog(
    *,
    max_relator_length: int,
    max_w_length: int = DEFAULT_MAX_W_LENGTH,
    output: str = "",
    upload: bool = True,
    bucket: str = DEFAULT_BUCKET,
    log: Logger = print,
) -> dict[str, Any]:
    """Enumerate the AK/MS catalog for these bounds, write it, and publish it.

    The upload is the default because a catalog is only useful shared: every model
    is scored against the same entry set, and a result's ``catalog`` name has to
    resolve to something for anyone else to check it. Pass ``upload=False`` to keep
    it local.

    A failed upload is reported, not raised -- the local file is the primary output
    and is already written, so a missing token or a network blip should not throw
    away the enumeration. The payload's ``uploaded`` field says what happened.
    """
    catalog = BenchmarkCatalog.build(
        max_relator_length=max_relator_length, max_w_length=max_w_length
    )
    path = Path(output) if output else default_catalog_path(catalog)
    catalog.write(path)
    log(f"[benchmark] wrote {len(catalog.entries)} entries to {path}")

    payload: dict[str, Any] = {
        "catalog": str(path),
        "name": catalog.name,
        "count": len(catalog.entries),
        "families": catalog.to_json()["families"],
        "max_relator_length": max_relator_length,
        "max_w_length": max_w_length,
        "uploaded": False,
    }
    if not upload:
        return payload

    from ac_zero.benchmarks.results import upload_catalog

    try:
        remote = upload_catalog(catalog, bucket=bucket)
    except Exception as exc:
        payload["upload_error"] = f"{type(exc).__name__}: {exc}"
        log(f"[benchmark] upload skipped ({payload['upload_error']}); {path} is written")
        return payload
    payload["uploaded"] = True
    payload["remote"] = remote
    log(f"[benchmark] published {remote} to {bucket}")
    return payload


def _resolve_catalog(config: BenchmarkConfig, *, log: Logger) -> BenchmarkCatalog:
    """Read the configured catalog, building it on the spot when absent.

    A benchmark run should not fail because nobody ran ``benchmark create``
    first: the catalog is a pure function of its two bounds, so it can always be
    regenerated. The bounds come from the file name the config points at.
    """
    path = Path(config.catalog_path)
    if path.is_file():
        return BenchmarkCatalog.read(path)
    bounds = _bounds_from_name(path.stem)
    if bounds is None:
        raise FileNotFoundError(
            f"no catalog at {path} and its name does not encode the bounds to rebuild it; "
            "run `aczero benchmark create` or point --catalog at an existing file"
        )
    log(f"[benchmark] {path} missing; rebuilding it from its name")
    catalog = BenchmarkCatalog.build(max_relator_length=bounds[0], max_w_length=bounds[1])
    catalog.write(path)
    return catalog


def _bounds_from_name(stem: str) -> tuple[int, int] | None:
    """Recover ``(max_relator_length, max_w_length)`` from an ``ak-ms-relL-wW`` name."""
    parts = stem.split("-")
    if len(parts) != 4 or not parts[2].startswith("rel") or not parts[3].startswith("w"):
        return None
    try:
        return int(parts[2][3:]), int(parts[3][1:])
    except ValueError:
        return None


def _resolve_checkpoint(config: BenchmarkConfig, *, log: Logger) -> Path | None:
    """Find the model to evaluate: an explicit file, else the lineage's best on HF."""
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    if not config.checkpoint_name:
        return None
    from ac_zero.training.checkpointing.hub_checkpoints import download_best_checkpoint

    local = Path("runs/benchmark") / f"{config.checkpoint_name}.best.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    path = download_best_checkpoint(config.checkpoint_name, local, bucket=config.bucket)
    if path is None:
        log(f"[benchmark] no best.json for {config.checkpoint_name} in {config.bucket}")
    return path


def run_benchmark(
    config: BenchmarkConfig,
    *,
    run_id: str,
    upload: bool = False,
    output: str = "",
    log: Logger = print,
) -> dict[str, Any]:
    """Evaluate one checkpoint against a catalog and record the result.

    Runs without a checkpoint too -- the classical scan alone is a meaningful
    baseline, and it is what the numbers of a trained model are read against.
    """
    catalog = _resolve_catalog(config, log=log)
    checkpoint = _resolve_checkpoint(config, log=log)
    model = load_checkpoint_model(checkpoint) if checkpoint is not None else None
    log(
        f"[benchmark] {catalog.name}: {len(catalog.entries)} entries, "
        f"model={'yes' if model else 'no'}, run_id={run_id}"
    )

    report = BenchmarkEvaluator(catalog, config, model).run(log=log)
    payload = {
        "run_id": run_id,
        "catalog": catalog.name,
        "checkpoint_name": config.checkpoint_name,
        "attempted": report.attempted,
        "solved": len(report.solved),
        "solve_rate": report.solve_rate,
        "solved_ids": [entry.presentation_id for entry in report.solved],
        "by_family": report.counts_by_family(),
        "out_of_capacity": report.out_of_capacity,
        "stopped_early": report.stopped_early,
        "seconds": round(report.seconds, 1),
    }

    local = Path(output) if output else Path("runs/benchmark") / f"{run_id}.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["local"] = str(local)

    if upload:
        if not config.checkpoint_name:
            raise ValueError(
                "uploading a benchmark result needs --checkpoint-name to file it under"
            )
        payload["remote"] = publish_benchmark(
            report, config, catalog, run_id=run_id, bucket=config.bucket or DEFAULT_BUCKET
        )
    return payload


def load_benchmark_config(path: str) -> dict[str, Any]:
    """Read a benchmark budget YAML, tolerating an absent path (returns ``{}``)."""
    if not path:
        return {}
    import yaml  # type: ignore[import-untyped]

    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data
