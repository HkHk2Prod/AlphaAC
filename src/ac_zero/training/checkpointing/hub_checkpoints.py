"""Push training checkpoint bundles to Hugging Face and pull them for warm starts.

Everything a run produces lives under one prefix keyed by the checkpoint name::

    model_checkpoints/<name>/
        best.json                     # best model across every run of this name
        latest.json                   # most recent run's latest checkpoint
        index.json                    # rollup: best pointer + one entry per run
        runs/<run_id>.metrics.jsonl   # each run's per-update metrics
        runs/<run_id>.meta.json       # each run's provenance
        plots/<run_id>/*.png          # that run's progress plots
        plots/all_runs/*.png          # every run's metrics concatenated

:func:`push_checkpoint_bundle` reads the local bundle, pulls the other runs'
metrics to redraw the all-runs plots, promotes ``best.json`` only when this run
beats the recorded best, and uploads the lot. :class:`PeriodicCheckpointUploader`
drives that on a time interval from the training callbacks, mirroring the dataset
notebooks' periodic upload.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from ac_zero.datasets.hub import DEFAULT_BUCKET, download_file, list_remote, upload_files
from ac_zero.training.checkpointing.checkpoint_bundle import (
    BEST_FILE,
    LATEST_FILE,
    META_FILE,
    METRICS_FILE,
)
from ac_zero.training.logging.plots import PlotsUnavailable, render_training_plots

INDEX_FILE = "index.json"
_METRICS_SUFFIX = ".metrics.jsonl"


def checkpoint_prefix(name: str) -> str:
    """Return the bucket prefix that holds every artifact for ``name``."""
    return f"model_checkpoints/{name}"


def download_best_checkpoint(
    name: str,
    local_path: str | Path,
    *,
    bucket: str = DEFAULT_BUCKET,
    missing_ok: bool = True,
) -> Path | None:
    """Fetch ``best.json`` for ``name`` to ``local_path`` for a warm start.

    Returns ``None`` (rather than raising) when no checkpoint exists yet, so a
    first run on a fresh name simply trains from scratch.
    """
    remote = f"{checkpoint_prefix(name)}/{BEST_FILE}"
    return download_file(remote, local_path, bucket=bucket, missing_ok=missing_ok)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _fetch_other_runs(
    name: str, run_id: str, dest: Path, *, bucket: str
) -> dict[str, list[dict[str, Any]]]:
    """Download every *other* run's metrics under ``name`` into ``dest``.

    Returns a ``run_id -> rows`` mapping used to redraw the all-runs plots.
    """
    prefix = f"{checkpoint_prefix(name)}/runs/"
    history: dict[str, list[dict[str, Any]]] = {}
    for remote in list_remote(prefix, bucket=bucket):
        base = remote.rsplit("/", 1)[-1]
        if not base.endswith(_METRICS_SUFFIX):
            continue
        other = base[: -len(_METRICS_SUFFIX)]
        if other == run_id:
            continue
        local = dest / base
        download_file(remote, local, bucket=bucket)
        history[other] = _read_jsonl(local)
    return history


def _combined_rows(runs: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Concatenate runs in chronological order with a monotonic ``optimizer_step``.

    Each run restarts its optimizer step at 1, so the raw values overlap; the
    all-runs plot re-indexes them onto one global x-axis instead.
    """
    combined: list[dict[str, Any]] = []
    step = 0
    for _run_id, rows in sorted(runs, key=lambda item: item[0]):
        for row in rows:
            step += 1
            combined.append({**row, "optimizer_step": step})
    return combined


def _render(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Render progress plots, tolerating a missing matplotlib install."""
    try:
        return render_training_plots(rows, out_dir)
    except PlotsUnavailable:
        return []


def push_checkpoint_bundle(bundle_dir: str | Path, *, bucket: str = DEFAULT_BUCKET) -> str:
    """Upload the local bundle and refresh its cross-run rollups.

    The checkpoint name and run id are read from the bundle's ``meta.json`` (the
    pipeline's single source of truth). Renders this run's plots and the all-runs
    plots (pulling sibling runs first), promotes ``best.json`` only when this
    run's metric beats the recorded best, and rewrites ``index.json``. Returns the
    checkpoint prefix that was written.
    """
    bundle = Path(bundle_dir)
    meta = _read_json(bundle / META_FILE) or {}
    name, run_id = meta.get("checkpoint_name"), meta.get("run_id")
    if not name or not run_id:
        raise ValueError(f"bundle meta is missing checkpoint_name/run_id: {bundle / META_FILE}")
    prefix = checkpoint_prefix(name)
    local_metrics = _read_jsonl(bundle / METRICS_FILE)
    pairs: list[tuple[str | Path, str]] = [
        (bundle / LATEST_FILE, f"{prefix}/{LATEST_FILE}"),
        (bundle / METRICS_FILE, f"{prefix}/runs/{run_id}{_METRICS_SUFFIX}"),
        (bundle / META_FILE, f"{prefix}/runs/{run_id}.meta.json"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        history = _fetch_other_runs(name, run_id, work / "history", bucket=bucket)
        history[run_id] = local_metrics
        for plot in _render(local_metrics, work / "current"):
            pairs.append((plot, f"{prefix}/plots/{run_id}/{plot.name}"))
        combined = _combined_rows(list(history.items()))
        for plot in _render(combined, work / "all_runs"):
            pairs.append((plot, f"{prefix}/plots/all_runs/{plot.name}"))

        index = _build_index(bundle, meta, name, run_id, bucket=bucket)
        if index.pop("_promote_best", False):
            pairs.append((bundle / BEST_FILE, f"{prefix}/{BEST_FILE}"))
        index_path = work / INDEX_FILE
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pairs.append((index_path, f"{prefix}/{INDEX_FILE}"))

        megabytes = sum(Path(local).stat().st_size for local, _ in pairs) / 1e6
        upload_files(pairs, bucket=bucket)
    print(
        f"[checkpoint-upload] pushed {len(pairs)} files ({megabytes:.2f} MB) to {bucket}/{prefix}"
    )
    return prefix


def _build_index(
    bundle: Path, meta: dict[str, Any], name: str, run_id: str, *, bucket: str
) -> dict[str, Any]:
    """Merge this run into the remote ``index.json``, deciding the best pointer.

    The returned dict carries a private ``_promote_best`` flag telling the caller
    whether ``best.json`` should be uploaded (i.e. this run is the new best).
    """
    best = _read_json(bundle / BEST_FILE)
    metric = None if best is None else best.get("checkpoint_metric")

    with tempfile.TemporaryDirectory() as tmp:
        remote_index = _read_json(
            download_file(
                f"{checkpoint_prefix(name)}/{INDEX_FILE}",
                Path(tmp) / INDEX_FILE,
                bucket=bucket,
                missing_ok=True,
            )
            or Path(tmp) / "absent"
        ) or {"name": name, "best": None, "runs": []}

    entry = {
        "run_id": run_id,
        "metric": metric,
        "iteration": meta.get("iteration"),
        "optimizer_step": meta.get("optimizer_step"),
        "warm_started_from": meta.get("warm_started_from"),
        "updated_at": meta.get("updated_at"),
    }
    runs = [r for r in remote_index.get("runs", []) if r.get("run_id") != run_id]
    runs.append(entry)

    prev_best = remote_index.get("best")
    prev_metric = None if prev_best is None else prev_best.get("metric")
    promote = metric is not None and (prev_metric is None or metric >= prev_metric)
    best_pointer = entry if promote else prev_best
    return {
        "name": name,
        "best": best_pointer,
        "runs": sorted(runs, key=lambda r: str(r.get("run_id"))),
        "_promote_best": promote,
    }


class PeriodicCheckpointUploader:
    """Training callback that pushes the bundle to HF every ``every_hours``.

    Uploads on ``checkpoint``/``completed`` events once the interval elapses, and
    once more on :meth:`close`, so the final best model is always pushed. Upload
    failures are reported but never interrupt training.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        bucket: str = DEFAULT_BUCKET,
        every_hours: float = 3.0,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.bucket = bucket
        self.interval_s = every_hours * 3600.0
        self._last_upload = time.monotonic()

    def _upload(self) -> None:
        if not (self.bundle_dir / LATEST_FILE).is_file():
            return  # no checkpoint written yet
        try:
            push_checkpoint_bundle(self.bundle_dir, bucket=self.bucket)
            self._last_upload = time.monotonic()
        except Exception as exc:  # never let an upload hiccup kill a long run
            print(f"[checkpoint-upload] skipped: {type(exc).__name__}: {exc}")

    def on_event(self, event: Any) -> None:
        if event.phase not in ("checkpoint", "completed"):
            return
        if time.monotonic() - self._last_upload >= self.interval_s:
            self._upload()

    def close(self) -> None:
        self._upload()
