"""The on-disk bundle a run keeps ready to push to Hugging Face.

A bundle is a single directory the training pipeline keeps current as it runs::

    <run>/model_checkpoint/
        best.json      # best-by-metric checkpoint so far (the warm-start model)
        latest.json    # most recent checkpoint (exact resume of this run)
        metrics.jsonl  # every per-update metric row, for the progress plots
        meta.json      # run provenance: name, run id, best metric, lineage

:class:`hub_checkpoints.push_checkpoint_bundle` uploads this directory (plus the
plots and cross-run history it renders) under ``model_checkpoints/<name>/``. The
split between "write the bundle" (here, always on) and "upload it" (there, gated
on a token and an interval) keeps the pipeline free of any network dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BEST_FILE = "best.json"
LATEST_FILE = "latest.json"
METRICS_FILE = "metrics.jsonl"
META_FILE = "meta.json"


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class CheckpointBundle:
    """Keeps ``best.json``/``latest.json``/``metrics.jsonl``/``meta.json`` current.

    ``best`` tracks the highest metric handed to :meth:`save_checkpoint`; the best
    model's weights are written the moment that metric improves, so an early stop
    still leaves the best-so-far model on disk.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_metric: float | None = None

    def save_checkpoint(self, payload: dict[str, Any], *, metric: float | None) -> bool:
        """Write ``latest.json``; also write ``best.json`` when ``metric`` improves.

        Returns whether this checkpoint became the new best. A ``None`` metric
        (no self-play statistics yet) only updates ``latest``.
        """
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        _atomic_write(self.directory / LATEST_FILE, text)
        if metric is not None and (self.best_metric is None or metric > self.best_metric):
            self.best_metric = metric
            _atomic_write(self.directory / BEST_FILE, text)
            return True
        return False

    def save_metrics(self, rows: list[dict[str, Any]]) -> None:
        """Overwrite ``metrics.jsonl`` with the run's per-update metric rows."""
        _atomic_write(
            self.directory / METRICS_FILE,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )

    def save_meta(self, meta: dict[str, Any]) -> None:
        """Overwrite ``meta.json`` with the run's provenance."""
        _atomic_write(self.directory / META_FILE, json.dumps(meta, indent=2, sort_keys=True) + "\n")

    def has_checkpoint(self) -> bool:
        """Whether at least one checkpoint has been written (safe to upload)."""
        return (self.directory / LATEST_FILE).is_file()
