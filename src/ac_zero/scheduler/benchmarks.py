"""The evaluation queue: which trained checkpoints are waiting to be benchmarked.

A training run earns an evaluation by reaching a decent self-play success rate.
Rather than have the training notebook push itself onto a queue, the controller
*pulls*: every tick it reads each training task's ``model_checkpoints/<name>/
index.json`` -- already published by the training pipeline -- and enqueues any
best-model whose metric clears the threshold. That keeps the gate in one place,
makes it idempotent (a run is keyed by ``(checkpoint_name, run_id)`` and only
ever enqueued once), and means a checkpoint that crossed the line while the
scheduler was down is still picked up on the next tick.

The queue itself is one document on the state repo, written as part of the same
commit as the rest of the tick::

    queue/benchmark_queue.json
        {"pending": [...], "dispatched": [...]}

``dispatched`` is the memory that stops a completed evaluation from being
re-queued forever; it is trimmed, since only the recent tail is ever consulted.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ac_zero.scheduler.backend import StateBackend
from ac_zero.scheduler.models import Queue, utc_now

BENCHMARK_QUEUE_PATH = "queue/benchmark_queue.json"
DEFAULT_METRIC_THRESHOLD = 0.30
MAX_DISPATCHED_HISTORY = 200

Logger = Callable[[str], None]


@dataclass(slots=True)
class PendingEvaluation:
    """One checkpoint waiting for -- or sent to -- a benchmark run."""

    checkpoint_name: str
    run_id: str
    metric: float
    enqueued_at: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.checkpoint_name, self.run_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingEvaluation:
        return cls(
            checkpoint_name=str(data.get("checkpoint_name", "")),
            run_id=str(data.get("run_id", "")),
            metric=float(data.get("metric", 0.0)),
            enqueued_at=str(data.get("enqueued_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "run_id": self.run_id,
            "metric": self.metric,
            "enqueued_at": self.enqueued_at,
        }


@dataclass(slots=True)
class BenchmarkQueue:
    """Parsed ``queue/benchmark_queue.json``."""

    pending: list[PendingEvaluation] = field(default_factory=list)
    dispatched: list[PendingEvaluation] = field(default_factory=list)

    @classmethod
    def load(cls, backend: StateBackend) -> BenchmarkQueue:
        """Read the queue, treating a missing or malformed document as empty."""
        raw = backend.read_text(BENCHMARK_QUEUE_PATH)
        if raw is None:
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls(
            pending=[PendingEvaluation.from_dict(e) for e in data.get("pending") or []],
            dispatched=[PendingEvaluation.from_dict(e) for e in data.get("dispatched") or []],
        )

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "pending": [e.to_dict() for e in self.pending],
                    "dispatched": [e.to_dict() for e in self.dispatched],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def known_keys(self) -> set[tuple[str, str]]:
        """Every run this queue has already seen, pending or dispatched."""
        return {e.key for e in self.pending} | {e.key for e in self.dispatched}

    def enqueue(self, entry: PendingEvaluation) -> bool:
        """Add ``entry`` unless this exact run was queued before. Reports whether it landed."""
        if entry.key in self.known_keys():
            return False
        self.pending.append(
            PendingEvaluation(
                entry.checkpoint_name, entry.run_id, entry.metric, entry.enqueued_at or utc_now()
            )
        )
        return True

    def take(self) -> PendingEvaluation | None:
        """Pop the highest-metric pending entry and record it as dispatched.

        Best-first rather than oldest-first: evaluation capacity is scarce, so the
        most promising checkpoint should be the one that gets it.
        """
        if not self.pending:
            return None
        best = max(self.pending, key=lambda e: e.metric)
        self.pending.remove(best)
        self.dispatched.append(best)
        del self.dispatched[:-MAX_DISPATCHED_HISTORY]
        return best


def _checkpoint_name_for(task_config: dict[str, Any]) -> str | None:
    """Derive the checkpoint name a training task writes under.

    Uses the very code the trainer uses, so the name here is the name there by
    construction. A task whose config cannot be parsed as a training config is
    skipped rather than guessed at.
    """
    from ac_zero.training.checkpointing.checkpoint_name import derive_checkpoint_name
    from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig

    explicit = task_config.get("checkpoint_name")
    if explicit:
        return str(explicit)
    try:
        return derive_checkpoint_name(TrainingPipelineConfig.from_mapping(task_config))
    except (TypeError, ValueError, KeyError):
        return None


def _read_index(name: str, *, bucket: str) -> dict[str, Any] | None:
    from ac_zero.datasets.hub import download_file

    remote = f"model_checkpoints/{name}/index.json"
    with TemporaryDirectory() as tmp:
        local = download_file(remote, Path(tmp) / "index.json", bucket=bucket, missing_ok=True)
        if local is None:
            return None
        parsed = json.loads(local.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else None


def scan_for_ready_checkpoints(
    queue: Queue,
    benchmark_queue: BenchmarkQueue,
    *,
    bucket: str,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
    log: Logger = print,
) -> list[PendingEvaluation]:
    """Enqueue every training task's best model that now clears ``threshold``.

    The metric read is the one the training pipeline itself selects best models
    by -- the self-play success-rate EMA on navigation runs, the return EMA
    otherwise -- so the gate means "this model got decent at the task it was
    trained on", not a separate notion of accuracy.

    Bucket reads are best-effort: a missing or unreadable index just means this
    task has nothing to offer yet, never a failed tick.
    """
    added: list[PendingEvaluation] = []
    for task in queue.tasks:
        if task.mode != "training":
            continue
        name = _checkpoint_name_for(task.config)
        if name is None:
            log(f"  benchmark-gate {task.id}: cannot derive a checkpoint name; skipped")
            continue
        try:
            index = _read_index(name, bucket=bucket)
        except Exception as exc:  # a bucket hiccup must not fail the tick
            log(f"  benchmark-gate {task.id}: could not read {name}/index.json ({exc})")
            continue
        best = (index or {}).get("best")
        if not isinstance(best, dict):
            continue
        metric, run_id = best.get("metric"), best.get("run_id")
        if metric is None or run_id is None:
            continue
        if float(metric) < threshold:
            continue
        entry = PendingEvaluation(name, str(run_id), float(metric))
        if benchmark_queue.enqueue(entry):
            added.append(entry)
            log(f"  benchmark-gate {task.id}: queued {name} (metric={float(metric):.3f})")
    return added
