"""Dataclasses for the Kaggle-run scheduler state.

Two documents live in the private Hugging Face state dataset repo:

* ``queue.yaml`` -- the human-editable desired task definitions plus the mutable
  scheduling knobs (``active``, ``remaining_runs``, ``priority``,
  ``stop_after_current_iteration``).
* ``scheduler_state.json`` -- machine-owned runtime state: global flags plus a
  per-task record of the currently active run, timestamps, and latest status.

All timestamps are ISO-8601 UTC strings (``...Z``). Storing them as strings keeps
(de)serialisation trivial and lets "oldest first" ordering fall out of a plain
lexicographic sort, since ISO-8601 UTC sorts chronologically.

``remaining_runs is None`` means *infinite* -- the task is never exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    """Current UTC time as an ISO-8601 ``...Z`` string (second precision)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 ``...Z`` timestamp, tolerating ``None``/blank/offset."""
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _keep_known(cls: type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys so hand-edited files with extra fields still load."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass(slots=True)
class Limits:
    """Configurable scheduler slot limits (all overridable from ``queue.yaml``)."""

    max_total_active: int = 5
    max_cpu_active: int = 5
    max_gpu_active: int = 1
    max_launches_per_tick: int = 1
    stale_heartbeat_minutes: int = 180

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Limits:
        return cls(**_keep_known(cls, data or {}))


@dataclass(slots=True)
class Task:
    """A desired, repeatable Kaggle run. Lives in ``queue.yaml``."""

    id: str
    mode: str
    notebook_slug: str
    notebook_dir: str
    active: bool = True
    remaining_runs: int | None = None  # None == infinite
    accelerator: str = "cpu"  # "cpu" | "gpu"
    priority: int = 0
    max_runtime_minutes: int = 705
    stop_after_current_iteration: bool = False
    # One-shot: the next launch abandons this task's checkpoint history and starts over
    # from its pretrained checkpoint (or from zero). The controller clears it on launch,
    # so setting it restarts the task exactly once rather than every tick.
    start_fresh: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(**_keep_known(cls, data))


@dataclass(slots=True)
class TaskRuntimeState:
    """Machine-owned per-task run state. Lives in ``scheduler_state.json``."""

    active_run_id: str | None = None
    kaggle_slug: str | None = None
    kaggle_status: str | None = None
    last_launch_at: str | None = None
    last_heartbeat_at: str | None = None
    last_finish_at: str | None = None
    latest_status: str | None = None
    latest_error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TaskRuntimeState:
        return cls(**_keep_known(cls, data or {}))


@dataclass(slots=True)
class Queue:
    """Parsed ``queue.yaml``."""

    version: int = 1
    # Restart *everything*: one edit at the top of the file instead of one per task.
    # Expanded into the per-task flags and cleared before any other scheduling work.
    start_fresh_all: bool = False
    limits: Limits = field(default_factory=Limits)
    tasks: list[Task] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Queue:
        return cls(
            version=int(data.get("version", 1)),
            start_fresh_all=bool(data.get("start_fresh_all", False)),
            limits=Limits.from_dict(data.get("limits")),
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "start_fresh_all": self.start_fresh_all,
            "limits": _asdict(self.limits),
            "tasks": [_asdict(t) for t in self.tasks],
        }

    def apply_start_fresh_all(self) -> bool:
        """Fan the global flag out onto every task and clear it; report whether it fired.

        Runs before anything else a tick does, so a task launched in the same tick the
        flag was set carries the fresh start rather than picking it up a tick later. The
        flag is cleared here and the per-task flags are cleared on launch, so each is a
        one-shot: the operator sets it once and the queue returns to steady state.
        """
        if not self.start_fresh_all:
            return False
        for task in self.tasks:
            task.start_fresh = True
        self.start_fresh_all = False
        return True


@dataclass(slots=True)
class SchedulerState:
    """Parsed ``scheduler_state.json``."""

    scheduler_paused: bool = False
    stop_launching: bool = False  # drain: no new launches, keep active runs
    last_scheduler_started_at: str | None = None
    last_scheduler_finished_at: str | None = None
    tasks: dict[str, TaskRuntimeState] = field(default_factory=dict)
    last_decisions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchedulerState:
        return cls(
            scheduler_paused=bool(data.get("scheduler_paused", False)),
            stop_launching=bool(data.get("stop_launching", False)),
            last_scheduler_started_at=data.get("last_scheduler_started_at"),
            last_scheduler_finished_at=data.get("last_scheduler_finished_at"),
            tasks={k: TaskRuntimeState.from_dict(v) for k, v in (data.get("tasks") or {}).items()},
            last_decisions=list(data.get("last_decisions") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_paused": self.scheduler_paused,
            "stop_launching": self.stop_launching,
            "last_scheduler_started_at": self.last_scheduler_started_at,
            "last_scheduler_finished_at": self.last_scheduler_finished_at,
            "tasks": {k: _asdict(v) for k, v in self.tasks.items()},
            "last_decisions": self.last_decisions,
        }

    def task_state(self, task_id: str) -> TaskRuntimeState:
        """Return the mutable runtime record for ``task_id``, creating it lazily."""
        rt = self.tasks.get(task_id)
        if rt is None:
            rt = TaskRuntimeState()
            self.tasks[task_id] = rt
        return rt


@dataclass(slots=True)
class Lease:
    """The scheduler lease file (``locks/scheduler_lease.json``)."""

    owner: str
    github_run_id: str
    acquired_at: str
    expires_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lease:
        return cls(**_keep_known(cls, data))

    def is_expired(self, *, now: str | None = None) -> bool:
        ref = parse_iso(now or utc_now())
        exp = parse_iso(self.expires_at)
        if ref is None or exp is None:
            return True
        return ref >= exp


def _asdict(obj: Any) -> dict[str, Any]:
    """Shallow dataclass -> dict without dataclasses.asdict's deep-copy cost."""
    return {f.name: getattr(obj, f.name) for f in fields(obj)}
