"""Pure task-selection and slot logic (no IO, fully unit-testable).

Given a :class:`Queue` and a :class:`SchedulerState`, decide which tasks to
launch this tick. The controller wraps these with Kaggle/HF side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

from ac_zero.scheduler.models import (
    Limits,
    Queue,
    SchedulerState,
    Task,
    TaskRuntimeState,
    parse_iso,
)


@dataclass(slots=True)
class Decision:
    """Why a task was or was not launched -- recorded for the state log."""

    task_id: str
    launch: bool
    reason: str


def run_is_live(rt: TaskRuntimeState, *, now: str, stale_minutes: int) -> bool:
    """Whether the task holds a run we still believe is executing.

    A run is live if it has an ``active_run_id`` and its most recent signal
    (heartbeat, else launch time) is within the stale window. A run whose
    heartbeat has gone stale is treated as no longer occupying a slot.
    """
    if rt.active_run_id is None:
        return False
    anchor = parse_iso(rt.last_heartbeat_at) or parse_iso(rt.last_launch_at)
    ref = parse_iso(now)
    if anchor is None or ref is None:
        return True  # can't prove staleness -> assume still live (avoid dup launch)
    return ref - anchor < timedelta(minutes=stale_minutes)


def active_counts(
    queue: Queue, state: SchedulerState, *, now: str, stale_minutes: int
) -> dict[str, int]:
    """Count live runs in total and per accelerator across the queue."""
    counts = {"total": 0, "cpu": 0, "gpu": 0}
    for task in queue.tasks:
        rt = state.tasks.get(task.id)
        if rt is None or not run_is_live(rt, now=now, stale_minutes=stale_minutes):
            continue
        counts["total"] += 1
        counts[task.accelerator] = counts.get(task.accelerator, 0) + 1
    return counts


def _slot_available(counts: dict[str, int], accelerator: str, limits: Limits) -> bool:
    if counts["total"] >= limits.max_total_active:
        return False
    if accelerator == "gpu":
        return counts.get("gpu", 0) < limits.max_gpu_active
    return counts.get("cpu", 0) < limits.max_cpu_active


def _order_key(task: Task, rt: TaskRuntimeState | None) -> tuple[int, str, str]:
    # Higher priority first (negate), then oldest last-launch first (empty
    # string sorts before any ISO timestamp -> never-launched wins), then id.
    last_launch = (rt.last_launch_at if rt else None) or ""
    return (-task.priority, last_launch, task.id)


def eligible(
    task: Task,
    rt: TaskRuntimeState | None,
    *,
    now: str,
    stale_minutes: int,
    forced: bool,
    blocked_reason: str | None = None,
) -> tuple[bool, str]:
    """Whether ``task`` may launch, ignoring free-slot availability.

    ``blocked_reason`` lets the controller veto a task for a reason this pure
    module cannot see -- a benchmark task with nothing queued to evaluate -- while
    keeping the explanation in the decision log alongside every other one.
    """
    if not task.active:
        return False, "inactive"
    if blocked_reason:
        return False, blocked_reason
    if task.remaining_runs is not None and task.remaining_runs <= 0:
        return False, "remaining_runs exhausted"
    if not forced and rt is not None and run_is_live(rt, now=now, stale_minutes=stale_minutes):
        return False, "already has an active run"
    return True, "eligible"


def select_launches(
    queue: Queue,
    state: SchedulerState,
    *,
    now: str,
    force_task_id: str | None = None,
    force: bool = False,
    max_launches_override: int | None = None,
    blocked: Mapping[str, str] | None = None,
) -> tuple[list[Task], list[Decision]]:
    """Pick tasks to launch this tick, respecting eligibility and slot limits.

    ``force_task_id`` is moved to the front of the ordering. When ``force`` is
    also set, that task may launch even if it already has an active run (slots
    still apply). ``blocked`` maps a task id to the reason it may not launch this
    tick. Returns the chosen tasks and a decision log for every task.
    """
    limits = queue.limits
    counts = active_counts(queue, state, now=now, stale_minutes=limits.stale_heartbeat_minutes)

    ordered = sorted(queue.tasks, key=lambda t: _order_key(t, state.tasks.get(t.id)))
    if force_task_id:
        ordered.sort(key=lambda t: 0 if t.id == force_task_id else 1)

    budget = limits.max_launches_per_tick
    if max_launches_override is not None:
        budget = min(budget, max_launches_override) if budget > 0 else max_launches_override
    budget = max(0, budget)

    selected: list[Task] = []
    decisions: list[Decision] = []
    for task in ordered:
        rt = state.tasks.get(task.id)
        forced = force and task.id == force_task_id
        ok, reason = eligible(
            task,
            rt,
            now=now,
            stale_minutes=limits.stale_heartbeat_minutes,
            forced=forced,
            blocked_reason=(blocked or {}).get(task.id),
        )
        if not ok:
            decisions.append(Decision(task.id, False, reason))
            continue
        if len(selected) >= budget:
            decisions.append(Decision(task.id, False, "launch budget reached"))
            continue
        if not _slot_available(counts, task.accelerator, limits):
            decisions.append(Decision(task.id, False, f"no free {task.accelerator} slot"))
            continue
        selected.append(task)
        counts["total"] += 1
        counts[task.accelerator] = counts.get(task.accelerator, 0) + 1
        decisions.append(Decision(task.id, True, "launching" + (" (forced)" if forced else "")))
    return selected, decisions
