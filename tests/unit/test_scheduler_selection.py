"""Pure task-selection and slot-limit logic for the Kaggle scheduler."""

from __future__ import annotations

from ac_zero.scheduler.models import Limits, Queue, SchedulerState, Task, TaskRuntimeState
from ac_zero.scheduler.selection import active_counts, run_is_live, select_launches

NOW = "2026-07-08T12:00:00Z"


def _task(task_id: str, **kw: object) -> Task:
    base = dict(id=task_id, mode="generation", notebook_slug="u/n", notebook_dir="d")
    base.update(kw)
    return Task(**base)  # type: ignore[arg-type]


def _queue(tasks: list[Task], **limits: int) -> Queue:
    return Queue(limits=Limits(**limits), tasks=tasks)


def test_inactive_task_is_skipped() -> None:
    queue = _queue([_task("a", active=False)])
    selected, decisions = select_launches(queue, SchedulerState(), now=NOW)
    assert selected == []
    assert decisions[0].reason == "inactive"


def test_remaining_runs_none_is_infinite() -> None:
    queue = _queue([_task("a", remaining_runs=None)])
    selected, _ = select_launches(queue, SchedulerState(), now=NOW)
    assert [t.id for t in selected] == ["a"]


def test_remaining_runs_zero_not_launched() -> None:
    queue = _queue([_task("a", remaining_runs=0)])
    selected, decisions = select_launches(queue, SchedulerState(), now=NOW)
    assert selected == []
    assert "exhausted" in decisions[0].reason


def test_priority_ordering_and_launch_budget() -> None:
    queue = _queue(
        [_task("low", priority=1), _task("high", priority=100)],
        max_launches_per_tick=1,
    )
    selected, _ = select_launches(queue, SchedulerState(), now=NOW)
    assert [t.id for t in selected] == ["high"]


def test_tie_break_oldest_last_launch_then_id() -> None:
    queue = _queue(
        [_task("b", priority=5), _task("a", priority=5), _task("c", priority=5)],
        max_launches_per_tick=3,
        max_total_active=3,
        max_cpu_active=3,
    )
    state = SchedulerState(
        tasks={
            "b": TaskRuntimeState(last_launch_at="2026-07-08T09:00:00Z"),
            "a": TaskRuntimeState(last_launch_at="2026-07-08T11:00:00Z"),
            # c never launched -> should sort first (empty < any timestamp)
        }
    )
    selected, _ = select_launches(queue, state, now=NOW)
    assert [t.id for t in selected] == ["c", "b", "a"]


def test_slot_limits_respected_per_accelerator() -> None:
    queue = _queue(
        [_task("g1", accelerator="gpu"), _task("g2", accelerator="gpu")],
        max_gpu_active=1,
        max_launches_per_tick=5,
    )
    selected, decisions = select_launches(queue, SchedulerState(), now=NOW)
    assert [t.id for t in selected] == ["g1"]
    assert any("no free gpu slot" in d.reason for d in decisions)


def test_total_active_limit_counts_live_runs() -> None:
    queue = _queue(
        [_task("a"), _task("b")],
        max_total_active=1,
        max_cpu_active=5,
        max_launches_per_tick=5,
    )
    state = SchedulerState(
        tasks={"a": TaskRuntimeState(active_run_id="a-run", last_heartbeat_at=NOW)}
    )
    counts = active_counts(queue, state, now=NOW, stale_minutes=180)
    assert counts["total"] == 1
    selected, decisions = select_launches(queue, state, now=NOW)
    # 'a' already has a live run -> skipped; 'b' blocked by the total limit.
    assert selected == []
    assert any("already has an active run" in d.reason for d in decisions)
    assert any("no free cpu slot" in d.reason for d in decisions)


def test_task_with_live_run_skipped_unless_forced() -> None:
    queue = _queue([_task("a")], max_launches_per_tick=5)
    state = SchedulerState(
        tasks={"a": TaskRuntimeState(active_run_id="a-run", last_heartbeat_at=NOW)}
    )
    skipped, _ = select_launches(queue, state, now=NOW)
    assert skipped == []
    forced, _ = select_launches(queue, state, now=NOW, force_task_id="a", force=True)
    assert [t.id for t in forced] == ["a"]


def test_stale_heartbeat_frees_the_slot() -> None:
    rt = TaskRuntimeState(active_run_id="a-run", last_heartbeat_at="2026-07-08T06:00:00Z")
    assert run_is_live(rt, now=NOW, stale_minutes=180) is False
    fresh = TaskRuntimeState(active_run_id="a-run", last_heartbeat_at="2026-07-08T11:59:00Z")
    assert run_is_live(fresh, now=NOW, stale_minutes=180) is True


def test_max_launches_override_caps_budget() -> None:
    queue = _queue(
        [_task("a", priority=3), _task("b", priority=2)],
        max_launches_per_tick=5,
        max_cpu_active=5,
        max_total_active=5,
    )
    selected, _ = select_launches(queue, SchedulerState(), now=NOW, max_launches_override=1)
    assert [t.id for t in selected] == ["a"]
