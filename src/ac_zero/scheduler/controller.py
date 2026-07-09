"""Orchestrate one scheduler tick.

Read state -> acquire lease -> reconcile active runs -> select launches ->
push Kaggle notebooks -> decrement counters -> write state back -> release lease.

Every step is idempotent and safe to re-run: a crashed tick leaves the lease to
expire and the next tick reconciles from the persisted state and Kaggle status.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ac_zero.scheduler.kaggle import KaggleClient
from ac_zero.scheduler.models import Queue, SchedulerState, Task, utc_now
from ac_zero.scheduler.runtime import (
    build_runtime_config,
    patch_kernel_metadata,
    write_runtime_config,
)
from ac_zero.scheduler.selection import Decision, run_is_live, select_launches
from ac_zero.scheduler.store import RUNTIME_CONFIG_LATEST, Snapshot, StateConflict, StateStore

Logger = Callable[[str], None]
MAX_DECISION_HISTORY = 20


@dataclass(slots=True)
class SchedulerConfig:
    state_repo_id: str
    secrets_dataset: str
    owner: str
    github_run_id: str
    state_repo_type: str = "dataset"
    lease_ttl_minutes: int = 30
    force_task_id: str | None = None
    force: bool = False
    dry_run: bool = False
    max_launches_override: int | None = None


@dataclass(slots=True)
class TickReport:
    launched: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def _sanitize_run_id(task_id: str, now: str, index: int) -> str:
    stamp = now.replace(":", "-")
    suffix = f"-{index}" if index else ""
    return f"{task_id}-{stamp}{suffix}"


def _reconcile(
    store: StateStore,
    kaggle: KaggleClient,
    queue: Queue,
    state: SchedulerState,
    *,
    now: str,
    log: Logger,
) -> None:
    """Fold Kaggle status + notebook heartbeat files into per-task run state.

    Marks active runs finished (terminal Kaggle status), stale (heartbeat older
    than the limit), or leaves them live. Never restores ``remaining_runs``: a
    finished run -- success or failure -- stays counted.
    """
    stale_minutes = queue.limits.stale_heartbeat_minutes
    for task in queue.tasks:
        rt = state.tasks.get(task.id)
        if rt is None or rt.active_run_id is None:
            continue

        heartbeat = _read_run_file(store, rt.active_run_id)
        if heartbeat is not None:
            rt.last_heartbeat_at = heartbeat.get("heartbeat_at") or rt.last_heartbeat_at
            rt.latest_status = heartbeat.get("status") or rt.latest_status
            if heartbeat.get("error"):
                rt.latest_error = str(heartbeat["error"])

        status = kaggle.status(task.notebook_slug)
        if status is not None:
            rt.kaggle_status = status

        note_status = (rt.latest_status or "").lower()
        finished_by_notebook = note_status in {"finished", "failed", "stopped", "complete", "error"}
        if kaggle.is_terminal(status) or finished_by_notebook:
            rt.active_run_id = None
            rt.last_finish_at = now
            rt.latest_status = rt.latest_status or status or "complete"
            log(f"  reconcile {task.id}: run finished (status={rt.latest_status})")
        elif not run_is_live(rt, now=now, stale_minutes=stale_minutes):
            rt.active_run_id = None
            rt.last_finish_at = now
            rt.latest_status = "stale"
            log(f"  reconcile {task.id}: run marked stale (no heartbeat in {stale_minutes} min)")


def _read_run_file(store: StateStore, run_id: str) -> dict[str, Any] | None:
    raw = store.backend.read_text(f"runs/{run_id}.json")
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _launch(
    task: Task,
    run_id: str,
    kaggle: KaggleClient,
    config: SchedulerConfig,
    state: SchedulerState,
    *,
    now: str,
    log: Logger,
) -> dict[str, Any] | None:
    """Prepare inputs and push one Kaggle run.

    Returns the (secret-free) runtime config on a successful launch so the
    caller can archive it, or ``None`` if the launch failed.
    """
    runtime = build_runtime_config(
        task,
        run_id=run_id,
        state_repo_id=config.state_repo_id,
        state_repo_type=config.state_repo_type,
    )
    write_runtime_config(task.notebook_dir, runtime)
    patch_kernel_metadata(task.notebook_dir, task, secrets_dataset=config.secrets_dataset)

    rt = state.task_state(task.id)
    from ac_zero.scheduler.kaggle import KaggleError

    try:
        result = kaggle.push(task.notebook_dir)
    except (KaggleError, FileNotFoundError) as exc:
        rt.latest_error = str(exc)
        rt.latest_status = "launch_failed"
        log(f"  LAUNCH FAILED {task.id}: {exc}")
        return None

    # remaining_runs counts *launches*: decrement now, never restore on later
    # failure. A task that hits zero is deactivated.
    if task.remaining_runs is not None:
        task.remaining_runs -= 1
        if task.remaining_runs <= 0:
            task.active = False
    rt.active_run_id = run_id
    rt.kaggle_slug = task.notebook_slug
    rt.kaggle_status = "queued"
    rt.last_launch_at = now
    rt.last_heartbeat_at = None
    rt.last_finish_at = None
    rt.latest_status = "launched"
    rt.latest_error = None
    log(f"  LAUNCHED {task.id} run_id={run_id} ({result.output.splitlines()[-1:] or ['ok']})")
    return runtime


def run_tick(
    store: StateStore,
    kaggle: KaggleClient,
    config: SchedulerConfig,
    *,
    now: str | None = None,
    log: Logger = print,
) -> TickReport:
    now = now or utc_now()
    report = TickReport(dry_run=config.dry_run)
    last_runtime: dict[str, Any] | None = None

    snapshot = store.load()
    queue, state = snapshot.queue, snapshot.state
    log(f"state repo: {config.state_repo_id} (base_sha={snapshot.base_sha})")
    log(f"loaded {len(queue.tasks)} task(s); limits={queue.limits}")
    for task in queue.tasks:
        rt = state.tasks.get(task.id)
        live = rt is not None and run_is_live(
            rt, now=now, stale_minutes=queue.limits.stale_heartbeat_minutes
        )
        log(
            f"  task {task.id}: active={task.active} remaining_runs={task.remaining_runs} "
            f"priority={task.priority} accel={task.accelerator} live_run={live}"
        )

    if not config.dry_run:
        from ac_zero.scheduler.store import LeaseError

        try:
            store.acquire_lease(
                owner=config.owner,
                github_run_id=config.github_run_id,
                ttl_minutes=config.lease_ttl_minutes,
            )
        except LeaseError as exc:
            log(f"could not acquire scheduler lease: {exc}")
            report.errors.append(str(exc))
            return report

    state.last_scheduler_started_at = now
    _reconcile(store, kaggle, queue, state, now=now, log=log)

    if state.scheduler_paused:
        log("scheduler is paused; not launching.")
    elif state.stop_launching:
        log("stop_launching is set; draining -- no new launches, active runs kept.")
    else:
        selected, decisions = select_launches(
            queue,
            state,
            now=now,
            force_task_id=config.force_task_id,
            force=config.force,
            max_launches_override=config.max_launches_override,
        )
        report.decisions = decisions
        for decision in decisions:
            log(
                f"  decision {decision.task_id}: {'LAUNCH' if decision.launch else 'skip'} "
                f"-- {decision.reason}"
            )

        if selected and not config.dry_run and not store.owns_lease(config.owner):
            msg = "lost the scheduler lease before launching; aborting launches."
            log(msg)
            report.errors.append(msg)
            selected = []

        for index, task in enumerate(selected):
            run_id = _sanitize_run_id(task.id, now, index)
            if config.dry_run:
                log(f"  DRY-RUN would launch {task.id} as run_id={run_id}")
                report.launched.append(run_id)
                continue
            runtime = _launch(task, run_id, kaggle, config, state, now=now, log=log)
            if runtime is not None:
                report.launched.append(run_id)
                last_runtime = runtime

    _record_decisions(state, report.decisions)
    state.last_scheduler_finished_at = utc_now()

    if config.dry_run:
        log("dry-run: state not written.")
        return report

    extra = None
    if last_runtime is not None:
        extra = {RUNTIME_CONFIG_LATEST: json.dumps(last_runtime, indent=2) + "\n"}
    _save_with_retry(store, snapshot, log=log, report=report, extra_files=extra)
    store.release_lease(config.owner)
    return report


def _record_decisions(state: SchedulerState, decisions: list[Decision]) -> None:
    entry = {
        "at": utc_now(),
        "decisions": [
            {"task_id": d.task_id, "launch": d.launch, "reason": d.reason} for d in decisions
        ],
    }
    state.last_decisions = [entry, *state.last_decisions][:MAX_DECISION_HISTORY]


def _save_with_retry(
    store: StateStore,
    snapshot: Snapshot,
    *,
    log: Logger,
    report: TickReport,
    extra_files: dict[str, str] | None = None,
) -> None:
    try:
        store.save(snapshot, message=f"scheduler tick {utc_now()}", extra_files=extra_files)
        log("state written.")
    except StateConflict as exc:
        # scheduler_state.json is single-writer (the controller) and lease-guarded,
        # so a conflict is an anomaly. Our state already reflects this tick's
        # launches, so re-read the head and overwrite once, loudly.
        log(f"WARNING: state changed under us ({exc}); retrying save over current head.")
        snapshot.base_sha = store.backend.head_sha()
        try:
            store.save(
                snapshot,
                message=f"scheduler tick {utc_now()} (conflict override)",
                extra_files=extra_files,
            )
            log("state written after conflict retry.")
        except StateConflict as exc2:
            log(f"ERROR: failed to write state after retry: {exc2}")
            report.errors.append(str(exc2))
