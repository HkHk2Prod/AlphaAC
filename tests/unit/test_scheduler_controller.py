"""End-to-end scheduler tick: launch counting, reconciliation, lease, dry-run.

Kaggle CLI and the HF Hub are both faked -- no network or credentials needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from ac_zero.scheduler.backend import MemoryStateBackend
from ac_zero.scheduler.controller import SchedulerConfig, run_tick
from ac_zero.scheduler.kaggle import KaggleClient
from ac_zero.scheduler.store import StateStore

NOW = "2026-07-08T12:00:00Z"


class FakeKaggleRunner:
    """Fake ``kaggle`` CLI: records pushes, replies with configurable status."""

    def __init__(self, *, push_rc: int = 0, status_map: dict[str, str] | None = None) -> None:
        self.push_rc = push_rc
        self.status_map = status_map or {}
        self.pushes: list[str] = []

    def __call__(self, argv: list[str]) -> CompletedProcess[str]:
        if argv[:3] == ["kaggle", "kernels", "push"]:
            self.pushes.append(argv[-1])
            out = "successfully pushed" if self.push_rc == 0 else "push error"
            return CompletedProcess(argv, self.push_rc, out, "")
        if argv[:3] == ["kaggle", "kernels", "status"]:
            status = self.status_map.get(argv[3])
            if status is None:
                return CompletedProcess(argv, 1, "", "not found")
            return CompletedProcess(argv, 0, f'{argv[3]} has status "{status}"', "")
        return CompletedProcess(argv, 0, "", "")


def _notebook_dir(tmp_path: Path) -> Path:
    nb = tmp_path / "nb"
    nb.mkdir()
    (nb / "kernel-metadata.json").write_text(
        json.dumps({"id": "u/runner", "code_file": "runner.ipynb", "dataset_sources": []}),
        encoding="utf-8",
    )
    minimal_nb = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["#"]}],
        "nbformat": 4,
    }
    (nb / "runner.ipynb").write_text(json.dumps(minimal_nb), encoding="utf-8")
    return nb


def _queue_yaml(nb_dir: Path, *, remaining_runs: str = "null", extra: str = "") -> str:
    return f"""
version: 1
limits:
  max_total_active: 5
  max_cpu_active: 5
  max_gpu_active: 1
  max_launches_per_tick: 2
  stale_heartbeat_minutes: 180
tasks:
  - id: gen
    mode: generation
    accelerator: cpu
    notebook_slug: u/runner
    notebook_dir: {nb_dir}
    remaining_runs: {remaining_runs}
    config:
      rank: 2
{extra}
"""


def _store(files: dict[str, str]) -> tuple[StateStore, MemoryStateBackend]:
    backend = MemoryStateBackend(files)
    return StateStore(backend), backend


def _config(**kw: object) -> SchedulerConfig:
    base: dict[str, object] = dict(
        state_repo_id="u/state",
        secrets_dataset="u/runtime-secrets",
        owner="gha-1",
        github_run_id="1",
    )
    base.update(kw)
    return SchedulerConfig(**base)  # type: ignore[arg-type]


def _silent(_: str) -> None:
    pass


def test_launch_pushes_and_decrements_remaining_runs(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({"queue.yaml": _queue_yaml(nb, remaining_runs="3")})
    fake = FakeKaggleRunner()
    report = run_tick(store, KaggleClient(runner=fake), _config(), now=NOW, log=_silent)

    assert len(report.launched) == 1
    assert fake.pushes == [str(nb)]
    snap = store.load()
    assert snap.queue.tasks[0].remaining_runs == 2
    rt = snap.state.tasks["gen"]
    assert rt.active_run_id == report.launched[0]
    assert rt.last_launch_at == NOW
    assert rt.latest_status == "launched"


def test_remaining_runs_reaching_zero_deactivates(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({"queue.yaml": _queue_yaml(nb, remaining_runs="1")})
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)
    snap = store.load()
    assert snap.queue.tasks[0].remaining_runs == 0
    assert snap.queue.tasks[0].active is False


def test_infinite_remaining_runs_never_decrements(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({"queue.yaml": _queue_yaml(nb, remaining_runs="null")})
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)
    snap = store.load()
    assert snap.queue.tasks[0].remaining_runs is None
    assert snap.queue.tasks[0].active is True


def test_launch_failure_does_not_decrement(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({"queue.yaml": _queue_yaml(nb, remaining_runs="3")})
    report = run_tick(
        store, KaggleClient(runner=FakeKaggleRunner(push_rc=1)), _config(), now=NOW, log=_silent
    )
    assert report.launched == []
    snap = store.load()
    assert snap.queue.tasks[0].remaining_runs == 3  # not restored, not spent
    assert snap.state.tasks["gen"].latest_status == "launch_failed"
    assert snap.state.tasks["gen"].latest_error


def test_later_failure_status_does_not_restore_remaining_runs(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({"queue.yaml": _queue_yaml(nb, remaining_runs="2")})
    # Tick 1: launch (remaining 2 -> 1), run is now active.
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)
    assert store.load().queue.tasks[0].remaining_runs == 1

    # Tick 2: Kaggle reports the run errored. Reconcile marks it finished; the
    # failed run stays counted -- remaining_runs must not go back up.
    later = "2026-07-08T14:00:00Z"
    fake2 = FakeKaggleRunner(status_map={"u/runner": "error"})
    cfg = _config(max_launches_override=0)
    run_tick(store, KaggleClient(runner=fake2), cfg, now=later, log=_silent)
    snap = store.load()
    assert snap.queue.tasks[0].remaining_runs == 1
    assert snap.state.tasks["gen"].active_run_id is None
    assert snap.state.tasks["gen"].last_finish_at == later


def test_dry_run_does_not_mutate_state(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, backend = _store({"queue.yaml": _queue_yaml(nb, remaining_runs="3")})
    head_before = backend.head_sha()
    fake = FakeKaggleRunner()
    report = run_tick(store, KaggleClient(runner=fake), _config(dry_run=True), now=NOW, log=_silent)
    assert report.dry_run is True
    assert report.launched  # decided to launch...
    assert fake.pushes == []  # ...but did not push
    assert backend.head_sha() == head_before  # ...and did not write state
    assert store.load().queue.tasks[0].remaining_runs == 3


def test_foreign_live_lease_prevents_launch(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    lease = {
        "owner": "gha-999",
        "github_run_id": "999",
        "acquired_at": NOW,
        # Far future so the lease is live regardless of the test's wall clock
        # (lease expiry is deliberately checked against real time, not `now`).
        "expires_at": "2099-01-01T00:00:00Z",
    }
    store, _ = _store(
        {
            "queue.yaml": _queue_yaml(nb, remaining_runs="3"),
            "locks/scheduler_lease.json": json.dumps(lease),
        }
    )
    fake = FakeKaggleRunner()
    report = run_tick(store, KaggleClient(runner=fake), _config(), now=NOW, log=_silent)
    assert report.launched == []
    assert fake.pushes == []
    assert report.errors
    assert store.load().queue.tasks[0].remaining_runs == 3


def test_paused_scheduler_does_not_launch(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store(
        {
            "queue.yaml": _queue_yaml(nb, remaining_runs="3"),
            "scheduler_state.json": json.dumps({"scheduler_paused": True}),
        }
    )
    fake = FakeKaggleRunner()
    report = run_tick(store, KaggleClient(runner=fake), _config(), now=NOW, log=_silent)
    assert fake.pushes == []
    assert report.launched == []


def test_stop_launching_drains_without_killing(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store(
        {
            "queue.yaml": _queue_yaml(nb, remaining_runs="3"),
            "scheduler_state.json": json.dumps({"stop_launching": True}),
        }
    )
    fake = FakeKaggleRunner()
    report = run_tick(store, KaggleClient(runner=fake), _config(), now=NOW, log=_silent)
    assert fake.pushes == []
    assert report.launched == []


def test_runtime_config_injected_into_notebook_and_archived(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, backend = _store({"queue.yaml": _queue_yaml(nb, remaining_runs="null")})
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)
    # The config rides inside the pushed notebook (kaggle push uploads only the .ipynb).
    notebook = json.loads((nb / "runner.ipynb").read_text())
    injected = "".join(notebook["cells"][0]["source"])
    assert "scheduler-runtime-config" in notebook["cells"][0]["metadata"]["tags"]
    assert '"mode": "generation"' in injected and "runtime_config.json" in injected
    # ...and is archived to the state repo for auditing.
    archived = backend.read_text("runtime_configs/latest/runtime_config.json")
    assert archived is not None and json.loads(archived)["task_id"] == "gen"
