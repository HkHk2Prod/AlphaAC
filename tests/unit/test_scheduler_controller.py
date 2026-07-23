"""End-to-end scheduler tick: launch counting, reconciliation, lease, dry-run.

Kaggle CLI and the HF Hub are both faked -- no network or credentials needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from ac_zero.scheduler.backend import MemoryStateBackend
from ac_zero.scheduler.benchmarks import BENCHMARK_QUEUE_PATH
from ac_zero.scheduler.controller import SchedulerConfig, run_tick
from ac_zero.scheduler.kaggle import KaggleClient
from ac_zero.scheduler.store import (
    LEASE_PATH,
    QUEUE_PATH,
    RUNTIME_CONFIG_LATEST,
    STATE_PATH,
    StateStore,
)

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


def _queue_yaml(
    nb_dir: Path,
    *,
    remaining_runs: str = "null",
    extra: str = "",
    start_fresh_all: str = "false",
    start_fresh: str = "false",
) -> str:
    return f"""
version: 1
start_fresh_all: {start_fresh_all}
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
    start_fresh: {start_fresh}
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
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, remaining_runs="3")})
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
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, remaining_runs="1")})
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)
    snap = store.load()
    assert snap.queue.tasks[0].remaining_runs == 0
    assert snap.queue.tasks[0].active is False


def test_infinite_remaining_runs_never_decrements(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, remaining_runs="null")})
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)
    snap = store.load()
    assert snap.queue.tasks[0].remaining_runs is None
    assert snap.queue.tasks[0].active is True


def test_launch_failure_does_not_decrement(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, remaining_runs="3")})
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
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, remaining_runs="2")})
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
    store, backend = _store({QUEUE_PATH: _queue_yaml(nb, remaining_runs="3")})
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
            QUEUE_PATH: _queue_yaml(nb, remaining_runs="3"),
            LEASE_PATH: json.dumps(lease),
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
            QUEUE_PATH: _queue_yaml(nb, remaining_runs="3"),
            STATE_PATH: json.dumps({"scheduler_paused": True}),
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
            QUEUE_PATH: _queue_yaml(nb, remaining_runs="3"),
            STATE_PATH: json.dumps({"stop_launching": True}),
        }
    )
    fake = FakeKaggleRunner()
    report = run_tick(store, KaggleClient(runner=fake), _config(), now=NOW, log=_silent)
    assert fake.pushes == []
    assert report.launched == []


_SECOND_TASK = """  - id: gen2
    mode: generation
    accelerator: cpu
    notebook_slug: u/runner2
    notebook_dir: {nb_dir}
    config:
      rank: 2
"""


def test_start_fresh_all_marks_every_task_and_clears_itself(tmp_path: Path) -> None:
    """One global edit restarts the whole queue, and only once."""
    nb = _notebook_dir(tmp_path)
    queue = _queue_yaml(nb, start_fresh_all="true", extra=_SECOND_TASK.format(nb_dir=nb))
    store, _ = _store({QUEUE_PATH: queue})

    # No launches this tick, so the expansion is all that is observed.
    run_tick(
        store,
        KaggleClient(runner=FakeKaggleRunner()),
        _config(max_launches_override=0),
        now=NOW,
        log=_silent,
    )

    snap = store.load()
    assert snap.queue.start_fresh_all is False
    assert [t.start_fresh for t in snap.queue.tasks] == [True, True]


def test_start_fresh_all_applies_to_a_task_launched_in_the_same_tick(tmp_path: Path) -> None:
    """The expansion runs before selection, so the launch it triggers is a fresh one."""
    nb = _notebook_dir(tmp_path)
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, start_fresh_all="true")})

    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)

    injected = "".join(json.loads((nb / "runner.ipynb").read_text())["cells"][0]["source"])
    assert '\\"start_fresh\\": true' in injected


def test_launch_consumes_start_fresh(tmp_path: Path) -> None:
    """The flag rides one launch and is cleared, so the next run resumes normally."""
    nb = _notebook_dir(tmp_path)
    store, backend = _store({QUEUE_PATH: _queue_yaml(nb, start_fresh="true")})

    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)

    archived = backend.read_text(RUNTIME_CONFIG_LATEST)
    assert archived is not None and json.loads(archived)["start_fresh"] is True
    assert store.load().queue.tasks[0].start_fresh is False


def test_failed_launch_keeps_start_fresh(tmp_path: Path) -> None:
    """Nothing archived the lineage, so the restart is still owed and must survive."""
    nb = _notebook_dir(tmp_path)
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, start_fresh="true")})

    report = run_tick(
        store, KaggleClient(runner=FakeKaggleRunner(push_rc=1)), _config(), now=NOW, log=_silent
    )

    assert report.launched == []
    assert store.load().queue.tasks[0].start_fresh is True


def test_dry_run_does_not_consume_start_fresh_all(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, start_fresh_all="true")})

    run_tick(
        store, KaggleClient(runner=FakeKaggleRunner()), _config(dry_run=True), now=NOW, log=_silent
    )

    assert store.load().queue.start_fresh_all is True


def test_runtime_config_injected_into_notebook_and_archived(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, backend = _store({QUEUE_PATH: _queue_yaml(nb, remaining_runs="null")})
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)
    # The config rides inside the pushed notebook (kaggle push uploads only the .ipynb).
    notebook = json.loads((nb / "runner.ipynb").read_text())
    first = notebook["cells"][0]
    assert "scheduler-runtime-config" in first["metadata"]["tags"]
    injected = "".join(first["source"])
    assert "runtime_config.json" in injected and "generation" in injected
    # ...and is archived to the state repo for auditing.
    archived = backend.read_text(RUNTIME_CONFIG_LATEST)
    assert archived is not None and json.loads(archived)["task_id"] == "gen"


def _benchmark_task(nb_dir: Path) -> str:
    return f"""  - id: bench
    mode: benchmark
    accelerator: cpu
    priority: 99
    notebook_slug: u/runner
    notebook_dir: {nb_dir}
    config:
      rank: 2
"""


def _pending_doc(*entries: tuple[str, str, float]) -> str:
    return json.dumps(
        {
            "pending": [
                {"checkpoint_name": name, "run_id": run, "metric": metric, "enqueued_at": NOW}
                for name, run, metric in entries
            ],
            "dispatched": [],
        }
    )


def test_a_benchmark_task_does_not_launch_with_an_empty_evaluation_queue(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store({QUEUE_PATH: _queue_yaml(nb, extra=_benchmark_task(nb))})
    report = run_tick(
        store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent
    )

    launched = {run_id.rsplit("-", 4)[0] for run_id in report.launched}
    assert "bench" not in launched
    reasons = {d.task_id: d.reason for d in report.decisions}
    assert reasons["bench"] == "no checkpoints pending evaluation"


def test_a_benchmark_launch_takes_the_highest_metric_checkpoint(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, _ = _store(
        {
            QUEUE_PATH: _queue_yaml(nb, extra=_benchmark_task(nb)),
            BENCHMARK_QUEUE_PATH: _pending_doc(("model-a", "r1", 0.4), ("model-b", "r2", 0.8)),
        }
    )
    report = run_tick(
        store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent
    )

    assert any(run_id.startswith("bench-") for run_id in report.launched)
    task = next(t for t in store.load().queue.tasks if t.id == "bench")
    assert task.config["checkpoint_name"] == "model-b"
    assert task.config["checkpoint_run_id"] == "r2"


def test_a_dispatched_checkpoint_leaves_the_pending_queue(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, backend = _store(
        {
            QUEUE_PATH: _queue_yaml(nb, extra=_benchmark_task(nb)),
            BENCHMARK_QUEUE_PATH: _pending_doc(("model-a", "r1", 0.9)),
        }
    )
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)

    doc = json.loads(backend.read_text(BENCHMARK_QUEUE_PATH) or "{}")
    assert doc["pending"] == []
    assert [e["checkpoint_name"] for e in doc["dispatched"]] == ["model-a"]


def test_the_evaluation_queue_is_written_even_when_nothing_launches(tmp_path: Path) -> None:
    nb = _notebook_dir(tmp_path)
    store, backend = _store({QUEUE_PATH: _queue_yaml(nb, extra=_benchmark_task(nb))})
    run_tick(store, KaggleClient(runner=FakeKaggleRunner()), _config(), now=NOW, log=_silent)

    assert json.loads(backend.read_text(BENCHMARK_QUEUE_PATH) or "{}") == {
        "pending": [],
        "dispatched": [],
        "ladder": {},
    }
