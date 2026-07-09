"""State backend + store: load/save, optimistic concurrency, lease, malformed files."""

from __future__ import annotations

import json

import pytest
import yaml  # type: ignore[import-untyped]

from ac_zero.scheduler.backend import MemoryStateBackend, StateConflict
from ac_zero.scheduler.store import LeaseError, StateError, StateStore

QUEUE = """
version: 1
limits:
  max_total_active: 3
tasks:
  - id: gen
    mode: generation
    notebook_slug: u/n
    notebook_dir: d
    remaining_runs: null
"""


def _store(files: dict[str, str] | None = None) -> tuple[StateStore, MemoryStateBackend]:
    backend = MemoryStateBackend(files if files is not None else {"queue.yaml": QUEUE})
    return StateStore(backend), backend


def test_load_parses_queue_and_defaults_state() -> None:
    store, _ = _store()
    snap = store.load()
    assert snap.queue.tasks[0].id == "gen"
    assert snap.queue.tasks[0].remaining_runs is None
    assert snap.queue.limits.max_total_active == 3
    assert snap.state.scheduler_paused is False


def test_missing_queue_raises_clear_error() -> None:
    store, _ = _store(files={})
    with pytest.raises(StateError, match=r"queue\.yaml not found"):
        store.load()


def test_malformed_queue_yaml_fails_clearly() -> None:
    store, _ = _store(files={"queue.yaml": "version: 1\ntasks: : ["})
    with pytest.raises(StateError, match=r"malformed queue\.yaml"):
        store.load()


def test_queue_without_tasks_key_fails_clearly() -> None:
    store, _ = _store(files={"queue.yaml": "version: 1\nlimits: {}"})
    with pytest.raises(StateError, match="expected a mapping with a 'tasks' list"):
        store.load()


def test_save_roundtrips_queue_and_state() -> None:
    store, _ = _store()
    snap = store.load()
    snap.queue.tasks[0].remaining_runs = 4
    snap.state.scheduler_paused = True
    store.save(snap, message="test")
    reloaded = store.load()
    assert reloaded.queue.tasks[0].remaining_runs == 4
    assert reloaded.state.scheduler_paused is True


def test_save_with_stale_base_sha_conflicts() -> None:
    store, backend = _store()
    snap = store.load()
    # Someone else commits after we read -> our base_sha is now stale.
    backend.commit({"queue.yaml": QUEUE}, message="other", parent_sha=None)
    with pytest.raises(StateConflict):
        store.save(snap, message="ours")


def test_extra_files_are_committed() -> None:
    store, backend = _store()
    snap = store.load()
    store.save(snap, message="test", extra_files={"runs/latest.json": "{}\n"})
    assert backend.read_text("runs/latest.json") == "{}\n"


def test_lease_acquire_and_reject_foreign_live_lease() -> None:
    store, _ = _store()
    store.acquire_lease(owner="owner-1", github_run_id="1", ttl_minutes=30)
    assert store.owns_lease("owner-1")
    with pytest.raises(LeaseError, match="held by 'owner-1'"):
        store.acquire_lease(owner="owner-2", github_run_id="2", ttl_minutes=30)


def test_expired_lease_is_stealable() -> None:
    store, backend = _store()
    expired = {
        "owner": "owner-1",
        "github_run_id": "1",
        "acquired_at": "2020-01-01T00:00:00Z",
        "expires_at": "2020-01-01T00:00:00Z",
    }
    backend.commit(
        {"locks/scheduler_lease.json": json.dumps(expired)}, message="seed", parent_sha=None
    )
    lease = store.acquire_lease(owner="owner-2", github_run_id="2", ttl_minutes=30)
    assert lease.owner == "owner-2"
    assert store.owns_lease("owner-2")


def test_release_lease_makes_it_no_longer_owned() -> None:
    store, _ = _store()
    store.acquire_lease(owner="owner-1", github_run_id="1", ttl_minutes=30)
    store.release_lease("owner-1")
    assert store.owns_lease("owner-1") is False


def test_saved_state_is_valid_json_and_yaml() -> None:
    store, backend = _store()
    store.save(store.load(), message="test")
    yaml.safe_load(backend.read_text("queue.yaml"))
    json.loads(backend.read_text("scheduler_state.json"))
