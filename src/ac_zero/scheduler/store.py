"""Read/write the scheduler documents on a :class:`StateBackend`.

Knows the repo file layout and (de)serialises ``queue.yaml`` /
``scheduler_state.json`` / the lease. All writes go through the backend's
optimistic-concurrency ``commit`` so a stale scheduler tick is rejected rather
than silently clobbering a newer state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC

import yaml  # type: ignore[import-untyped]

from ac_zero.scheduler.backend import StateBackend, StateConflict
from ac_zero.scheduler.models import Lease, Queue, SchedulerState, utc_now

# All scheduler state lives under one bucket folder, kept apart from the datasets
# and model checkpoints that share the bucket.
QUEUE_PREFIX = "queue"
QUEUE_PATH = f"{QUEUE_PREFIX}/queue.yaml"
STATE_PATH = f"{QUEUE_PREFIX}/scheduler_state.json"
LEASE_PATH = f"{QUEUE_PREFIX}/locks/scheduler_lease.json"
RUNTIME_CONFIG_LATEST = f"{QUEUE_PREFIX}/runtime_configs/latest/runtime_config.json"
RUNS_DIR = f"{QUEUE_PREFIX}/runs"
# The most recent run record, overwritten each heartbeat for a quick "what's live" read.
LATEST_RUN_PATH = f"{RUNS_DIR}/latest.json"


def run_path(run_id: str) -> str:
    """The bucket path of one run's status/heartbeat record: ``queue/runs/<run_id>.json``."""
    return f"{RUNS_DIR}/{run_id}.json"


class StateError(RuntimeError):
    """Raised on malformed or inaccessible scheduler state."""


@dataclass(slots=True)
class Snapshot:
    """A consistent read of the scheduler state plus its base revision."""

    queue: Queue
    state: SchedulerState
    base_sha: str | None


class StateStore:
    """Typed accessor over a :class:`StateBackend` for the scheduler files."""

    def __init__(self, backend: StateBackend) -> None:
        self.backend = backend

    def load(self) -> Snapshot:
        """Read queue + state at a single base revision.

        A missing ``scheduler_state.json`` starts empty (first ever run). A
        missing or malformed ``queue.yaml`` is a hard error -- the scheduler has
        nothing to do and should say so clearly.
        """
        base_sha = self.backend.head_sha()
        raw_queue = self.backend.read_text(QUEUE_PATH)
        if raw_queue is None:
            raise StateError(
                f"{QUEUE_PATH} not found in the HF state repo; create it before scheduling."
            )
        try:
            queue_doc = yaml.safe_load(raw_queue) or {}
        except yaml.YAMLError as exc:
            raise StateError(f"malformed {QUEUE_PATH}: {exc}") from exc
        if not isinstance(queue_doc, dict) or "tasks" not in queue_doc:
            raise StateError(f"malformed {QUEUE_PATH}: expected a mapping with a 'tasks' list.")
        queue = Queue.from_dict(queue_doc)

        raw_state = self.backend.read_text(STATE_PATH)
        if raw_state is None:
            state = SchedulerState()
        else:
            try:
                state = SchedulerState.from_dict(json.loads(raw_state))
            except json.JSONDecodeError as exc:
                raise StateError(f"malformed {STATE_PATH}: {exc}") from exc
        return Snapshot(queue=queue, state=state, base_sha=base_sha)

    def save(
        self, snapshot: Snapshot, *, message: str, extra_files: dict[str, str] | None = None
    ) -> str:
        """Commit queue + state (+ any ``extra_files``), guarding on the base SHA.

        Raises :class:`StateConflict` (from the backend) if the remote moved
        since the snapshot was read.
        """
        files = {
            QUEUE_PATH: yaml.safe_dump(snapshot.queue.to_dict(), sort_keys=False),
            STATE_PATH: json.dumps(snapshot.state.to_dict(), indent=2) + "\n",
        }
        if extra_files:
            files.update(extra_files)
        return self.backend.commit(files, message=message, parent_sha=snapshot.base_sha)

    # --- lease ---------------------------------------------------------------

    def read_lease(self) -> Lease | None:
        raw = self.backend.read_text(LEASE_PATH)
        if raw is None:
            return None
        try:
            return Lease.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return None

    def acquire_lease(self, *, owner: str, github_run_id: str, ttl_minutes: int) -> Lease:
        """Acquire the scheduler lease unless a live one is held by someone else.

        GitHub Actions ``concurrency`` is the first line of defence; this lease
        is the second, guarding against overlap the runner cannot see. An
        expired lease is stealable; a live foreign lease raises.
        """
        existing = self.read_lease()
        if existing is not None and existing.owner != owner and not existing.is_expired():
            raise LeaseError(
                f"scheduler lease held by {existing.owner!r} until {existing.expires_at}"
            )
        now = utc_now()
        from datetime import datetime, timedelta

        expires = (datetime.now(UTC) + timedelta(minutes=ttl_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        lease = Lease(owner=owner, github_run_id=github_run_id, acquired_at=now, expires_at=expires)
        self._write_lease(lease)
        return lease

    def owns_lease(self, owner: str) -> bool:
        """Re-read the lease and confirm ``owner`` still holds a live one."""
        current = self.read_lease()
        return current is not None and current.owner == owner and not current.is_expired()

    def release_lease(self, owner: str) -> None:
        """Expire our lease so the next tick need not wait for the TTL."""
        current = self.read_lease()
        if current is None or current.owner != owner:
            return
        expired = Lease(
            owner=owner,
            github_run_id=current.github_run_id,
            acquired_at=current.acquired_at,
            expires_at=utc_now(),
        )
        self._write_lease(expired)

    def _write_lease(self, lease: Lease) -> None:
        from dataclasses import asdict

        # The lease commit does not guard on a parent SHA: GitHub concurrency
        # already serialises ticks, and the read-then-check in acquire/owns is
        # what defends against a foreign holder.
        self.backend.commit(
            {LEASE_PATH: json.dumps(asdict(lease), indent=2) + "\n"},
            message=f"scheduler lease {lease.owner}",
            parent_sha=None,
        )


class LeaseError(RuntimeError):
    """Raised when the scheduler lease is held by another live owner."""


__all__ = [
    "LATEST_RUN_PATH",
    "LEASE_PATH",
    "QUEUE_PATH",
    "QUEUE_PREFIX",
    "RUNS_DIR",
    "RUNTIME_CONFIG_LATEST",
    "STATE_PATH",
    "LeaseError",
    "Snapshot",
    "StateConflict",
    "StateError",
    "StateStore",
    "run_path",
]
