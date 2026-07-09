"""Notebook-side helpers for a scheduled Kaggle run.

Imported from inside the unified Kaggle notebook (which ``pip install``s
``ac_zero`` from GitHub). Keeps the notebook cells thin and the logic testable:

* :func:`load_runtime_config` reads the secret-free ``runtime_config.json`` the
  scheduler dropped next to the notebook.
* :func:`login_from_secret_dataset` reads the HF token from the private Kaggle
  dataset and logs in -- the token is never printed or written to working.
* :class:`RunReporter` publishes started/heartbeat/finished records to the HF
  state repo and polls the queue for a manual stop request.

The token is used both for private/gated HF model access and for writing these
run records, so it must have read+write on the state repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ac_zero.scheduler.models import utc_now

DEFAULT_TOKEN_PATH = "/kaggle/input/runtime-secrets/hf_token.txt"


def load_runtime_config(path: str = "runtime_config.json") -> dict[str, Any]:
    """Read the scheduler's per-run config. Raises if it is missing/malformed."""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(
            f"runtime config not found at {path}; was the notebook launched by the scheduler?"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "run_id" not in data:
        raise RuntimeError(f"malformed runtime config at {path}: expected a mapping with a run_id.")
    return data


def login_from_secret_dataset(token_path: str = DEFAULT_TOKEN_PATH) -> str:
    """Read the HF token from the private Kaggle dataset and log in.

    Returns the token so the caller can set env vars; it is never logged. The
    token is validated shape-only (``hf_`` prefix) to fail fast on a bad mount.
    """
    p = Path(token_path)
    if not p.exists():
        raise RuntimeError(f"missing Hugging Face token dataset input at {token_path}.")
    token = p.read_text().strip()
    if not token.startswith("hf_"):
        raise RuntimeError("HF token file exists but does not look like a Hugging Face token.")
    from huggingface_hub import login

    login(token=token)
    return token


class RunReporter:
    """Publish run status/heartbeats to the HF state store and poll for stops.

    Works against either the shared HF **bucket** or a dataset repo -- it goes
    through a :class:`StateBackend`, so the notebook writes to the same place the
    scheduler reads from, whatever ``repo_type`` the run config specifies.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        run_id: str,
        task_id: str,
        token: str,
        repo_type: str = "dataset",
    ) -> None:
        from ac_zero.scheduler.backend import make_state_backend

        self.run_id = run_id
        self.task_id = task_id
        self._backend = make_state_backend(repo_id, token=token, repo_type=repo_type)

    def _publish(self, status: str, *, error: str | None, extra: dict[str, Any] | None) -> None:
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": status,
            "heartbeat_at": utc_now(),
        }
        if error:
            record["error"] = error
        if extra:
            record["extra"] = extra
        payload = json.dumps(record, indent=2) + "\n"
        self._backend.commit(
            {f"runs/{self.run_id}.json": payload, "runs/latest.json": payload},
            message=f"{status} {self.run_id}",
            parent_sha=None,
        )

    def started(self, extra: dict[str, Any] | None = None) -> None:
        self._publish("running", error=None, extra=extra)

    def heartbeat(self, status: str = "running", extra: dict[str, Any] | None = None) -> None:
        self._publish(status, error=None, extra=extra)

    def finished(
        self,
        *,
        status: str = "finished",
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._publish(status, error=error, extra=extra)

    def should_stop(self) -> bool:
        """Return whether the operator asked this run to stop.

        True when the task was deactivated or flagged
        ``stop_after_current_iteration`` in ``queue.yaml``. Network errors are
        swallowed (returns ``False``) so a transient blip never kills a run.
        """
        import yaml  # type: ignore[import-untyped]

        try:
            raw = self._backend.read_text("queue.yaml")
        except Exception:
            return False
        if raw is None:
            return False
        try:
            doc = yaml.safe_load(raw) or {}
        except yaml.YAMLError:
            return False
        for task in doc.get("tasks", []):
            if task.get("id") == self.task_id:
                inactive = not task.get("active", True)
                return inactive or bool(task.get("stop_after_current_iteration"))
        return False
