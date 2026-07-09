#!/usr/bin/env python3
"""Controller entrypoint: run one scheduler tick from GitHub Actions (or locally).

Reads configuration from the environment (set by the workflow) and CLI flags,
builds an :class:`HubStateBackend` over the private HF state dataset repo, and
drives :func:`ac_zero.scheduler.controller.run_tick`.

Environment:
  HF_TOKEN / HF_STATE_TOKEN   token with read+write on the state repo
  HF_STATE_REPO_ID            e.g. ``your-user/kaggle-run-scheduler-state``
  HF_STATE_REPO_TYPE          default ``dataset``
  KAGGLE_USERNAME             used to derive the runtime-secrets dataset slug
  KAGGLE_SECRETS_DATASET      override the ``<user>/runtime-secrets`` slug
  GITHUB_RUN_ID               lease owner id
  TASK_ID / FORCE / DRY_RUN / MAX_LAUNCHES   optional dispatch inputs

Never prints tokens.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a bare script (`python scripts/kaggle_scheduler.py`) by making
# the src/ layout importable without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ac_zero.scheduler.backend import make_state_backend
from ac_zero.scheduler.controller import SchedulerConfig, run_tick
from ac_zero.scheduler.kaggle import KaggleClient
from ac_zero.scheduler.store import StateError, StateStore


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Kaggle scheduler tick.")
    parser.add_argument("--dry-run", action="store_true", help="show decisions without launching")
    parser.add_argument("--task-id", default=None, help="prioritize (with --force, force) a task")
    parser.add_argument("--force", action="store_true", help="launch even if the task is running")
    parser.add_argument("--max-launches", type=int, default=None, help="cap launches this tick")
    parser.add_argument("--lease-ttl-minutes", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    token = os.environ.get("HF_STATE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_STATE_TOKEN or HF_TOKEN for state-repo access.", file=sys.stderr)
        return 2
    repo_id = os.environ.get("HF_STATE_REPO_ID", "").strip()
    if not repo_id:
        print("ERROR: set HF_STATE_REPO_ID to the private HF state repo/bucket.", file=sys.stderr)
        return 2
    # Default to a bucket -- the scheduler state lives in the same HF bucket as
    # the training dataset (HkHk2Prod/alphaac-data).
    repo_type = os.environ.get("HF_STATE_REPO_TYPE", "bucket").strip() or "bucket"
    # The bucket wrappers authenticate via the ambient HF token.
    os.environ["HF_TOKEN"] = token

    kaggle_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    secrets_dataset = os.environ.get("KAGGLE_SECRETS_DATASET", "").strip()
    if not secrets_dataset:
        if not kaggle_user:
            print("ERROR: set KAGGLE_USERNAME or KAGGLE_SECRETS_DATASET.", file=sys.stderr)
            return 2
        secrets_dataset = f"{kaggle_user}/runtime-secrets"

    github_run_id = os.environ.get("GITHUB_RUN_ID", "local")

    config = SchedulerConfig(
        state_repo_id=repo_id,
        state_repo_type=repo_type,
        secrets_dataset=secrets_dataset,
        owner=f"gha-{github_run_id}",
        github_run_id=github_run_id,
        lease_ttl_minutes=args.lease_ttl_minutes,
        force_task_id=args.task_id or (os.environ.get("TASK_ID") or None),
        force=args.force or _env_flag("FORCE"),
        dry_run=args.dry_run or _env_flag("DRY_RUN"),
        max_launches_override=(
            args.max_launches if args.max_launches is not None else _env_int("MAX_LAUNCHES")
        ),
    )

    store = StateStore(make_state_backend(repo_id, token=token, repo_type=repo_type))
    kaggle = KaggleClient()

    try:
        report = run_tick(store, kaggle, config)
    except StateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"tick done: launched={len(report.launched)} "
        f"decisions={len(report.decisions)} errors={len(report.errors)} dry_run={report.dry_run}"
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
