"""Runtime-config generation and kernel-metadata patching (no secrets leak)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ac_zero.scheduler.models import Task
from ac_zero.scheduler.runtime import (
    build_runtime_config,
    patch_kernel_metadata,
    write_runtime_config,
)

REPO = "user/kaggle-run-scheduler-state"


def _task(task_id: str, mode: str, accelerator: str = "cpu", **config: object) -> Task:
    return Task(
        id=task_id,
        mode=mode,
        accelerator=accelerator,
        notebook_slug="user/runner",
        notebook_dir="d",
        config=dict(config),
    )


def test_generation_runtime_config_is_valid_and_secret_free() -> None:
    task = _task("generation-main", "generation", rank=2, batch_size=16)
    cfg = build_runtime_config(task, run_id="r1", state_repo_id=REPO, state_repo_type="dataset")
    assert cfg["task_id"] == "generation-main"
    assert cfg["mode"] == "generation"
    assert cfg["config"]["rank"] == 2
    assert cfg["hf_state_repo_id"] == REPO

    # No secret material: no token-shaped values, no credential-named keys.
    def _values(obj: object) -> list[str]:
        if isinstance(obj, dict):
            return [v for val in obj.values() for v in _values(val)]
        return [obj] if isinstance(obj, str) else []

    assert not any(v.startswith("hf_") for v in _values(cfg))
    assert "token" not in cfg and "kaggle_key" not in cfg


def test_annotation_runtime_config_carries_moveset() -> None:
    task = _task("annotation-main", "annotation", moveset="strict-ac")
    cfg = build_runtime_config(task, run_id="r2", state_repo_id=REPO, state_repo_type="dataset")
    assert cfg["mode"] == "annotation"
    assert cfg["config"]["moveset"] == "strict-ac"


def test_write_runtime_config_writes_file(tmp_path: Path) -> None:
    task = _task("t", "generation")
    cfg = build_runtime_config(task, run_id="r", state_repo_id=REPO, state_repo_type="dataset")
    path = write_runtime_config(tmp_path, cfg)
    assert path.name == "runtime_config.json"
    assert json.loads(path.read_text())["run_id"] == "r"


def _seed_metadata(tmp_path: Path) -> Path:
    meta = {
        "id": "user/runner",
        "title": "Runner",
        "code_file": "scheduler_runner.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
    }
    path = tmp_path / "kernel-metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_patch_metadata_sets_gpu_and_secrets_source(tmp_path: Path) -> None:
    _seed_metadata(tmp_path)
    task = _task("t", "training", accelerator="gpu")
    task.notebook_slug = "user/runner"
    patch_kernel_metadata(tmp_path, task, secrets_dataset="user/runtime-secrets")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] == "true"
    assert meta["is_private"] == "true"
    assert "user/runtime-secrets" in meta["dataset_sources"]


def test_patch_metadata_cpu_disables_gpu_and_is_idempotent(tmp_path: Path) -> None:
    _seed_metadata(tmp_path)
    task = _task("t", "generation", accelerator="cpu")
    patch_kernel_metadata(tmp_path, task, secrets_dataset="user/runtime-secrets")
    patch_kernel_metadata(tmp_path, task, secrets_dataset="user/runtime-secrets")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] == "false"
    assert meta["dataset_sources"].count("user/runtime-secrets") == 1


def test_patch_metadata_missing_file_raises(tmp_path: Path) -> None:
    task = _task("t", "generation")
    with pytest.raises(FileNotFoundError):
        patch_kernel_metadata(tmp_path, task, secrets_dataset="user/runtime-secrets")
