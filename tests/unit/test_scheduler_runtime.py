"""Runtime-config generation and kernel-metadata patching (no secrets leak)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ac_zero.scheduler.models import Task
from ac_zero.scheduler.runtime import (
    CONFIG_CELL_TAG,
    build_runtime_config,
    code_file_of,
    inject_runtime_config,
    patch_kernel_metadata,
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


def _seed_notebook(tmp_path: Path, code_file: str = "runner.ipynb") -> Path:
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# t"]}], "nbformat": 4}
    path = tmp_path / code_file
    path.write_text(json.dumps(nb), encoding="utf-8")
    (tmp_path / "kernel-metadata.json").write_text(
        json.dumps({"id": "user/runner", "code_file": code_file, "dataset_sources": []}),
        encoding="utf-8",
    )
    return path


def _exec_injected_cell(tmp_path: Path, run_dir: Path) -> dict:
    """Run the notebook's first (injected) cell in run_dir; return the config it writes."""
    nb = json.loads((tmp_path / "runner.ipynb").read_text())
    first = nb["cells"][0]
    assert CONFIG_CELL_TAG in first["metadata"]["tags"]
    cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        exec(compile("".join(first["source"]), "<injected>", "exec"), {})
    finally:
        os.chdir(cwd)
    return json.loads((run_dir / "runtime_config.json").read_text())


def test_inject_runtime_config_cell_writes_valid_config(tmp_path: Path) -> None:
    _seed_notebook(tmp_path)
    # stop_after_current_iteration -> JSON `false`, which must survive the round trip
    # (embedding as a Python literal would raise NameError: name 'false' is not defined).
    cfg = build_runtime_config(
        _task("t", "generation"), run_id="r", state_repo_id=REPO, state_repo_type="bucket"
    )
    inject_runtime_config(tmp_path, "runner.ipynb", cfg)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    written = _exec_injected_cell(tmp_path, run_dir)
    assert written == cfg
    assert written["stop_after_current_iteration"] is False


def test_inject_runtime_config_is_idempotent(tmp_path: Path) -> None:
    _seed_notebook(tmp_path)
    cfg = build_runtime_config(
        _task("t", "generation"), run_id="r", state_repo_id=REPO, state_repo_type="bucket"
    )
    inject_runtime_config(tmp_path, "runner.ipynb", cfg)
    inject_runtime_config(tmp_path, "runner.ipynb", {**cfg, "run_id": "r2"})
    nb = json.loads((tmp_path / "runner.ipynb").read_text())
    tagged = [c for c in nb["cells"] if CONFIG_CELL_TAG in c.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1  # replaced, not accumulated
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert _exec_injected_cell(tmp_path, run_dir)["run_id"] == "r2"


def test_code_file_of_reads_metadata(tmp_path: Path) -> None:
    _seed_notebook(tmp_path, code_file="scheduler_runner.ipynb")
    assert code_file_of(tmp_path) == "scheduler_runner.ipynb"


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
