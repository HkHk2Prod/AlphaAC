"""Build the per-run ``runtime_config.json`` and patch ``kernel-metadata.json``.

Immediately before ``kaggle kernels push``, the controller drops a
``runtime_config.json`` into the local notebook directory and rewrites the
kernel metadata for this task (GPU flag, privacy, secrets dataset input). The
notebook reads the config at startup and branches on ``mode``.

Secrets never appear here: the HF token travels only through the private Kaggle
``runtime-secrets`` dataset, never through this config or the metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ac_zero.scheduler.models import Task

RUNTIME_CONFIG_NAME = "runtime_config.json"
KERNEL_METADATA_NAME = "kernel-metadata.json"


def build_runtime_config(
    task: Task,
    *,
    run_id: str,
    state_repo_id: str,
    state_repo_type: str,
) -> dict[str, Any]:
    """Assemble the secret-free config the notebook consumes at startup."""
    return {
        "task_id": task.id,
        "run_id": run_id,
        "mode": task.mode,
        "accelerator": task.accelerator,
        "max_runtime_minutes": task.max_runtime_minutes,
        "hf_state_repo_id": state_repo_id,
        "hf_state_repo_type": state_repo_type,
        "stop_after_current_iteration": task.stop_after_current_iteration,
        "config": task.config,
    }


def write_runtime_config(notebook_dir: str | Path, config: dict[str, Any]) -> Path:
    """Write ``runtime_config.json`` into the notebook dir; return its path."""
    path = Path(notebook_dir) / RUNTIME_CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def patch_kernel_metadata(
    notebook_dir: str | Path,
    task: Task,
    *,
    secrets_dataset: str,
) -> Path:
    """Patch the existing ``kernel-metadata.json`` for this launch.

    Sets the kernel id to the task's slug, toggles the GPU per the task's
    accelerator, forces the kernel private, and ensures the private
    runtime-secrets dataset is listed as an input source. Preserves any other
    fields (title, code_file, extra sources) already in the file.
    """
    path = Path(notebook_dir) / KERNEL_METADATA_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; the notebook dir must ship a kernel-metadata.json template."
        )
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    meta["id"] = task.notebook_slug
    meta["kernel_type"] = meta.get("kernel_type", "notebook")
    meta["language"] = meta.get("language", "python")
    meta["is_private"] = "true"
    meta["enable_internet"] = "true"
    meta["enable_gpu"] = "true" if task.accelerator == "gpu" else "false"

    sources = list(meta.get("dataset_sources") or [])
    if secrets_dataset not in sources:
        sources.append(secrets_dataset)
    meta["dataset_sources"] = sources
    meta.setdefault("competition_sources", [])
    meta.setdefault("kernel_sources", [])
    meta.setdefault("model_sources", [])

    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path
