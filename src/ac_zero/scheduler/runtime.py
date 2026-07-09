"""Build the per-run runtime config and prepare the notebook for ``kaggle push``.

Immediately before ``kaggle kernels push``, the controller injects the per-run
config **into the notebook itself** and rewrites the kernel metadata for this
task (GPU flag, privacy, secrets dataset input). The notebook reads the config
at startup and branches on ``mode``.

``kaggle kernels push`` uploads only the notebook and ``kernel-metadata.json`` --
not other files in the directory -- so the config cannot be shipped as a
separate ``runtime_config.json`` file; it is embedded as the notebook's first
cell, which recreates ``runtime_config.json`` at runtime on Kaggle.

Secrets never appear here: the HF token travels only through the private Kaggle
``runtime-secrets`` dataset, never through this config or the metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ac_zero.scheduler.models import Task

KERNEL_METADATA_NAME = "kernel-metadata.json"
CONFIG_CELL_TAG = "scheduler-runtime-config"


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


def _config_cell_source(config: dict[str, Any]) -> list[str]:
    """Python source (as nbformat line list) that recreates the config file.

    The config is embedded as a JSON *string* (double-encoded) and written
    verbatim -- embedding the dict as a Python literal would break on JSON
    ``false``/``true``/``null``, which are not Python identifiers.
    """
    payload = json.dumps(config)
    return [
        "# Injected by the Kaggle scheduler: recreate runtime_config.json for this run.\n",
        "# `kaggle kernels push` uploads only the notebook, so the config rides inside it.\n",
        f"_CONFIG_JSON = {json.dumps(payload)}\n",
        'with open("runtime_config.json", "w") as _f:\n',
        "    _f.write(_CONFIG_JSON)\n",
    ]


def inject_runtime_config(notebook_dir: str | Path, code_file: str, config: dict[str, Any]) -> Path:
    """Embed ``config`` as the notebook's first cell (idempotent).

    Removes any previously injected config cell first, then inserts a fresh one
    that writes ``runtime_config.json`` at runtime. Returns the notebook path.
    """
    path = Path(notebook_dir) / code_file
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; the notebook dir must ship {code_file}.")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in notebook.get("cells", [])
        if CONFIG_CELL_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    config_cell = {
        "cell_type": "code",
        "metadata": {"tags": [CONFIG_CELL_TAG]},
        "execution_count": None,
        "outputs": [],
        "source": _config_cell_source(config),
    }
    cells.insert(0, config_cell)
    notebook["cells"] = cells
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return path


def code_file_of(notebook_dir: str | Path) -> str:
    """Read the notebook's filename from its ``kernel-metadata.json``."""
    meta_path = Path(notebook_dir) / KERNEL_METADATA_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found; cannot determine the notebook file.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    code_file = meta.get("code_file")
    if not code_file:
        raise ValueError(f"{meta_path} has no 'code_file' field.")
    return str(code_file)


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
