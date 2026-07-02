"""Upload/download the training dataset to a Hugging Face storage bucket.

The grown dataset outgrows GitHub's 100 MB per-file limit, so the training set
lives in a Hugging Face bucket instead of git. These helpers wrap the
``huggingface_hub`` bucket API so the CLI (``aczero dataset upload|download``)
and the Kaggle generation notebook can pull the current dataset before use and
push an updated one back after a grow.

Authentication uses the ambient Hugging Face token: set the ``HF_TOKEN``
environment variable or run ``hf auth login`` once. ``huggingface_hub`` is an
optional dependency -- install it with ``pip install ac-zero[hub]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# Bucket holding the AlphaAC training datasets (namespace/bucket-name).
DEFAULT_BUCKET = "HkHk2Prod/alphaac-data"

_INSTALL_HINT = (
    "Hugging Face bucket access needs the optional `huggingface_hub` dependency; "
    "install it with `pip install ac-zero[hub]` (or `pip install 'huggingface_hub>=1.21'`)."
)


def _hub() -> ModuleType:
    """Import ``huggingface_hub`` lazily with a friendly error if it is absent."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return huggingface_hub


def remote_exists(remote_name: str, *, bucket: str = DEFAULT_BUCKET) -> bool:
    """Return whether a file named ``remote_name`` is present in the bucket."""
    hub = _hub()
    return any(
        item.path == remote_name
        for item in hub.list_bucket_tree(bucket, recursive=True)
        if getattr(item, "type", "file") == "file"
    )


def upload_dataset(
    local_path: str | Path,
    *,
    remote_name: str | None = None,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    """Upload a local dataset file to the bucket and return its ``hf://`` URI.

    ``remote_name`` defaults to the local file's basename, so the bucket mirrors
    the on-disk name (e.g. ``train_rank2.json``).
    """
    hub = _hub()
    local = Path(local_path)
    if not local.is_file():
        raise FileNotFoundError(f"dataset not found: {local}")
    name = remote_name or local.name
    hub.batch_bucket_files(bucket, add=[(str(local), name)])
    return f"hf://buckets/{bucket}/{name}"


def download_dataset(
    local_path: str | Path,
    *,
    remote_name: str | None = None,
    bucket: str = DEFAULT_BUCKET,
    missing_ok: bool = False,
) -> Path | None:
    """Download a dataset file from the bucket to ``local_path``.

    ``remote_name`` defaults to the local file's basename. With
    ``missing_ok=True`` this returns ``None`` instead of raising when the file
    is absent from the bucket -- used by the notebook's "resume the run if a
    dataset already exists" first pass.
    """
    hub = _hub()
    local = Path(local_path)
    name = remote_name or local.name
    if missing_ok and not remote_exists(name, bucket=bucket):
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    hub.download_bucket_files(bucket, files=[(name, str(local))])
    return local
