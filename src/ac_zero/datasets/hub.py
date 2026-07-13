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
from typing import Any

# Bucket holding the AlphaAC training datasets (namespace/bucket-name).
DEFAULT_BUCKET = "HkHk2Prod/alphaac-data"

_INSTALL_HINT = (
    "Hugging Face bucket access needs the optional `huggingface_hub` dependency; "
    "install it with `pip install ac-zero[hub]` (or `pip install 'huggingface_hub>=1.21'`)."
)


def _hub() -> Any:
    """Import ``huggingface_hub`` lazily with a friendly error if it is absent.

    Typed as ``Any`` because ``huggingface_hub`` is an optional dependency that
    need not be installed for type-checking or for code paths that never touch
    the bucket.
    """
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    _disable_progress_bars(huggingface_hub)
    return huggingface_hub


def _disable_progress_bars(hub: Any) -> None:
    """Turn off the hub's per-file transfer bars.

    Kaggle's training log is not a terminal, so each bar redraw is appended as a
    fresh line -- hundreds of them per upload. Callers print one summary line
    instead. Set ``HF_HUB_DISABLE_PROGRESS_BARS=0`` to keep the bars.
    """
    disable = getattr(getattr(hub, "utils", None), "disable_progress_bars", None)
    if disable is not None:
        disable()


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
    local = Path(local_path)
    return download_file(remote_name or local.name, local, bucket=bucket, missing_ok=missing_ok)


def upload_files(pairs: list[tuple[str | Path, str]], *, bucket: str = DEFAULT_BUCKET) -> None:
    """Upload ``(local_path, remote_path)`` pairs to the bucket in one batch.

    ``remote_path`` may contain slashes, so a whole ``model_checkpoints/<name>/``
    tree uploads in a single call. Missing local files raise before any upload.
    """
    resolved: list[tuple[str, str]] = []
    for local, remote in pairs:
        path = Path(local)
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path}")
        resolved.append((str(path), remote))
    if resolved:
        _hub().batch_bucket_files(bucket, add=resolved)


def download_file(
    remote_name: str,
    local_path: str | Path,
    *,
    bucket: str = DEFAULT_BUCKET,
    missing_ok: bool = False,
) -> Path | None:
    """Download a single ``remote_name`` object from the bucket to ``local_path``.

    With ``missing_ok=True`` returns ``None`` instead of raising when the object
    is absent from the bucket. Otherwise an absent object raises: the hub skips
    missing files with a warning by default, which would leave callers holding a
    path to a file that was never written.
    """
    local = Path(local_path)
    if missing_ok and not remote_exists(remote_name, bucket=bucket):
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    _hub().download_bucket_files(
        bucket, files=[(remote_name, str(local))], raise_on_missing_files=True
    )
    return local


def list_remote(prefix: str = "", *, bucket: str = DEFAULT_BUCKET) -> list[str]:
    """Return the paths of every file in the bucket under ``prefix``."""
    hub = _hub()
    return [
        item.path
        for item in hub.list_bucket_tree(bucket, recursive=True)
        if getattr(item, "type", "file") == "file" and item.path.startswith(prefix)
    ]
