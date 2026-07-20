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

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from ac_zero.datasets.remote_paths import dataset_remote_name

# Bucket holding the AlphaAC training datasets (namespace/bucket-name).
DEFAULT_BUCKET = "HkHk2Prod/alphaac-data"

_INSTALL_HINT = (
    "Hugging Face bucket access needs the optional `huggingface_hub` dependency; "
    "install it with `pip install ac-zero[hub]` (or `pip install 'huggingface_hub>=1.21'`)."
)

# The bucket transfer (an ``hf_xet`` call with no timeout) can wedge on a stalled
# connection and block forever -- a scheduled Kaggle run then burns its whole session
# in a silent download and reports itself healthy the entire time. So each fetch runs
# under a wall-clock deadline in a daemon thread: a transfer that overruns is abandoned
# and retried on a fresh connection, and a run that cannot download fails in minutes
# rather than hanging for hours. The native call cannot be interrupted, so the wedged
# thread is left to die with the process; the retry simply opens a new one. Tunable by
# environment so a genuinely large transfer can be given more room without a code change.
_DOWNLOAD_TIMEOUT_S = float(os.environ.get("ACZERO_DOWNLOAD_TIMEOUT_S", "300"))
_DOWNLOAD_ATTEMPTS = max(1, int(os.environ.get("ACZERO_DOWNLOAD_ATTEMPTS", "3")))
_DOWNLOAD_BACKOFF_S = float(os.environ.get("ACZERO_DOWNLOAD_BACKOFF_S", "2"))


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


def _remote_size_once(remote_name: str, bucket: str) -> int | None:
    """The real byte size of a single bucket path; ``None`` when it is absent.

    Lists the tree scoped to ``remote_name`` rather than walking the whole bucket (the
    bucket also holds every run's checkpoint tree, so an unscoped recursive listing is
    slow and, with no deadline, can wedge a run at startup) and rather than the per-file
    metadata endpoint. ``get_bucket_file_metadata`` reads ``Content-Length`` from the
    resolve endpoint, which for a Xet-backed file is the size of the ~1 KB Xet pointer,
    not the reconstructed file -- so a 29 GB dataset reports ~1 KB and every freshness
    check sees a mismatch and re-downloads. The tree listing carries the true object
    size, and a ``prefix`` scopes it to the one path so it stays as cheap as a single
    metadata fetch.
    """
    for item in _hub().list_bucket_tree(bucket, prefix=remote_name, recursive=True):
        if getattr(item, "type", "file") == "file" and item.path == remote_name:
            return int(item.size)
    return None


def remote_size(remote_name: str, *, bucket: str = DEFAULT_BUCKET) -> int | None:
    """Return the byte size of ``remote_name`` in the bucket, or ``None`` if it is absent.

    Used to decide whether a local dataset copy is already the bucket's current one:
    these files are hundreds of megabytes of JSON, so any content change moves the byte
    count, and a size match means a re-download would only rewrite identical bytes. It is
    cheaper than a content hash and needs no local re-chunking -- the bucket's ``xet_hash``
    cannot be recomputed on disk without the Xet chunker.

    Runs under the same per-attempt wall-clock deadline as :func:`download_file`, so a
    stalled connection is abandoned and retried on a fresh one rather than hanging the
    caller (a supervised run queries this before it trains anything).
    """
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        status, value, error = _run_with_deadline(
            lambda: _remote_size_once(remote_name, bucket), f"hf-metadata-{attempt}"
        )
        if status == "ok":
            return cast("int | None", value)
        if status == "error":
            assert error is not None  # an error status always carries its exception
            raise error
        print(
            f"hub: metadata for {remote_name!r} did not respond within "
            f"{_DOWNLOAD_TIMEOUT_S:.0f}s (attempt {attempt}/{_DOWNLOAD_ATTEMPTS})",
            file=sys.stderr,
            flush=True,
        )
        if attempt < _DOWNLOAD_ATTEMPTS:
            time.sleep(_DOWNLOAD_BACKOFF_S * attempt)
    raise TimeoutError(
        f"metadata for {remote_name!r} from bucket {bucket!r} did not complete in "
        f"{_DOWNLOAD_ATTEMPTS} attempts of {_DOWNLOAD_TIMEOUT_S:.0f}s; the bucket "
        f"is not responding"
    )


def upload_dataset(
    local_path: str | Path,
    *,
    remote_name: str | None = None,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    """Upload a local dataset file to the bucket and return its ``hf://`` URI.

    ``remote_name`` defaults to the file's folder in the bucket, derived from its
    name (see :func:`ac_zero.datasets.remote_paths.dataset_remote_name`) -- e.g.
    ``ball_rank2_rel48.groups.json`` lands under ``datasets/rank2/rel-48/``. Pass
    ``remote_name`` to place the file at an explicit path instead.
    """
    hub = _hub()
    local = Path(local_path)
    if not local.is_file():
        raise FileNotFoundError(f"dataset not found: {local}")
    name = remote_name or dataset_remote_name(local.name)
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

    ``remote_name`` defaults to the file's folder in the bucket, derived from its
    name (see :func:`ac_zero.datasets.remote_paths.dataset_remote_name`), so a run
    pulls ``ball_rank2_rel48.groups.json`` from ``datasets/rank2/rel-48/`` while
    writing it under its bare on-disk name. With ``missing_ok=True`` this returns
    ``None`` instead of raising when the file is absent from the bucket -- used by
    the notebook's "resume the run if a dataset already exists" first pass.
    """
    local = Path(local_path)
    remote = remote_name or dataset_remote_name(local.name)
    return download_file(remote, local, bucket=bucket, missing_ok=missing_ok)


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


def _download_once(
    remote_name: str, local_path: str | Path, bucket: str, missing_ok: bool
) -> Path | None:
    """One bucket fetch, no timeout: the transfer this module guards with a deadline."""
    local = Path(local_path)
    if missing_ok and not remote_exists(remote_name, bucket=bucket):
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    _hub().download_bucket_files(
        bucket, files=[(remote_name, str(local))], raise_on_missing_files=True
    )
    return local


def _run_with_deadline(
    thunk: Callable[[], Any], name: str
) -> tuple[str, Any, BaseException | None]:
    """Run ``thunk`` in a daemon thread under the deadline; report ``ok``/``error``/``timeout``.

    A bucket call is an uninterruptible native call, so a thread that overruns the
    deadline is left to die with the process and reported as ``timeout`` for the caller
    to retry on a fresh connection. Shared by the download and metadata paths.
    """
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = thunk()
        except BaseException as exc:  # surfaced in the calling thread below
            outcome["error"] = exc

    worker = threading.Thread(target=run, name=name, daemon=True)
    worker.start()
    worker.join(_DOWNLOAD_TIMEOUT_S)
    if worker.is_alive():
        return "timeout", None, None
    error = outcome.get("error")
    if error is not None:
        return "error", None, error
    return "ok", outcome.get("result"), None


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

    The transfer runs under a per-attempt wall-clock deadline (see
    ``_DOWNLOAD_TIMEOUT_S``); one that overruns is abandoned and retried on a fresh
    connection, and a transfer that never responds raises ``TimeoutError`` after
    ``_DOWNLOAD_ATTEMPTS`` rather than blocking the caller indefinitely. Errors that
    the fetch itself raises (a missing required object, a bad token) are propagated
    at once -- only an unresponsive transfer is retried.
    """
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        status, value, error = _run_with_deadline(
            lambda: _download_once(remote_name, local_path, bucket, missing_ok),
            f"hf-download-{attempt}",
        )
        if status == "ok":
            return cast("Path | None", value)
        if status == "error":
            assert error is not None  # an error status always carries its exception
            raise error
        # The transfer overran its deadline. The native call cannot be interrupted, so
        # the thread is left as a daemon to die with the process; a fresh attempt opens
        # a new connection. Drop any partial file so it is never mistaken for a success.
        print(
            f"hub: download of {remote_name!r} did not respond within "
            f"{_DOWNLOAD_TIMEOUT_S:.0f}s (attempt {attempt}/{_DOWNLOAD_ATTEMPTS})",
            file=sys.stderr,
            flush=True,
        )
        Path(local_path).unlink(missing_ok=True)
        if attempt < _DOWNLOAD_ATTEMPTS:
            time.sleep(_DOWNLOAD_BACKOFF_S * attempt)
    raise TimeoutError(
        f"download of {remote_name!r} from bucket {bucket!r} did not complete in "
        f"{_DOWNLOAD_ATTEMPTS} attempts of {_DOWNLOAD_TIMEOUT_S:.0f}s; the bucket "
        f"transfer is not responding"
    )


def list_remote(prefix: str = "", *, bucket: str = DEFAULT_BUCKET) -> list[str]:
    """Return the paths of every file in the bucket under ``prefix``."""
    hub = _hub()
    return [
        item.path
        for item in hub.list_bucket_tree(bucket, recursive=True)
        if getattr(item, "type", "file") == "file" and item.path.startswith(prefix)
    ]


def _files_under(prefix: str, *, bucket: str) -> list[Any]:
    """Bucket file entries under ``prefix``, listed with the prefix pushed server-side."""
    return [
        item
        for item in _hub().list_bucket_tree(bucket, prefix=prefix, recursive=True)
        if getattr(item, "type", "file") == "file" and item.path.startswith(prefix)
    ]


def move_remote_tree(prefix: str, dest_prefix: str, *, bucket: str = DEFAULT_BUCKET) -> list[str]:
    """Move every file under ``prefix`` to ``dest_prefix``; return the new paths.

    The copy is a server-side xet-hash copy -- no bytes are downloaded or re-uploaded,
    so moving a whole checkpoint lineage aside costs one API round trip rather than a
    transfer of every model it ever wrote. Copy and delete are issued as two batches,
    not one: ``batch_bucket_files`` is non-transactional, and a copy that half-fails
    must leave the originals in place rather than delete what it did not copy.
    """
    items = _files_under(prefix, bucket=bucket)
    if not items:
        return []
    sources = [str(item.path) for item in items]
    destinations = [dest_prefix + source[len(prefix) :] for source in sources]
    copies = [
        ("bucket", bucket, item.xet_hash, destination)
        for item, destination in zip(items, destinations, strict=True)
    ]
    hub = _hub()
    hub.batch_bucket_files(bucket, copy=copies)
    hub.batch_bucket_files(bucket, delete=sources)
    return destinations


def delete_remote(paths: list[str], *, bucket: str = DEFAULT_BUCKET) -> None:
    """Delete ``paths`` from the bucket, ignoring any that are already absent."""
    if paths:
        _hub().batch_bucket_files(bucket, delete=paths)
