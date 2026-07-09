"""File backends for the scheduler state store.

The scheduler treats a private Hugging Face location as a lightweight,
file-backed state store (not a transactional database). This module hides the
storage API behind a small :class:`StateBackend` protocol so the controller and
its tests share one code path:

* :class:`BucketStateBackend` -- stores state **in the same HF bucket as the
  training dataset** (``HkHk2Prod/alphaac-data``), via the tested
  ``ac_zero.datasets.hub`` bucket wrappers. Buckets are last-writer-wins (no
  cheap head SHA), so it reports ``head_sha() is None`` and relies on the
  scheduler lease + GitHub Actions ``concurrency`` to serialise ticks.
* :class:`HubStateBackend` -- stores state in a Hugging Face **dataset repo**;
  reads via ``hf_hub_download`` and writes an atomic multi-file commit via
  ``HfApi.create_commit``. Passing ``parent_commit`` gives optimistic
  concurrency: the commit is rejected if the branch head moved since we read it,
  surfacing as :class:`StateConflict`.
* :class:`MemoryStateBackend` -- an in-process dict used by unit tests; it
  emulates the same head-SHA guard so concurrency logic is testable without any
  network or credentials.

:func:`make_state_backend` picks the backend from a ``repo_type`` string
(``"bucket"`` or ``"dataset"``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Protocol


class StateConflict(RuntimeError):
    """Raised when a write's ``parent_sha`` no longer matches the remote head."""


# huggingface_hub is an optional dependency, so its exception classes are matched
# by name rather than imported -- keeps this module import-clean and type-checkable
# whether or not the extra is installed.
_NOT_FOUND_ERRORS = frozenset(
    {"RepositoryNotFoundError", "EntryNotFoundError", "RevisionNotFoundError"}
)


def _is_not_found(exc: Exception) -> bool:
    """Whether ``exc`` is a Hub 'missing repo/file' error (treated as absent)."""
    if type(exc).__name__ in _NOT_FOUND_ERRORS:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 404


class StateBackend(Protocol):
    """Minimal read/commit surface the scheduler needs from a file store."""

    def head_sha(self) -> str | None:
        """Current head commit SHA, or ``None`` if the repo is empty/unknown."""
        ...

    def read_text(self, path: str) -> str | None:
        """Return the file's text, or ``None`` if it does not exist."""
        ...

    def commit(
        self, files: dict[str, str], *, message: str, parent_sha: str | None
    ) -> str:
        """Atomically write ``{path: text}``.

        If ``parent_sha`` is not ``None`` and differs from the current head,
        raise :class:`StateConflict` without writing. Returns the new head SHA.
        """
        ...


class MemoryStateBackend:
    """In-memory :class:`StateBackend` for tests (deterministic fake SHAs)."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files: dict[str, str] = dict(files or {})
        self._counter = 0
        self._sha = self._next_sha() if self._files else None

    def _next_sha(self) -> str:
        self._counter += 1
        return f"sha{self._counter:04d}"

    def head_sha(self) -> str | None:
        return self._sha

    def read_text(self, path: str) -> str | None:
        return self._files.get(path)

    def commit(
        self, files: dict[str, str], *, message: str, parent_sha: str | None
    ) -> str:
        if parent_sha is not None and parent_sha != self._sha:
            raise StateConflict(
                f"remote state changed (head={self._sha!r}, expected={parent_sha!r})"
            )
        self._files.update(files)
        self._sha = self._next_sha()
        return self._sha


class HubStateBackend:
    """:class:`StateBackend` over a Hugging Face Hub dataset repo."""

    def __init__(self, repo_id: str, *, token: str, repo_type: str = "dataset") -> None:
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._token = token
        self._api = self._make_api(token)

    @staticmethod
    def _make_api(token: str) -> Any:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "The scheduler needs `huggingface_hub`; install `ac-zero[hub]`."
            ) from exc
        return HfApi(token=token)

    def head_sha(self) -> str | None:
        try:
            info = self._api.repo_info(self.repo_id, repo_type=self.repo_type)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        sha = getattr(info, "sha", None)
        return str(sha) if sha else None

    def read_text(self, path: str) -> str | None:
        from huggingface_hub import hf_hub_download

        try:
            local = hf_hub_download(
                self.repo_id,
                filename=path,
                repo_type=self.repo_type,
                token=self._token,
                force_download=True,
            )
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        with open(local, encoding="utf-8") as handle:
            return handle.read()

    def commit(
        self, files: dict[str, str], *, message: str, parent_sha: str | None
    ) -> str:
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=text.encode("utf-8"))
            for path, text in files.items()
        ]
        try:
            self._api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                operations=operations,
                commit_message=message,
                parent_commit=parent_sha,
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if parent_sha is not None and status in (409, 412):
                raise StateConflict(
                    f"remote {self.repo_id} moved past {parent_sha}: {exc}"
                ) from exc
            raise
        new = self.head_sha()
        return new or ""


class BucketStateBackend:
    """:class:`StateBackend` over an HF **bucket** (shared with the dataset).

    Reuses the tested ``ac_zero.datasets.hub`` bucket wrappers. Buckets are
    last-writer-wins, so ``head_sha`` is ``None`` (no optimistic-concurrency
    guard) -- overlap is prevented by the scheduler lease + GitHub concurrency.
    """

    def __init__(self, bucket: str, *, token: str | None = None) -> None:
        self.bucket = bucket
        if token:
            # The bucket wrappers authenticate via the ambient HF token.
            import os

            os.environ["HF_TOKEN"] = token

    def head_sha(self) -> str | None:
        return None

    def read_text(self, path: str) -> str | None:
        from ac_zero.datasets.hub import download_file

        with tempfile.TemporaryDirectory() as tmp:
            local = download_file(
                path, Path(tmp) / "state_file", bucket=self.bucket, missing_ok=True
            )
            if local is None:
                return None
            return local.read_text(encoding="utf-8")

    def commit(
        self, files: dict[str, str], *, message: str, parent_sha: str | None
    ) -> str:
        from ac_zero.datasets.hub import upload_files

        with tempfile.TemporaryDirectory() as tmp:
            pairs: list[tuple[str | Path, str]] = []
            for index, (remote, text) in enumerate(files.items()):
                local = Path(tmp) / f"state_{index}"
                local.write_text(text, encoding="utf-8")
                pairs.append((local, remote))
            upload_files(pairs, bucket=self.bucket)
        return ""


def make_state_backend(
    repo_id: str, *, token: str, repo_type: str = "dataset"
) -> StateBackend:
    """Build the backend for ``repo_type`` (``"bucket"`` or ``"dataset"``)."""
    if repo_type == "bucket":
        return BucketStateBackend(repo_id, token=token)
    return HubStateBackend(repo_id, token=token, repo_type=repo_type)
