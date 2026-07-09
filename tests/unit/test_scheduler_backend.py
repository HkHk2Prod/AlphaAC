"""BucketStateBackend + the make_state_backend factory.

The bucket wrappers (``ac_zero.datasets.hub``) are faked, so no real HF bucket or
``huggingface_hub`` install is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ac_zero.datasets.hub as hub
from ac_zero.scheduler.backend import (
    BucketStateBackend,
    HubStateBackend,
    make_state_backend,
)


def _fake_bucket(monkeypatch: pytest.MonkeyPatch, store: dict[str, str]) -> None:
    def _download(
        remote: str, local: str | Path, *, bucket: str, missing_ok: bool = False
    ) -> Path | None:
        if remote not in store:
            if missing_ok:
                return None
            raise FileNotFoundError(remote)
        path = Path(local)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(store[remote], encoding="utf-8")
        return path

    def _upload(pairs: list[tuple[str | Path, str]], *, bucket: str) -> None:
        for local, remote in pairs:
            store[remote] = Path(local).read_text(encoding="utf-8")

    monkeypatch.setattr(hub, "download_file", _download)
    monkeypatch.setattr(hub, "upload_files", _upload)


def test_bucket_backend_has_no_head_sha() -> None:
    assert BucketStateBackend("u/bucket").head_sha() is None


def test_bucket_backend_reads_existing_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_bucket(monkeypatch, {"queue.yaml": "hello"})
    b = BucketStateBackend("u/bucket")
    assert b.read_text("queue.yaml") == "hello"
    assert b.read_text("absent.json") is None


def test_bucket_backend_commit_writes_all_files(monkeypatch: pytest.MonkeyPatch) -> None:
    store: dict[str, str] = {}
    _fake_bucket(monkeypatch, store)
    b = BucketStateBackend("u/bucket")
    sha = b.commit(
        {"queue.yaml": "q", "runs/latest.json": "{}\n"}, message="m", parent_sha=None
    )
    assert sha == ""  # buckets have no head SHA
    assert store["queue.yaml"] == "q"
    assert store["runs/latest.json"] == "{}\n"


def test_bucket_backend_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    store: dict[str, str] = {}
    _fake_bucket(monkeypatch, store)
    b = BucketStateBackend("u/bucket")
    b.commit({"scheduler_state.json": '{"a": 1}'}, message="m", parent_sha=None)
    assert b.read_text("scheduler_state.json") == '{"a": 1}'


def test_make_state_backend_selects_bucket() -> None:
    made = make_state_backend("u/bucket", token="hf_x", repo_type="bucket")
    assert isinstance(made, BucketStateBackend)


def test_make_state_backend_selects_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    module = types.ModuleType("huggingface_hub")
    module.HfApi = lambda token=None: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    made = make_state_backend("u/repo", token="hf_x", repo_type="dataset")
    assert isinstance(made, HubStateBackend)
