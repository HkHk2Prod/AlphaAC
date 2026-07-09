"""Tests for the Hugging Face bucket dataset helpers.

`huggingface_hub` is an optional dependency, so every test injects a fake module
into `sys.modules` and asserts the wrapper calls the bucket API correctly. No
network access and no real `huggingface_hub` install is required.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ac_zero.datasets import hub


class _Item:
    def __init__(self, path: str, type: str = "file") -> None:
        self.path = path
        self.type = type


def _install_fake(monkeypatch: pytest.MonkeyPatch, tree: list[_Item]) -> dict:
    """Register a fake `huggingface_hub` module; return a call-record dict."""
    record: dict = {}
    module = types.ModuleType("huggingface_hub")

    def list_bucket_tree(bucket: str, recursive: bool = False):  # type: ignore[no-untyped-def]
        record["list"] = {"bucket": bucket, "recursive": recursive}
        return list(tree)

    def batch_bucket_files(bucket: str, add=None, delete=None, copy=None):  # type: ignore[no-untyped-def]
        record["add"] = {"bucket": bucket, "add": add}

    def download_bucket_files(bucket: str, files=None, *, raise_on_missing_files=False):  # type: ignore[no-untyped-def]
        record["download"] = {
            "bucket": bucket,
            "files": files,
            "raise_on_missing_files": raise_on_missing_files,
        }
        for _remote, local in files or []:
            Path(local).write_text("{}", encoding="utf-8")

    module.list_bucket_tree = list_bucket_tree  # type: ignore[attr-defined]
    module.batch_bucket_files = batch_bucket_files  # type: ignore[attr-defined]
    module.download_bucket_files = download_bucket_files  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return record


def test_upload_dataset_calls_batch_and_returns_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_fake(monkeypatch, [])
    local = tmp_path / "train_rank2.json"
    local.write_text("{}", encoding="utf-8")

    uri = hub.upload_dataset(local, bucket="ns/bucket")

    assert uri == "hf://buckets/ns/bucket/train_rank2.json"
    assert record["add"] == {"bucket": "ns/bucket", "add": [(str(local), "train_rank2.json")]}


def test_upload_uses_default_bucket_and_custom_remote_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_fake(monkeypatch, [])
    local = tmp_path / "local.json"
    local.write_text("{}", encoding="utf-8")

    uri = hub.upload_dataset(local, remote_name="train_rank2.json")

    assert uri == f"hf://buckets/{hub.DEFAULT_BUCKET}/train_rank2.json"
    assert record["add"]["bucket"] == hub.DEFAULT_BUCKET
    assert record["add"]["add"] == [(str(local), "train_rank2.json")]


def test_upload_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [])
    with pytest.raises(FileNotFoundError):
        hub.upload_dataset(tmp_path / "nope.json")


def test_download_dataset_writes_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_fake(monkeypatch, [_Item("train_rank2.json")])
    target = tmp_path / "sub" / "train_rank2.json"

    result = hub.download_dataset(target, bucket="ns/bucket")

    assert result == target
    assert target.is_file()  # parent dir was created and the file written
    # An absent object must raise rather than silently leave `target` unwritten.
    assert record["download"] == {
        "bucket": "ns/bucket",
        "files": [("train_rank2.json", str(target))],
        "raise_on_missing_files": True,
    }


def test_download_missing_ok_returns_none_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_fake(monkeypatch, [])  # empty bucket
    target = tmp_path / "train_rank2.json"

    result = hub.download_dataset(target, missing_ok=True)

    assert result is None
    assert not target.exists()
    assert "download" not in record  # nothing was fetched


def test_download_missing_ok_fetches_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, [_Item("train_rank2.json")])
    target = tmp_path / "train_rank2.json"

    result = hub.download_dataset(target, missing_ok=True)

    assert result == target
    assert target.is_file()


def test_remote_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [_Item("train_rank2.json"), _Item("sub", type="directory")])
    assert hub.remote_exists("train_rank2.json") is True
    assert hub.remote_exists("missing.json") is False
    assert hub.remote_exists("sub") is False  # directory entries are not files


def test_upload_files_batches_pairs_with_nested_remote_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_fake(monkeypatch, [])
    a = tmp_path / "best.json"
    b = tmp_path / "runs" / "1.jsonl"
    b.parent.mkdir()
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")

    hub.upload_files(
        [(a, "model_checkpoints/n/best.json"), (b, "model_checkpoints/n/runs/1.jsonl")],
        bucket="ns/b",
    )

    assert record["add"]["bucket"] == "ns/b"
    assert record["add"]["add"] == [
        (str(a), "model_checkpoints/n/best.json"),
        (str(b), "model_checkpoints/n/runs/1.jsonl"),
    ]


def test_upload_files_rejects_missing_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, [])
    with pytest.raises(FileNotFoundError):
        hub.upload_files([(tmp_path / "nope.json", "remote/x.json")])


def test_list_remote_filters_by_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(
        monkeypatch,
        [
            _Item("model_checkpoints/n/runs/1.jsonl"),
            _Item("model_checkpoints/n/best.json"),
            _Item("other/thing.json"),
            _Item("model_checkpoints/n/runs", type="directory"),
        ],
    )
    paths = hub.list_remote("model_checkpoints/n/runs/", bucket="ns/b")
    assert paths == ["model_checkpoints/n/runs/1.jsonl"]  # directory + other prefixes excluded


def test_download_file_missing_ok_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, [])  # empty bucket
    result = hub.download_file(
        "model_checkpoints/n/index.json", tmp_path / "i.json", missing_ok=True
    )
    assert result is None


def test_missing_dependency_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes `import huggingface_hub` raise ImportError.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError, match="pip install ac-zero\\[hub\\]"):
        hub.remote_exists("train_rank2.json")
