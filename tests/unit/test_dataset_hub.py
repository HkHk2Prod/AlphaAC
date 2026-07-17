"""Tests for the Hugging Face bucket dataset helpers.

`huggingface_hub` is an optional dependency, so every test injects a fake module
into `sys.modules` and asserts the wrapper calls the bucket API correctly. No
network access and no real `huggingface_hub` install is required.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

from ac_zero.datasets import hub


class _Item:
    def __init__(self, path: str, type: str = "file", size: int = 0) -> None:
        self.path = path
        self.type = type
        self.size = size


def _install_fake(monkeypatch: pytest.MonkeyPatch, tree: list[_Item]) -> dict:
    """Register a fake `huggingface_hub` module; return a call-record dict."""
    record: dict = {}
    module = types.ModuleType("huggingface_hub")

    def list_bucket_tree(bucket: str, prefix: str | None = None, *, recursive: bool = False):  # type: ignore[no-untyped-def]
        record["list"] = {"bucket": bucket, "prefix": prefix, "recursive": recursive}
        return [it for it in tree if prefix is None or it.path.startswith(prefix)]

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

    def disable_progress_bars() -> None:
        record["progress_disabled"] = True

    module.list_bucket_tree = list_bucket_tree  # type: ignore[attr-defined]
    module.batch_bucket_files = batch_bucket_files  # type: ignore[attr-defined]
    module.download_bucket_files = download_bucket_files  # type: ignore[attr-defined]
    module.utils = types.SimpleNamespace(  # type: ignore[attr-defined]
        disable_progress_bars=disable_progress_bars
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return record


def test_upload_dataset_calls_batch_and_returns_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_fake(monkeypatch, [])
    local = tmp_path / "train_rank2.json"
    local.write_text("{}", encoding="utf-8")

    uri = hub.upload_dataset(local, bucket="ns/bucket")

    # The remote name is derived from the filename: train_rank2.json is filed by rank.
    remote = "datasets/rank2/rel-unbounded/train_rank2.json"
    assert uri == f"hf://buckets/ns/bucket/{remote}"
    assert record["add"] == {"bucket": "ns/bucket", "add": [(str(local), remote)]}


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


def test_upload_disables_hub_progress_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake(monkeypatch, [])
    local = tmp_path / "train_rank2.json"
    local.write_text("{}", encoding="utf-8")

    hub.upload_dataset(local, bucket="ns/bucket")

    assert record["progress_disabled"] is True


def test_upload_tolerates_hub_without_progress_bar_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, [])
    del sys.modules["huggingface_hub"].utils  # type: ignore[attr-defined]
    local = tmp_path / "train_rank2.json"
    local.write_text("{}", encoding="utf-8")

    assert hub.upload_dataset(local, bucket="ns/bucket").endswith("train_rank2.json")


def test_upload_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [])
    with pytest.raises(FileNotFoundError):
        hub.upload_dataset(tmp_path / "nope.json")


def test_download_dataset_writes_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = "datasets/rank2/rel-unbounded/train_rank2.json"
    record = _install_fake(monkeypatch, [_Item(remote)])
    target = tmp_path / "sub" / "train_rank2.json"

    result = hub.download_dataset(target, bucket="ns/bucket")

    assert result == target
    assert target.is_file()  # parent dir was created and the file written
    # The remote name is derived from the filename; an absent object must raise
    # rather than silently leave `target` unwritten.
    assert record["download"] == {
        "bucket": "ns/bucket",
        "files": [(remote, str(target))],
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
    _install_fake(monkeypatch, [_Item("datasets/rank2/rel-unbounded/train_rank2.json")])
    target = tmp_path / "train_rank2.json"

    result = hub.download_dataset(target, missing_ok=True)

    assert result == target
    assert target.is_file()


def test_remote_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [_Item("train_rank2.json"), _Item("sub", type="directory")])
    assert hub.remote_exists("train_rank2.json") is True
    assert hub.remote_exists("missing.json") is False
    assert hub.remote_exists("sub") is False  # directory entries are not files


def test_remote_size_returns_bytes_or_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(
        monkeypatch,
        [_Item("ball.groups.json", size=1234), _Item("sub", type="directory", size=99)],
    )
    assert hub.remote_size("ball.groups.json") == 1234
    assert hub.remote_size("missing.json") is None
    assert hub.remote_size("sub") is None  # directory entries are not files


def test_remote_size_scopes_tree_listing_to_one_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # A prefix-scoped listing, not an unscoped recursive walk of the whole bucket tree.
    record = _install_fake(monkeypatch, [_Item("ball.groups.json", size=1234)])
    assert hub.remote_size("ball.groups.json", bucket="ns/b") == 1234
    assert record["list"] == {"bucket": "ns/b", "prefix": "ball.groups.json", "recursive": True}


def test_remote_size_ignores_a_prefix_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    # A prefix match on a sibling path must not be mistaken for the requested file.
    _install_fake(
        monkeypatch,
        [_Item("ball.groups.json.bak", size=1), _Item("ball.groups.json", size=1234)],
    )
    assert hub.remote_size("ball.groups.json") == 1234


def test_remote_size_retries_past_a_hung_metadata_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A metadata request that overruns its deadline is abandoned; the next one succeeds."""
    _fast_retry(monkeypatch)
    calls = {"n": 0}

    def flaky(remote_name, bucket):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(10)  # first attempt "hangs" past the 0.2s deadline
        return 42

    monkeypatch.setattr(hub, "_remote_size_once", flaky)
    assert hub.remote_size("ball.groups.json") == 42
    assert calls["n"] == 2  # the hung first attempt, then a clean retry


def test_remote_size_raises_when_every_metadata_call_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    _fast_retry(monkeypatch, attempts=2)
    monkeypatch.setattr(hub, "_remote_size_once", lambda *a, **k: time.sleep(10))
    with pytest.raises(TimeoutError, match="not responding"):
        hub.remote_size("ball.groups.json")


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


def _fast_retry(
    monkeypatch: pytest.MonkeyPatch, *, timeout: float = 0.2, attempts: int = 3
) -> None:
    """Shrink the download deadline/backoff so a hang test runs in a fraction of a second."""
    monkeypatch.setattr(hub, "_DOWNLOAD_TIMEOUT_S", timeout)
    monkeypatch.setattr(hub, "_DOWNLOAD_ATTEMPTS", attempts)
    monkeypatch.setattr(hub, "_DOWNLOAD_BACKOFF_S", 0.0)


def test_download_retries_past_a_hung_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transfer that overruns its deadline is abandoned and the next attempt succeeds."""
    _fast_retry(monkeypatch)
    target = tmp_path / "train_rank2.json"
    calls = {"n": 0}

    def flaky(remote_name, local_path, bucket, missing_ok):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(10)  # first attempt "hangs" past the 0.2s deadline
        Path(local_path).write_text("{}", encoding="utf-8")
        return Path(local_path)

    monkeypatch.setattr(hub, "_download_once", flaky)
    result = hub.download_file("train_rank2.json", target)

    assert result == target
    assert target.is_file()
    assert calls["n"] == 2  # the hung first attempt, then a clean retry


def test_download_raises_when_every_attempt_hangs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transfer that never responds fails fast instead of blocking the caller forever."""
    _fast_retry(monkeypatch, attempts=2)
    monkeypatch.setattr(hub, "_download_once", lambda *a, **k: time.sleep(10))

    with pytest.raises(TimeoutError, match="not responding"):
        hub.download_file("train_rank2.json", tmp_path / "x.json")


def test_download_propagates_fetch_errors_without_retrying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error the fetch itself raises (e.g. a missing required object) is not retried."""
    _fast_retry(monkeypatch)
    calls = {"n": 0}

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise FileNotFoundError("absent")

    monkeypatch.setattr(hub, "_download_once", boom)
    with pytest.raises(FileNotFoundError):
        hub.download_file("train_rank2.json", tmp_path / "x.json")
    assert calls["n"] == 1  # propagated on the first attempt, not retried three times
