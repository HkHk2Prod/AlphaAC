import json
from pathlib import Path

import pytest

import ac_zero.datasets.publish as publish
from ac_zero.datasets.publish import publish_to_bucket
from ac_zero.datasets.summary import write_dataset_summary


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train_rank2.groups.json"
    path.write_text(
        json.dumps({"rank": 2, "provenance": {"generator": "test"}, "groups": []}),
        encoding="utf-8",
    )
    return path


def _record_uploads(monkeypatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_upload_dataset(local, *, remote_name=None, bucket):
        remote = remote_name or Path(local).name
        calls.append((remote, bucket))
        return f"hf://buckets/{bucket}/{remote}"

    monkeypatch.setattr(publish, "upload_dataset", fake_upload_dataset)
    return calls


def test_publish_writes_summary_and_uploads_data_and_summary(monkeypatch, tmp_path: Path) -> None:
    data = _write_dataset(tmp_path)
    calls = _record_uploads(monkeypatch)

    result = publish_to_bucket(
        data, summary_writer=write_dataset_summary, summary_dir=tmp_path, bucket="ns/bucket"
    )

    assert result.summary_path == tmp_path / "train_rank2.groups.summary.md"
    assert result.summary_path.exists()
    # Data file and its groups summary both land in the dataset's bucket folder.
    assert {remote for remote, _ in calls} == {
        "datasets/rank2/rel-unbounded/train_rank2.groups.json",
        "datasets/rank2/rel-unbounded/train_rank2.groups.summary.md",
    }
    assert {bucket for _, bucket in calls} == {"ns/bucket"}
    assert len(result.uploaded_uris) == 2


def test_publish_upload_false_writes_summary_but_skips_bucket(monkeypatch, tmp_path: Path) -> None:
    data = _write_dataset(tmp_path)
    calls = _record_uploads(monkeypatch)

    result = publish_to_bucket(
        data, summary_writer=write_dataset_summary, summary_dir=tmp_path, upload=False
    )

    assert result.summary_path.exists()  # summary still written locally
    assert result.outcomes == []
    assert calls == []


def test_publish_without_summary_writer_uploads_only_the_data_file(
    monkeypatch, tmp_path: Path
) -> None:
    data = _write_dataset(tmp_path)
    calls = _record_uploads(monkeypatch)

    result = publish_to_bucket(data)

    assert result.summary_path is None
    assert [remote for remote, _ in calls] == [
        "datasets/rank2/rel-unbounded/train_rank2.groups.json"
    ]


def test_publish_is_lenient_when_an_upload_fails(monkeypatch, tmp_path: Path) -> None:
    data = _write_dataset(tmp_path)

    def boom(local, *, remote_name=None, bucket):
        raise RuntimeError("no HF_TOKEN")

    monkeypatch.setattr(publish, "upload_dataset", boom)

    # Never raises: every file is recorded as a skip with its error message.
    result = publish_to_bucket(data, summary_writer=write_dataset_summary, summary_dir=tmp_path)
    assert result.summary_path.exists()
    assert result.uploaded_uris == []
    assert all(not o.ok and "HF_TOKEN" in (o.error or "") for o in result.outcomes)


def test_publish_requires_summary_dir_when_a_writer_is_given(tmp_path: Path) -> None:
    data = _write_dataset(tmp_path)
    with pytest.raises(ValueError, match="summary_dir"):
        publish_to_bucket(data, summary_writer=write_dataset_summary)
