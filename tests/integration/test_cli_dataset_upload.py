"""`dataset grow`/`annotate` push their data file and summary to the HF bucket."""

from pathlib import Path

import ac_zero.cli as cli
import ac_zero.datasets.publish as publish
from ac_zero.cli import main


def _capture_uploads(monkeypatch):
    """Recorder for the bucket calls the publish helper makes; returns the call list.

    Patches the single ``upload_dataset`` the shared ``publish_to_bucket`` uses, so
    each ``(remote_name, bucket)`` the CLI would push is captured without touching HF.
    """
    calls: list[tuple[str, str]] = []

    def fake_upload_dataset(local, *, remote_name=None, bucket):
        remote = remote_name or Path(local).name
        calls.append((remote, bucket))
        return f"hf://buckets/{bucket}/{remote}"

    monkeypatch.setattr(publish, "upload_dataset", fake_upload_dataset)
    return calls


def test_cli_grow_uploads_dataset_and_summary_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _capture_uploads(monkeypatch)
    dataset_path = tmp_path / "data/generated/grown.json"

    assert main(["dataset", "grow", "--output", str(dataset_path), "--target", "15"]) == 0

    assert {bucket for _, bucket in calls} == {cli.DEFAULT_BUCKET}
    assert {remote for remote, _ in calls} == {
        "grown.json",
        "datasets_summaries/grown.summary.md",
    }


def test_cli_grow_no_upload_skips_the_bucket(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _capture_uploads(monkeypatch)
    dataset_path = tmp_path / "data/generated/grown.json"

    assert (
        main(["dataset", "grow", "--output", str(dataset_path), "--target", "15", "--no-upload"])
        == 0
    )
    assert calls == []
    # The summary is still written locally; only the push is suppressed.
    assert (tmp_path / "data/summaries/grown.summary.md").exists()


def test_cli_grow_no_summary_uploads_only_the_dataset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _capture_uploads(monkeypatch)
    dataset_path = tmp_path / "data/generated/grown.json"

    assert (
        main(["dataset", "grow", "--output", str(dataset_path), "--target", "15", "--no-summary"])
        == 0
    )
    assert [remote for remote, _ in calls] == ["grown.json"]


def test_cli_grow_upload_failure_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "data/generated/grown.json"

    def boom(local, *, remote_name=None, bucket):
        raise RuntimeError("no HF_TOKEN")

    monkeypatch.setattr(publish, "upload_dataset", boom)
    # A failed push (missing creds/dep) warns but the command still succeeds.
    assert main(["dataset", "grow", "--output", str(dataset_path), "--target", "15"]) == 0
    assert (tmp_path / "data/summaries/grown.summary.md").exists()


def test_cli_annotate_writes_and_uploads_its_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "data/generated/grown.groups.json"
    assert (
        main(["dataset", "grow", "--output", str(dataset_path), "--target", "15", "--no-upload"])
        == 0
    )

    calls = _capture_uploads(monkeypatch)
    assert (
        main(["dataset", "annotate", "--input", str(dataset_path), "--moveset", "universal"]) == 0
    )

    summary_path = tmp_path / "data/summaries/grown.universal.annotations.summary.md"
    assert summary_path.exists()
    assert {bucket for _, bucket in calls} == {cli.DEFAULT_BUCKET}
    assert {remote for remote, _ in calls} == {
        "grown.universal.annotations.json",
        "datasets_summaries/grown.universal.annotations.summary.md",
    }
