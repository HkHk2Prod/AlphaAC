"""Tests for the model-checkpoint Hugging Face push/pull helpers.

A fake ``huggingface_hub`` module backed by an in-memory store stands in for the
bucket, so uploads persist and later downloads (and the cross-run rollups) can be
asserted without any network or real dependency.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from ac_zero.training import hub_checkpoints as hc
from ac_zero.training.checkpoint_bundle import CheckpointBundle


class _Item:
    def __init__(self, path: str, type: str = "file") -> None:
        self.path = path
        self.type = type


def _install_bucket(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Install a fake hub whose bucket is a ``remote_path -> bytes`` dict."""
    store: dict[str, bytes] = {}
    module = types.ModuleType("huggingface_hub")

    def list_bucket_tree(bucket: str, recursive: bool = False):  # type: ignore[no-untyped-def]
        return [_Item(path) for path in store]

    def batch_bucket_files(bucket: str, add=None, delete=None, copy=None):  # type: ignore[no-untyped-def]
        for local, remote in add or []:
            store[remote] = Path(local).read_bytes()

    def download_bucket_files(bucket: str, files=None):  # type: ignore[no-untyped-def]
        for remote, local in files or []:
            Path(local).write_bytes(store[remote])

    module.list_bucket_tree = list_bucket_tree  # type: ignore[attr-defined]
    module.batch_bucket_files = batch_bucket_files  # type: ignore[attr-defined]
    module.download_bucket_files = download_bucket_files  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return store


def _write_bundle(
    directory: Path, *, name: str, run_id: str, metric: float, steps: int = 3
) -> CheckpointBundle:
    """Write a minimal bundle with ``steps`` metric rows and best metric ``metric``."""
    bundle = CheckpointBundle(directory)
    payload = {"schema_version": "v1", "checkpoint_metric": metric, "model_state": {"a": 1}}
    bundle.save_checkpoint(payload, metric=metric)
    rows = [
        {
            "optimizer_step": i + 1,
            "total_loss": 1.0 / (i + 1),
            "policy_loss": 0.5,
            "value_loss": 0.1,
            "mean_return": metric,
            "success_rate": 0.5,
        }
        for i in range(steps)
    ]
    bundle.save_metrics(rows)
    bundle.save_meta(
        {
            "checkpoint_name": name,
            "run_id": run_id,
            "iteration": steps,
            "optimizer_step": steps,
            "updated_at": 1,
        }
    )
    return bundle


def test_download_best_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_bucket(monkeypatch)
    result = hc.download_best_checkpoint("some-name", tmp_path / "best.json", bucket="ns/b")
    assert result is None


def test_push_uploads_bundle_index_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _install_bucket(monkeypatch)
    _write_bundle(tmp_path / "bundle", name="name-a", run_id="100-0", metric=0.3)

    prefix = hc.push_checkpoint_bundle(tmp_path / "bundle", bucket="ns/b")

    assert prefix == "model_checkpoints/name-a"
    assert f"{prefix}/best.json" in store
    assert f"{prefix}/latest.json" in store
    assert f"{prefix}/runs/100-0.metrics.jsonl" in store
    assert f"{prefix}/runs/100-0.meta.json" in store
    assert f"{prefix}/index.json" in store
    # Both per-run and all-runs plots were rendered and uploaded.
    assert any(k.startswith(f"{prefix}/plots/100-0/") for k in store)
    assert any(k.startswith(f"{prefix}/plots/all_runs/") for k in store)
    index = json.loads(store[f"{prefix}/index.json"])
    assert index["best"]["run_id"] == "100-0"
    assert index["best"]["metric"] == 0.3
    assert [r["run_id"] for r in index["runs"]] == ["100-0"]


def test_best_promoted_only_when_metric_improves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _install_bucket(monkeypatch)

    _write_bundle(tmp_path / "a", name="name", run_id="100-0", metric=0.5)
    hc.push_checkpoint_bundle(tmp_path / "a", bucket="ns/b")
    best_after_a = store["model_checkpoints/name/best.json"]

    # A weaker second run must not overwrite the recorded best.
    _write_bundle(tmp_path / "b", name="name", run_id="200-0", metric=0.2)
    hc.push_checkpoint_bundle(tmp_path / "b", bucket="ns/b")
    index = json.loads(store["model_checkpoints/name/index.json"])
    assert index["best"]["run_id"] == "100-0"
    assert store["model_checkpoints/name/best.json"] == best_after_a
    # Both runs are recorded, and the all-runs plot now spans both.
    assert {r["run_id"] for r in index["runs"]} == {"100-0", "200-0"}

    # A stronger third run takes over as best.
    _write_bundle(tmp_path / "c", name="name", run_id="300-0", metric=0.9)
    hc.push_checkpoint_bundle(tmp_path / "c", bucket="ns/b")
    index = json.loads(store["model_checkpoints/name/index.json"])
    assert index["best"]["run_id"] == "300-0"
    assert json.loads(store["model_checkpoints/name/best.json"])["checkpoint_metric"] == 0.9


def test_periodic_uploader_throttles_and_flushes_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _install_bucket(monkeypatch)
    _write_bundle(tmp_path / "bundle", name="name", run_id="100-0", metric=0.4)
    uploader = hc.PeriodicCheckpointUploader(tmp_path / "bundle", bucket="ns/b", every_hours=10.0)

    # The interval has not elapsed, so a checkpoint event does not upload yet.
    uploader.on_event(types.SimpleNamespace(phase="checkpoint"))
    assert "model_checkpoints/name/latest.json" not in store

    # close() always flushes the final bundle.
    uploader.close()
    assert "model_checkpoints/name/latest.json" in store


def test_periodic_uploader_swallows_upload_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_bucket(monkeypatch)
    _write_bundle(tmp_path / "bundle", name="name", run_id="1-0", metric=0.4)
    uploader = hc.PeriodicCheckpointUploader(tmp_path / "bundle", bucket="ns/b")

    def _boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("network down")

    monkeypatch.setattr(hc, "push_checkpoint_bundle", _boom)
    uploader.close()  # must not raise
    assert "checkpoint-upload" in capsys.readouterr().out
