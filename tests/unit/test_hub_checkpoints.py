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

from ac_zero.training.checkpointing import hub_checkpoints as hc
from ac_zero.training.checkpointing.checkpoint_bundle import CheckpointBundle


class _Item:
    def __init__(self, path: str, xet_hash: str = "", type: str = "file") -> None:
        self.path = path
        self.xet_hash = xet_hash
        self.type = type


def _install_bucket(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Install a fake hub whose bucket is a ``remote_path -> bytes`` dict.

    The xet hash a server-side copy addresses is faked as the path itself, so a
    ``copy`` resolves through the same dict the uploads wrote.
    """
    store: dict[str, bytes] = {}
    module = types.ModuleType("huggingface_hub")

    def list_bucket_tree(bucket: str, prefix: str = "", recursive: bool = False):  # type: ignore[no-untyped-def]
        return [_Item(path, xet_hash=path) for path in store if path.startswith(prefix)]

    def batch_bucket_files(bucket: str, add=None, delete=None, copy=None):  # type: ignore[no-untyped-def]
        for local, remote in add or []:
            store[remote] = Path(local).read_bytes()
        for _repo_type, _repo_id, xet_hash, destination in copy or []:
            store[destination] = store[xet_hash]
        for remote in delete or []:
            store.pop(remote, None)

    def download_bucket_files(bucket: str, files=None, *, raise_on_missing_files=False):  # type: ignore[no-untyped-def]
        for remote, local in files or []:
            if remote not in store:
                if raise_on_missing_files:
                    raise FileNotFoundError(remote)
                continue  # the real hub skips missing files with a warning
            Path(local).write_bytes(store[remote])

    module.list_bucket_tree = list_bucket_tree  # type: ignore[attr-defined]
    module.batch_bucket_files = batch_bucket_files  # type: ignore[attr-defined]
    module.download_bucket_files = download_bucket_files  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return store


def _write_bundle(
    directory: Path,
    *,
    name: str,
    run_id: str,
    metric: float,
    steps: int = 3,
    format_version: int = 1,
) -> CheckpointBundle:
    """Write a minimal bundle with ``steps`` metric rows and best metric ``metric``."""
    bundle = CheckpointBundle(directory)
    payload = {
        "schema_version": "v1",
        "checkpoint_metric": metric,
        "model_state": {"a": 1, "format_version": format_version},
    }
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


def test_push_prints_one_summary_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _install_bucket(monkeypatch)
    _write_bundle(tmp_path / "bundle", name="name-a", run_id="100-0", metric=0.3)

    hc.push_checkpoint_bundle(tmp_path / "bundle", bucket="ns/b")

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0] == (
        f"[checkpoint-upload] pushed {len(store)} files "
        f"({sum(len(b) for b in store.values()) / 1e6:.2f} MB) to ns/b/model_checkpoints/name-a"
    )


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


def test_push_copies_all_runs_plots_into_the_comparison_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every model's all-runs figures also land under plots/<type>/<name>.png."""
    store = _install_bucket(monkeypatch)
    _write_bundle(tmp_path / "a", name="model-a", run_id="100-0", metric=0.3)
    _write_bundle(tmp_path / "b", name="model-b", run_id="200-0", metric=0.4)

    hc.push_checkpoint_bundle(tmp_path / "a", bucket="ns/b")
    hc.push_checkpoint_bundle(tmp_path / "b", bucket="ns/b")

    # The comparison copy is byte-identical to the all-runs plot it mirrors...
    for name in ("model-a", "model-b"):
        source = f"model_checkpoints/{name}/plots/all_runs/loss_curves.png"
        assert store[hc.comparison_path("loss_curves", name)] == store[source]
    # ...and one folder now holds the same figure for both models, named after each.
    in_folder = {k for k in store if k.startswith("plots/selfplay_progress/")}
    assert in_folder == {
        "plots/selfplay_progress/model-a.png",
        "plots/selfplay_progress/model-b.png",
    }


def test_archive_lineage_empties_the_name_and_keeps_the_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start-fresh archive leaves nothing to resume from but discards nothing."""
    store = _install_bucket(monkeypatch)
    _write_bundle(tmp_path / "a", name="model-a", run_id="100-0", metric=0.3)
    hc.push_checkpoint_bundle(tmp_path / "a", bucket="ns/b")
    before = dict(store)

    moved = hc.archive_checkpoint_lineage("model-a", "20260719T000000Z", bucket="ns/b")

    assert moved == len([k for k in before if k.startswith("model_checkpoints/model-a/")])
    assert not [k for k in store if k.startswith("model_checkpoints/model-a/")]
    assert not [k for k in store if k.startswith("plots/") and k.endswith("/model-a.png")]
    archive = "model_checkpoints/_archive/model-a/20260719T000000Z"
    assert store[f"{archive}/best.json"] == before["model_checkpoints/model-a/best.json"]
    assert store[f"{archive}/runs/100-0.metrics.jsonl"]

    # With the name empty, the next run's rollups start over rather than replaying it.
    _write_bundle(tmp_path / "b", name="model-a", run_id="200-0", metric=0.1)
    hc.push_checkpoint_bundle(tmp_path / "b", bucket="ns/b")
    index = json.loads(store["model_checkpoints/model-a/index.json"])
    assert [r["run_id"] for r in index["runs"]] == ["200-0"]
    assert index["best"]["run_id"] == "200-0"  # the archived 0.3 no longer outranks it


def test_archive_lineage_on_an_unknown_name_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _install_bucket(monkeypatch)
    _write_bundle(tmp_path / "a", name="model-a", run_id="100-0", metric=0.3)
    hc.push_checkpoint_bundle(tmp_path / "a", bucket="ns/b")
    before = dict(store)

    assert hc.archive_checkpoint_lineage("never-trained", "20260719T000000Z", bucket="ns/b") == 0
    assert store == before  # a sibling model's comparison copy is untouched


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


def test_a_newer_model_format_promotes_even_with_a_worse_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics are only comparable within one format: a re-pretrain must land regardless.

    Otherwise the lineage keeps serving a `best.json` this code can no longer load, and
    every task seeding from it fails.
    """
    store = _install_bucket(monkeypatch)

    _write_bundle(tmp_path / "a", name="name", run_id="100-0", metric=0.9, format_version=1)
    hc.push_checkpoint_bundle(tmp_path / "a", bucket="ns/b")

    _write_bundle(tmp_path / "b", name="name", run_id="200-0", metric=0.2, format_version=2)
    hc.push_checkpoint_bundle(tmp_path / "b", bucket="ns/b")

    index = json.loads(store["model_checkpoints/name/index.json"])
    assert index["best"]["run_id"] == "200-0"
    assert index["best"]["format_version"] == 2
    best = json.loads(store["model_checkpoints/name/best.json"])
    assert best["model_state"]["format_version"] == 2

    # Within the new format the metric gate is back in force.
    _write_bundle(tmp_path / "c", name="name", run_id="300-0", metric=0.1, format_version=2)
    hc.push_checkpoint_bundle(tmp_path / "c", bucket="ns/b")
    index = json.loads(store["model_checkpoints/name/index.json"])
    assert index["best"]["run_id"] == "200-0"


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
