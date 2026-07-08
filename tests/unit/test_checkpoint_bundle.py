"""Tests for the on-disk checkpoint bundle writer."""

from __future__ import annotations

import json
from pathlib import Path

from ac_zero.training.checkpoint_bundle import CheckpointBundle


def _payload(metric: float) -> dict:
    return {"schema_version": "v1", "checkpoint_metric": metric, "model_state": {"w": [1.0]}}


def test_latest_always_written_best_tracks_improvement(tmp_path: Path) -> None:
    bundle = CheckpointBundle(tmp_path / "b")

    assert bundle.save_checkpoint(_payload(0.2), metric=0.2) is True  # first is best
    assert bundle.save_checkpoint(_payload(0.1), metric=0.1) is False  # worse: latest only
    assert bundle.save_checkpoint(_payload(0.5), metric=0.5) is True  # improves best

    best = json.loads((tmp_path / "b" / "best.json").read_text())
    latest = json.loads((tmp_path / "b" / "latest.json").read_text())
    assert best["checkpoint_metric"] == 0.5
    assert latest["checkpoint_metric"] == 0.5  # last save was also the best
    assert bundle.best_metric == 0.5


def test_none_metric_updates_latest_but_not_best(tmp_path: Path) -> None:
    bundle = CheckpointBundle(tmp_path / "b")
    assert bundle.save_checkpoint(_payload(0.0), metric=None) is False
    assert (tmp_path / "b" / "latest.json").exists()
    assert not (tmp_path / "b" / "best.json").exists()
    assert bundle.best_metric is None


def test_metrics_and_meta_round_trip(tmp_path: Path) -> None:
    bundle = CheckpointBundle(tmp_path / "b")
    rows = [{"optimizer_step": 1, "total_loss": 0.5}, {"optimizer_step": 2, "total_loss": 0.3}]
    bundle.save_metrics(rows)
    bundle.save_meta({"run_id": "42-0", "iteration": 2})

    lines = (tmp_path / "b" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["optimizer_step"] for line in lines] == [1, 2]
    assert json.loads((tmp_path / "b" / "meta.json").read_text())["run_id"] == "42-0"


def test_has_checkpoint_reflects_state(tmp_path: Path) -> None:
    bundle = CheckpointBundle(tmp_path / "b")
    assert bundle.has_checkpoint() is False
    bundle.save_checkpoint(_payload(0.1), metric=0.1)
    assert bundle.has_checkpoint() is True
