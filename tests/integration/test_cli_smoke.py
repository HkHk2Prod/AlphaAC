from pathlib import Path

from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.cli import main
from ac_zero.training.checkpointing.checkpointing import CheckpointManager


def test_cli_smoke_writes_verified_certificate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["smoke-test"]) == 0
    cert_path = tmp_path / "runs/smoke/certificates/example.json"
    assert cert_path.exists()
    assert CertificateVerifier().verify_path(cert_path).ok
    assert (tmp_path / "runs/smoke/checkpoints/latest.json").exists()
    assert (tmp_path / "runs/smoke/manifest.json").exists()
    assert (tmp_path / "runs/smoke/artifacts/smoke_summary.json").exists()
    assert (tmp_path / "runs/smoke/artifacts/final_graphs.txt").exists()
    assert (tmp_path / "runs/smoke/artifacts/live_graphs.txt").exists()
    assert (tmp_path / "runs/smoke/logs/progress.log").exists()
    assert (tmp_path / "runs/smoke/logs/training_events.jsonl").exists()
    assert (tmp_path / "runs/smoke/evaluation/benchmark.json").exists()
    checkpoint = CheckpointManager(tmp_path / "runs/smoke/checkpoints").load_json("latest")
    assert checkpoint["schema_version"] == "aczero-smoke-checkpoint-v1"
    assert checkpoint["optimizer_state"]["step"] == 1
    assert checkpoint["replay_size"] == 1
    final_graphs = (tmp_path / "runs/smoke/artifacts/final_graphs.txt").read_text()
    assert "final training graphs" in final_graphs
    assert "loss" in final_graphs
    progress = (tmp_path / "runs/smoke/logs/progress.log").read_text()
    assert "smoke training completed" in progress
