import json
from pathlib import Path

from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.cli import main


def test_cli_greedy_pipeline_writes_verified_solution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    dataset_path = tmp_path / "data/generated/greedy_rl.json"
    assert (
        main(["dataset", "grow", "--output", str(dataset_path), "--target", "20", "--workers", "1"])
        == 0
    )
    assert main(["solve", "--presentation", str(dataset_path), "--agent", "greedy"]) == 0

    cert_path = tmp_path / "runs/solve/certificates/solution.json"
    assert cert_path.exists()
    assert CertificateVerifier().verify_path(cert_path).ok

    assert main(["benchmark", "--config", "configs/experiments/greedy_rl.yaml"]) == 0
    benchmark_path = tmp_path / "runs/smoke/evaluation/benchmark.json"
    rows = json.loads(benchmark_path.read_text())
    greedy_row = next(row for row in rows if row["agent"] == "greedy")
    assert greedy_row["verified_success"] is True
    assert greedy_row["certificate"]


def test_cli_grow_expands_the_database_across_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "data/generated/grown.json"

    assert main(["dataset", "grow", "--output", str(dataset_path), "--target", "15"]) == 0
    first = len(json.loads(dataset_path.read_text())["instances"])

    # A second run resumes from the same file and only ever grows the database.
    assert main(["dataset", "grow", "--input", str(dataset_path), "--target", "15"]) == 0
    data = json.loads(dataset_path.read_text())
    assert data["schema_version"] == "aczero-dataset-v3"
    assert len(data["instances"]) > first
    assert main(["dataset", "validate", "--input", str(dataset_path)]) == 0
