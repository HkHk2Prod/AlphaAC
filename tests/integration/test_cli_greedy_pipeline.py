import json
from pathlib import Path

from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.cli import main


def test_cli_greedy_pipeline_writes_verified_solution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    dataset_path = tmp_path / "data/generated/greedy_rl.json"
    assert main(["dataset", "generate", "--output", str(dataset_path)]) == 0
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


def test_dataset_generate_uses_configured_difficulty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "rank: 2",
                "seed: 11",
                "dataset:",
                "  count: 5",
                "  depth: 7",
                "  min_total_length: 8",
                "  min_relator_length: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "data/generated/configured.json"

    assert (
        main(["dataset", "generate", "--config", str(config_path), "--output", str(dataset_path)])
        == 0
    )

    data = json.loads(dataset_path.read_text())
    assert data["rank"] == 2
    assert data["provenance"]["seed"] == 11
    assert data["provenance"]["depth"] == 7
    assert data["provenance"]["min_total_length"] == 8
    assert data["provenance"]["min_relator_length"] == 2
    assert len(data["instances"]) == 5
    assert {instance["provenance"]["depth"] for instance in data["instances"]} == {7}
    for instance in data["instances"]:
        relator_lengths = [len(relator) for relator in instance["relators"]]
        assert sum(relator_lengths) >= 8
        assert min(relator_lengths) >= 2
