from pathlib import Path

from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.search.breadth_first import BreadthFirstSearch
from ac_zero.search.iterative_deepening import (
    IterativeDeepeningConfig,
    IterativeDeepeningSearch,
)


def _env(presentation, max_moves=6, capacity=32):
    config = ACEnvironmentConfig(max_moves=max_moves)
    return ACEnvironment(presentation, config, StateEncoder(capacity))


def test_iddfs_matches_bfs_length_and_verifies(tmp_path: Path) -> None:
    instance = generate_solvable(rank=2, depth=2, seed=3)
    env = _env(instance.presentation)
    cert = tmp_path / "cert.json"
    iddfs = IterativeDeepeningSearch().solve(
        instance.presentation, env_template=env, certificate_path=cert
    )
    bfs_env = _env(instance.presentation)
    bfs = BreadthFirstSearch().solve(instance.presentation, env_template=bfs_env)
    assert iddfs.success
    assert CertificateVerifier().verify_path(cert).ok
    # both are shortest-complete on this shallow instance
    assert len(iddfs.path) == len(bfs.path)
    assert iddfs.metrics["proved_optimal"] == 1.0


def test_iddfs_reports_failure_within_a_tiny_budget() -> None:
    instance = generate_solvable(rank=2, depth=8, seed=1)
    result = IterativeDeepeningSearch(IterativeDeepeningConfig(max_generated=50)).solve(
        instance.presentation, env_template=_env(instance.presentation, max_moves=8)
    )
    assert not result.success
    assert result.metrics["proved_optimal"] == 0.0
