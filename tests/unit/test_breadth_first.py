from pathlib import Path

from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.search.breadth_first import BreadthFirstConfig, BreadthFirstSearch


def _env(presentation, max_moves=8, capacity=48):
    config = ACEnvironmentConfig(max_moves=max_moves)
    return ACEnvironment(presentation, config, StateEncoder(capacity))


def test_bfs_finds_a_verified_shortest_certificate(tmp_path: Path) -> None:
    instance = generate_solvable(rank=2, depth=3, seed=0)
    cert = tmp_path / "cert.json"
    result = BreadthFirstSearch().solve(
        instance.presentation,
        env_template=_env(instance.presentation, max_moves=6, capacity=32),
        certificate_path=cert,
    )
    assert result.success
    assert CertificateVerifier().verify_path(cert).ok
    # BFS is shortest-first, so it never exceeds the reverse-scramble solution
    assert len(result.path) <= len(instance.reverse_moves)


def test_bfs_proves_optimality_on_a_short_instance() -> None:
    instance = generate_solvable(rank=2, depth=2, seed=3)
    env = _env(instance.presentation)
    result = BreadthFirstSearch().solve(instance.presentation, env_template=env)
    assert result.success
    assert result.metrics["proved_optimal"] == 1.0
    # no strictly shorter sequence can exist once optimality is proven
    assert len(result.path) <= 2


def test_bfs_solves_already_trivial_presentation_in_zero_moves() -> None:
    from ac_zero.algebra.presentation import BalancedPresentation

    standard = BalancedPresentation.standard(2)
    result = BreadthFirstSearch().solve(standard, env_template=_env(standard))
    assert result.success
    assert result.path == ()
    assert result.metrics["proved_optimal"] == 1.0


def test_bfs_reports_failure_within_a_tight_budget() -> None:
    instance = generate_solvable(rank=2, depth=12, seed=1)
    result = BreadthFirstSearch(BreadthFirstConfig(max_expansions=5, max_generated=20)).solve(
        instance.presentation, env_template=_env(instance.presentation, max_moves=12)
    )
    assert not result.success
    assert result.metrics["proved_optimal"] == 0.0
