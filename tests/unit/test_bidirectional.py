from pathlib import Path

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.search.bidirectional import (
    BidirectionalConfig,
    BidirectionalSearch,
    _goal_states,
    _predecessor,
)


def _env(presentation, max_moves=8, cap=48, goal_mode="exact_standard"):
    config = ACEnvironmentConfig(max_moves=max_moves, total_length_cap=cap, goal_mode=goal_mode)
    return ACEnvironment(presentation, config)


def test_finds_a_verified_shortest_certificate(tmp_path: Path) -> None:
    instance = generate_solvable(rank=2, depth=3, seed=0)
    cert = tmp_path / "cert.json"
    result = BidirectionalSearch().solve(
        instance.presentation,
        env_template=_env(instance.presentation, max_moves=6, cap=32),
        certificate_path=cert,
    )
    assert result.success
    assert CertificateVerifier().verify_path(cert).ok
    # Meet-in-the-middle is shortest-first, never worse than the reverse scramble.
    assert len(result.path) <= len(instance.reverse_moves)


def test_matches_forward_bfs_optimal_length() -> None:
    from ac_zero.search.breadth_first import BreadthFirstSearch

    instance = generate_solvable(rank=2, depth=4, seed=7)
    env = _env(instance.presentation, max_moves=8, cap=48)
    forward = BreadthFirstSearch().solve(instance.presentation, env_template=env)
    both = BidirectionalSearch().solve(instance.presentation, env_template=env)
    assert both.success and forward.success
    assert len(both.path) == len(forward.path)


def test_proves_optimality_on_a_short_instance() -> None:
    instance = generate_solvable(rank=2, depth=2, seed=3)
    env = _env(instance.presentation)
    result = BidirectionalSearch().solve(instance.presentation, env_template=env)
    assert result.success
    assert result.metrics["proved_optimal"] == 1.0
    assert len(result.path) <= 2


def test_solves_already_trivial_presentation_in_zero_moves() -> None:
    standard = BalancedPresentation.standard(2)
    result = BidirectionalSearch().solve(standard, env_template=_env(standard))
    assert result.success
    assert result.path == ()
    assert result.metrics["proved_optimal"] == 1.0


def test_reports_failure_within_a_tight_budget() -> None:
    instance = generate_solvable(rank=2, depth=12, seed=1)
    result = BidirectionalSearch(BidirectionalConfig(max_expansions=3, max_generated=12)).solve(
        instance.presentation, env_template=_env(instance.presentation, max_moves=12)
    )
    assert not result.success
    assert result.metrics["proved_optimal"] == 0.0
    assert "budget" in result.termination_reason


def test_solves_signed_permuted_basis_goal(tmp_path: Path) -> None:
    instance = generate_solvable(rank=2, depth=3, seed=5)
    cert = tmp_path / "cert.json"
    result = BidirectionalSearch().solve(
        instance.presentation,
        env_template=_env(instance.presentation, max_moves=6, goal_mode="signed_permuted_basis"),
        certificate_path=cert,
    )
    assert result.success
    assert CertificateVerifier().verify_path(cert).ok


def test_predecessor_inverts_every_catalog_move() -> None:
    instance = generate_solvable(rank=2, depth=3, seed=2)
    pres = instance.presentation
    catalog = ActionCatalog(pres.rank)
    for move in catalog.moves:
        child = move.apply(pres)
        # The forward move re-derives the child from its computed predecessor.
        assert move.apply(_predecessor(child, move)).content_hash == child.content_hash


def test_goal_state_orbit_sizes() -> None:
    standard = BalancedPresentation.standard(3)
    assert len(_goal_states(standard, "exact_standard")) == 1
    # 2^n * n! signed permutations of the basis for rank 3.
    assert len(_goal_states(standard, "signed_permuted_basis")) == 8 * 6
