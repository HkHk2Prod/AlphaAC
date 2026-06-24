from pathlib import Path

from ac_zero.agents.greedy import (
    GreedyBestFirstSearch,
    GreedyLengthAgent,
    GreedySolver,
)
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import InvertRelatorMove


def test_greedy_action_ranking_prioritizes_length_preserving_goal() -> None:
    pres = BalancedPresentation.from_letters(2, [[1], [-2]])
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=2))
    action = GreedyLengthAgent(env).select_action(env.legal_action_mask())
    assert action == ActionCatalog(2).action_id(InvertRelatorMove(1))


def test_greedy_solver_writes_verified_certificate(tmp_path: Path) -> None:
    instance = generate_solvable(2, depth=1, seed=0)
    env = ACEnvironment(instance.presentation, ACEnvironmentConfig(max_moves=4))
    cert_path = tmp_path / "greedy.json"
    result = GreedySolver().solve(env, certificate_path=cert_path, seed=11)
    assert result.success
    assert result.termination_reason == "goal"
    assert result.certificate_path == str(cert_path)
    assert CertificateVerifier().verify_path(cert_path).ok


def test_greedy_solver_stops_at_local_minimum_without_goal() -> None:
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=4))
    result = GreedySolver().solve(env)
    assert not result.success
    assert result.termination_reason == "local_minimum"
    assert result.path == ()


def test_greedy_best_first_finds_one_move_inverse_cleanup(tmp_path: Path) -> None:
    pres = BalancedPresentation.from_letters(2, [[1], [-2]])
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=2))
    cert_path = tmp_path / "best-first.json"
    result = GreedyBestFirstSearch().solve(pres, env_template=env, certificate_path=cert_path)
    assert result.success
    assert result.path == (ActionCatalog(2).action_id(InvertRelatorMove(1)),)
    assert result.peak_frontier_size >= 1
    assert CertificateVerifier().verify_path(cert_path).ok
