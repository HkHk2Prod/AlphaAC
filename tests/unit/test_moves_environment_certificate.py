from pathlib import Path

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.certificates.certificate import build_certificate
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import ConjugateRelatorMove, InvertRelatorMove, MultiplyRelatorsMove


def test_catalog_size_and_order_rank_two() -> None:
    catalog = ActionCatalog(2)
    assert len(catalog) == 12
    assert catalog.move(0) == MultiplyRelatorsMove(0, 1)
    assert catalog.action_id(InvertRelatorMove(0)) == 2


def test_catalog_moves_built_once_and_cached() -> None:
    catalog = ActionCatalog(3)
    # The property returns the same tuple object on every access (no rebuild).
    assert catalog.moves is catalog.moves


def test_catalog_action_id_round_trips_for_every_move() -> None:
    catalog = ActionCatalog(3)
    for expected_id, move in enumerate(catalog.moves):
        assert catalog.action_id(move) == expected_id
        assert catalog.move(expected_id) == move


def test_catalog_action_id_rejects_unknown_move() -> None:
    catalog = ActionCatalog(2)
    with pytest.raises(ValueError, match="not in catalog"):
        catalog.action_id(InvertRelatorMove(5))


def test_catalog_equality_ignores_internal_lookup() -> None:
    assert ActionCatalog(2) == ActionCatalog(2)
    assert hash(ActionCatalog(2)) == hash(ActionCatalog(2))


def test_hand_checked_moves_do_not_mutate_original() -> None:
    pres = BalancedPresentation.from_letters(2, [[1], [2]])
    nxt = MultiplyRelatorsMove(0, 1).apply(pres)
    assert pres.relators[0].letters == (1,)
    assert nxt.relators[0].letters == (1, 2)
    assert InvertRelatorMove(0).apply(nxt).relators[0].letters == (-2, -1)
    assert ConjugateRelatorMove(1, 1).apply(pres).relators[1].letters == (1, 2, -1)


def test_reward_telescopes_to_maximum_reduction() -> None:
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=2, mask_noops=False))
    rewards = []
    _, reward, _, _, _ = env.step(ActionCatalog(2).action_id(InvertRelatorMove(1)))
    rewards.append(reward)
    _, reward, _, _, info = env.step(ActionCatalog(2).action_id(MultiplyRelatorsMove(0, 1)))
    rewards.append(reward)
    assert sum(rewards) == pres.total_length - info["best_total_length"]


def test_state_key_distinguishes_horizon_and_best_length() -> None:
    pres = BalancedPresentation.standard(2)
    a = ACEnvironment(pres, ACEnvironmentConfig(max_moves=2)).state
    b = ACEnvironment(pres, ACEnvironmentConfig(max_moves=3)).state
    assert a.key != b.key


def test_generated_certificate_verifies_and_corruption_fails(tmp_path: Path) -> None:
    instance = generate_solvable(2, depth=2, seed=4)
    cert = build_certificate(
        instance.presentation,
        instance.reverse_moves,
        goal_mode="exact_standard",
        experiment_id="t",
        seed=4,
    )
    path = tmp_path / "cert.json"
    cert.write(path)
    verifier = CertificateVerifier()
    assert verifier.verify_path(path).ok
    data = path.read_text().replace('"type": "AC', '"type": "BAD', 1)
    bad = tmp_path / "bad.json"
    bad.write_text(data)
    assert not verifier.verify_path(bad).ok
