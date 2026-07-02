from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable


def test_generate_solvable_reports_difficulty() -> None:
    instance = generate_solvable(rank=2, depth=6, seed=3)
    assert 0 <= instance.difficulty <= 6
    assert instance.presentation.provenance["difficulty"] == instance.difficulty


def test_generate_solvable_is_deterministic_and_reversible() -> None:
    first = generate_solvable(rank=2, depth=8, seed=11)
    second = generate_solvable(rank=2, depth=8, seed=11)
    # Same seed reproduces the same scrambled instance and reverse certificate.
    assert first.presentation.to_json() == second.presentation.to_json()
    assert first.reverse_moves == second.reverse_moves
    # The reverse path is exactly the recorded scramble undone move by move.
    pres = first.presentation
    for move in first.reverse_moves:
        pres = move.apply(pres)
    assert pres.content_hash == BalancedPresentation.standard(2).content_hash
