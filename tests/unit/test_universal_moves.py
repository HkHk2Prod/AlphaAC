"""Tests for the universal invertible move catalog and named move sets."""

from __future__ import annotations

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import (
    ConcatRelatorMove,
    ConjugateRelatorMove,
    InvertRelatorMove,
    MultiplyRelatorsMove,
    inverse_move,
    move_from_json,
)
from ac_zero.moves.universal import MOVE_SET_NAMES, UniversalCatalog, move_set


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_universal_catalog_size(rank: int) -> None:
    assert len(UniversalCatalog(rank)) == 6 * rank * rank - 3 * rank


def test_inverse_id_is_an_involution_and_closed() -> None:
    catalog = UniversalCatalog(3)
    for move_id in range(len(catalog)):
        inverse = catalog.inverse_id(move_id)
        assert 0 <= inverse < len(catalog)
        assert catalog.inverse_id(inverse) == move_id


def test_apply_then_inverse_is_identity() -> None:
    catalog = UniversalCatalog(2)
    # A non-trivial start so concat/conjugation genuinely change the relators.
    start = BalancedPresentation.from_letters(2, [[1, 2], [2, -1]])
    for move_id, move in enumerate(catalog.moves):
        moved = move.apply(start)
        restored = catalog.move(catalog.inverse_id(move_id)).apply(moved)
        assert restored.content_hash == start.content_hash


def test_inverse_move_pairs_the_strict_multiply_with_its_concat() -> None:
    forward = MultiplyRelatorsMove(0, 1)
    inverse = inverse_move(forward)
    assert inverse == ConcatRelatorMove(0, 1, "right", True)
    assert inverse_move(inverse) == forward
    assert inverse_move(InvertRelatorMove(1)) == InvertRelatorMove(1)
    assert inverse_move(ConjugateRelatorMove(0, 2)) == ConjugateRelatorMove(0, -2)


def test_concat_move_json_round_trip() -> None:
    move = ConcatRelatorMove(1, 0, "left", True)
    assert move_from_json(move.to_json()) == move


def test_strict_ac_is_a_subset_matching_the_action_catalog() -> None:
    catalog = UniversalCatalog(2)
    strict = move_set("strict-ac", catalog)
    assert strict.code_name == "strict-ac"
    assert len(strict.ids) == 3 * 2 * 2  # 3n^2
    universal_moves = {catalog.move(i) for i in strict.ids}
    assert universal_moves == set(ActionCatalog(2).moves)


def test_move_set_names_include_universal_and_strict() -> None:
    assert "universal" in MOVE_SET_NAMES
    assert "strict-ac" in MOVE_SET_NAMES


def test_unknown_move_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown move set"):
        move_set("nope", UniversalCatalog(2))
