from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.algebra.word import FreeGroupWord


def test_free_reduction_idempotent_and_inverse_identity() -> None:
    word = FreeGroupWord((1, 2, -2, -1, 1), rank=2)
    assert word.letters == (1,)
    assert word.reduced() == word
    assert word.concat(word.inverse()).letters == ()


def test_mul_operator_and_concat() -> None:
    u = FreeGroupWord((1, 2), rank=2)
    v = FreeGroupWord((-2, 1), rank=2)
    assert (u * v).letters == (1, 1)
    assert u.concat(v) == u * v


def test_inversion_reverses_order() -> None:
    u = FreeGroupWord((1, 2), rank=2)
    v = FreeGroupWord((-1,), rank=2)
    assert u.concat(v).inverse() == v.inverse().concat(u.inverse())


def test_format_and_json_roundtrip() -> None:
    word = FreeGroupWord((1, -2), rank=2)
    assert word.format() == "x1 x2^-1"
    assert FreeGroupWord.from_json(word.to_json(), rank=2) == word


def test_presentation_length_and_hash_stability() -> None:
    pres = BalancedPresentation.from_letters(2, [[1, 2, -2], [2]])
    assert pres.total_length == 2
    assert pres.content_hash == BalancedPresentation.from_json(pres.to_json()).content_hash
