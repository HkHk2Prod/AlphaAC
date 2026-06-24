import pytest

from ac_zero.datasets.labels import UNKNOWN, known_solution, known_trivial


def test_unknown_label_is_all_none() -> None:
    assert UNKNOWN.to_json() == {
        "ac_trivial": None,
        "minimal_known_operations": None,
        "optimal": None,
    }


def test_known_solution_records_operations_and_optimality() -> None:
    label = known_solution(7, optimal=True)
    assert label.ac_trivial is True
    assert label.minimal_known_operations == 7
    assert label.optimal is True
    assert known_solution(4).optimal is False


def test_known_trivial_has_no_operation_count() -> None:
    label = known_trivial()
    assert label.ac_trivial is True
    assert label.minimal_known_operations is None
    assert label.optimal is None


def test_known_solution_rejects_negative_operations() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        known_solution(-1)
