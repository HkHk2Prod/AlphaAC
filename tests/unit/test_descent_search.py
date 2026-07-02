from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.search.descent import DescentConfig, descent_distance


def _search(rank, relators, config=None):
    presentation = BalancedPresentation.from_letters(rank, relators)
    return descent_distance(presentation, ActionCatalog(rank), config or DescentConfig())


def test_single_move_descent_is_distance_one() -> None:
    # <x y^-1, y>: one AC1 (r0 <- r0 r1) cancels to <x, y>, dropping length 3 -> 2.
    result = _search(2, [[1, -2], [2]])
    assert result.distance == 1
    assert result.proven is True


def test_plateau_requires_two_moves() -> None:
    # <x y, y>: no single move shortens it; invert r1 then multiply reaches <x, y^-1>.
    result = _search(2, [[1, 2], [2]])
    assert result.distance == 2
    assert result.proven is True


def test_local_minimum_has_no_proven_descent() -> None:
    # The rank-1 standard presentation <x | x> sits at the global length minimum:
    # its whole reachable region is tiny and holds nothing shorter.
    result = _search(1, [[1]])
    assert result.distance is None
    assert result.proven is True


def test_expansion_budget_leaves_answer_unproven() -> None:
    # With no expansions allowed the region is unexplored, so absence is not proven.
    result = _search(2, [[1, 2], [2]], DescentConfig(max_expansions=0))
    assert result.distance is None
    assert result.proven is False


def test_length_cap_prune_reports_upper_bound_not_proven() -> None:
    # A cap that forces pruning of longer branches yields the shortest descent found
    # (an upper bound) but cannot certify it as minimal.
    result = _search(2, [[1, 2], [2]], DescentConfig(total_length_cap=3))
    assert result.distance == 2
    assert result.proven is False


def test_depth_bound_truncates_without_proof() -> None:
    result = _search(2, [[1], [2]], DescentConfig(max_depth=1, max_expansions=1))
    assert result.distance is None
    assert result.proven is False
