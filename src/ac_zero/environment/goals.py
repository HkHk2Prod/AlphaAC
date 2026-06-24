from __future__ import annotations

from ac_zero.algebra.presentation import BalancedPresentation


def exact_standard_goal(presentation: BalancedPresentation) -> bool:
    """Return whether relators are exactly `(x1, ..., xn)` in order."""
    standard = BalancedPresentation.standard(presentation.rank)
    return presentation.rank == standard.rank and presentation.relators == standard.relators


def signed_permuted_basis_goal(presentation: BalancedPresentation) -> bool:
    """Return whether relators are distinct signed generators.

    This predicate is intentionally separate from `exact_standard_goal` because
    cleanup to the exact ordered tuple must be represented by explicit moves.
    """

    seen: set[int] = set()
    for relator in presentation.relators:
        if len(relator) != 1:
            return False
        gen = abs(relator.letters[0])
        if gen in seen:
            return False
        seen.add(gen)
    return seen == set(range(1, presentation.rank + 1))
