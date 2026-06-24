from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrivializationLabel:
    """Known Andrews-Curtis trivialization status for a presentation entry.

    `ac_trivial` is True when the presentation is known to be AC-trivial, False
    when known not to be, and None when the question is open or unstudied.
    `minimal_known_operations` is the fewest strict primitive AC operations of any
    trivialization currently known, or None when none is known. `optimal` records
    whether `minimal_known_operations` has been proven minimal (None when there is
    no number to qualify).
    """

    ac_trivial: bool | None
    minimal_known_operations: int | None
    optimal: bool | None

    def to_json(self) -> dict[str, bool | int | None]:
        """Serialize the label as flat per-entry dataset fields."""
        return {
            "ac_trivial": self.ac_trivial,
            "minimal_known_operations": self.minimal_known_operations,
            "optimal": self.optimal,
        }


UNKNOWN = TrivializationLabel(ac_trivial=None, minimal_known_operations=None, optimal=None)


def known_solution(operations: int, *, optimal: bool = False) -> TrivializationLabel:
    """Label a presentation known to be AC-trivial via a trivialization of `operations`."""
    if operations < 0:
        raise ValueError("operations must be non-negative")
    return TrivializationLabel(
        ac_trivial=True, minimal_known_operations=operations, optimal=optimal
    )


def known_trivial() -> TrivializationLabel:
    """Label a presentation known to be AC-trivial with no recorded operation count."""
    return TrivializationLabel(ac_trivial=True, minimal_known_operations=None, optimal=None)


def merge_labels(old: TrivializationLabel, new: TrivializationLabel) -> TrivializationLabel:
    """Combine two labels for the same presentation, keeping only improvements.

    The merge is monotonic and never regresses known information: a known
    triviality result is never demoted to unknown, and a shorter known
    trivialization is never replaced by a longer one. This guards a dataset
    against an older, worse solution overwriting a better one.
    """
    ac_trivial = _merge_known(old.ac_trivial, new.ac_trivial)
    values = [
        label.minimal_known_operations
        for label in (old, new)
        if label.minimal_known_operations is not None
    ]
    if not values:
        return TrivializationLabel(ac_trivial, None, None)
    minimal = min(values)
    # The shortest solution is proven optimal only if some source proved it at
    # exactly that length; a longer optimal proof does not certify a new minimum.
    optimal = any(
        label.optimal is True and label.minimal_known_operations == minimal for label in (old, new)
    )
    return TrivializationLabel(ac_trivial=True, minimal_known_operations=minimal, optimal=optimal)


def _merge_known(old: bool | None, new: bool | None) -> bool | None:
    """Combine two known-status flags, preferring a definite True then False."""
    if old is True or new is True:
        return True
    if old is False or new is False:
        return False
    return None
