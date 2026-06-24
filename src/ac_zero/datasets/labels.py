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
