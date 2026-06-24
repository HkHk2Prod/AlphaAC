from __future__ import annotations

from dataclasses import dataclass

from ac_zero.algebra.presentation import BalancedPresentation


@dataclass(frozen=True, slots=True)
class ACSearchState:
    """Immutable Markov state for one AC search episode.

    The key intentionally includes best-so-far length and remaining horizon:
    the same presentation can have a different value when these quantities
    differ.
    """

    presentation: BalancedPresentation
    initial_length: int
    best_length: int
    moves_used: int
    moves_remaining: int
    catalog_version: str
    last_action: int | None = None
    safety_truncated: bool = False

    @property
    def key(self) -> tuple[object, ...]:
        """Return a value-relevant transposition/cache key."""
        return (
            self.presentation.content_hash,
            self.initial_length,
            self.best_length,
            self.moves_used,
            self.moves_remaining,
            self.catalog_version,
            self.last_action,
            self.safety_truncated,
        )
