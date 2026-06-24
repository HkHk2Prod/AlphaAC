from __future__ import annotations

from dataclasses import dataclass

from ac_zero.moves.primitive import (
    ConjugateRelatorMove,
    InvertRelatorMove,
    MultiplyRelatorsMove,
    PrimitiveMove,
)


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    """Deterministic finite strict primitive action catalog."""

    rank: int
    version: str = "strict-ac-v1"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")

    @property
    def moves(self) -> tuple[PrimitiveMove, ...]:
        """Return the deterministic strict catalog in stable action-ID order."""
        result: list[PrimitiveMove] = []
        for target in range(self.rank):
            for source in range(self.rank):
                if target != source:
                    result.append(MultiplyRelatorsMove(target, source))
        for target in range(self.rank):
            result.append(InvertRelatorMove(target))
        for target in range(self.rank):
            for gen in range(1, self.rank + 1):
                result.append(ConjugateRelatorMove(target, gen))
                result.append(ConjugateRelatorMove(target, -gen))
        return tuple(result)

    def __len__(self) -> int:
        return len(self.moves)

    def move(self, action_id: int) -> PrimitiveMove:
        """Look up a primitive move by stable action ID."""
        return self.moves[action_id]

    def action_id(self, move: PrimitiveMove) -> int:
        """Look up the stable action ID for a primitive move."""
        return self.moves.index(move)
