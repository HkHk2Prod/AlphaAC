from __future__ import annotations

from dataclasses import dataclass, field

from ac_zero.moves.primitive import (
    ConjugateRelatorMove,
    InvertRelatorMove,
    MultiplyRelatorsMove,
    PrimitiveMove,
)


def _build_catalog(rank: int) -> tuple[PrimitiveMove, ...]:
    """Enumerate all strict primitive moves in stable action-ID order."""
    result: list[PrimitiveMove] = []
    for target in range(rank):
        for source in range(rank):
            if target != source:
                result.append(MultiplyRelatorsMove(target, source))
    for target in range(rank):
        result.append(InvertRelatorMove(target))
    for target in range(rank):
        for gen in range(1, rank + 1):
            result.append(ConjugateRelatorMove(target, gen))
            result.append(ConjugateRelatorMove(target, -gen))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    """Deterministic finite strict primitive action catalog.

    The move tuple and its reverse `move -> action_id` lookup are built once at
    construction, so `move` and `action_id` are O(1) and never re-allocate the
    move objects on the search/self-play hot paths.
    """

    rank: int
    version: str = "strict-ac-v1"
    _moves: tuple[PrimitiveMove, ...] = field(init=False, repr=False, compare=False)
    _index: dict[PrimitiveMove, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        moves = _build_catalog(self.rank)
        object.__setattr__(self, "_moves", moves)
        object.__setattr__(self, "_index", {move: idx for idx, move in enumerate(moves)})

    @property
    def moves(self) -> tuple[PrimitiveMove, ...]:
        """Return the deterministic strict catalog in stable action-ID order."""
        return self._moves

    def __len__(self) -> int:
        return len(self._moves)

    def move(self, action_id: int) -> PrimitiveMove:
        """Look up a primitive move by stable action ID."""
        return self._moves[action_id]

    def action_id(self, move: PrimitiveMove) -> int:
        """Look up the stable action ID for a primitive move."""
        try:
            return self._index[move]
        except KeyError:
            raise ValueError(f"move not in catalog: {move!r}") from None
