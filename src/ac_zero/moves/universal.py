from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ac_zero.moves.primitive import (
    ConcatRelatorMove,
    ConjugateRelatorMove,
    InvertRelatorMove,
    MultiplyRelatorsMove,
    PrimitiveMove,
    inverse_move,
)


def _build_universal(rank: int) -> tuple[PrimitiveMove, ...]:
    """Enumerate every universal invertible move in stable move-ID order.

    Concat moves come first (both sides, source and inverted source), then the
    self-inverse relator inversions, then conjugation by each signed generator.
    The right, non-inverted concat is emitted as the strict `MultiplyRelatorsMove`
    so the strict-AC catalog is a genuine subset with identical move objects.
    """
    result: list[PrimitiveMove] = []
    for target in range(rank):
        for source in range(rank):
            if target == source:
                continue
            for side in ("right", "left"):
                for invert in (False, True):
                    if side == "right" and not invert:
                        result.append(MultiplyRelatorsMove(target, source))
                    else:
                        result.append(ConcatRelatorMove(target, source, side, invert))
    for target in range(rank):
        result.append(InvertRelatorMove(target))
    for target in range(rank):
        for gen in range(1, rank + 1):
            result.append(ConjugateRelatorMove(target, gen))
            result.append(ConjugateRelatorMove(target, -gen))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class UniversalCatalog:
    """The universal invertible move catalog: `6n^2 - 3n` moves, inversion-closed.

    Every move's inverse is another move in the catalog, so `inverse_id` maps a
    move ID to the ID of its single-move inverse without leaving the catalog. The
    move tuple and its `move -> id` lookup are built once at construction, keeping
    lookups O(1) on the expansion and annotation hot paths.
    """

    rank: int
    version: str = "universal-v1"
    _moves: tuple[PrimitiveMove, ...] = field(init=False, repr=False, compare=False)
    _index: dict[PrimitiveMove, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        moves = _build_universal(self.rank)
        object.__setattr__(self, "_moves", moves)
        object.__setattr__(self, "_index", {move: idx for idx, move in enumerate(moves)})

    @property
    def moves(self) -> tuple[PrimitiveMove, ...]:
        """Return the universal catalog in stable move-ID order."""
        return self._moves

    def __len__(self) -> int:
        return len(self._moves)

    def move(self, move_id: int) -> PrimitiveMove:
        """Look up a universal move by stable move ID."""
        return self._moves[move_id]

    def move_id(self, move: PrimitiveMove) -> int:
        """Look up the stable move ID for a universal move."""
        try:
            return self._index[move]
        except KeyError:
            raise ValueError(f"move not in universal catalog: {move!r}") from None

    def inverse_id(self, move_id: int) -> int:
        """Return the move ID of the single-move inverse of `move_id`."""
        return self.move_id(inverse_move(self._moves[move_id]))


@dataclass(frozen=True, slots=True)
class MoveSet:
    """A named subset of the universal catalog, resolved for a specific rank.

    `code_name` labels the annotation file the set produces. `ids` are the
    universal move IDs in the set; `inverse_ids` maps them through the catalog's
    inversion (used to walk from the origin under the inverse move set).
    """

    code_name: str
    ids: frozenset[int]

    def inverse_ids(self, catalog: UniversalCatalog) -> frozenset[int]:
        """Return the IDs of the inverses of this set's moves (another subset)."""
        return frozenset(catalog.inverse_id(i) for i in self.ids)


MovePredicate = Callable[[PrimitiveMove], bool]


def _is_universal(_move: PrimitiveMove) -> bool:
    return True


def _is_strict_ac(move: PrimitiveMove) -> bool:
    """The classic strict catalog: right multiply, invert, conjugate."""
    return isinstance(move, MultiplyRelatorsMove | InvertRelatorMove | ConjugateRelatorMove)


_REGISTRY: dict[str, MovePredicate] = {
    "universal": _is_universal,
    "strict-ac": _is_strict_ac,
}

MOVE_SET_NAMES = tuple(_REGISTRY)


def move_set(code_name: str, catalog: UniversalCatalog) -> MoveSet:
    """Resolve a named move set to concrete universal move IDs for a rank."""
    try:
        predicate = _REGISTRY[code_name]
    except KeyError:
        raise ValueError(f"unknown move set {code_name!r}; choose from {MOVE_SET_NAMES}") from None
    ids = frozenset(i for i, move in enumerate(catalog.moves) if predicate(move))
    return MoveSet(code_name, ids)


@dataclass(frozen=True, slots=True)
class MoveSetCatalog:
    """A named move set's moves, reindexed to a dense local action-ID space.

    Mirrors `ActionCatalog`'s interface (`moves`, `move`, `__len__`, `version`) so
    it can stand in as an environment's action catalog: local ids `0..len-1` index
    into `moves` and are independent of `UniversalCatalog`'s own move IDs, which
    only have meaning against `UniversalCatalog` itself.
    """

    code_name: str
    moves: tuple[PrimitiveMove, ...]

    @property
    def version(self) -> str:
        return f"{self.code_name}-v1"

    def __len__(self) -> int:
        return len(self.moves)

    def move(self, action_id: int) -> PrimitiveMove:
        """Look up a move by its local action ID."""
        return self.moves[action_id]


def moveset_catalog(code_name: str, rank: int) -> MoveSetCatalog:
    """Resolve a named move set to a dense action catalog ready for self-play.

    `code_name = "strict-ac"` reproduces `ActionCatalog(rank)`'s moves in the same
    order (both enumerate the same underlying moves); `"universal"` reproduces the
    full `UniversalCatalog(rank)`. Other move sets fall in between, always ordered
    by their underlying universal move ID.
    """
    universal = UniversalCatalog(rank)
    selected = move_set(code_name, universal)
    moves = tuple(universal.moves[i] for i in sorted(selected.ids))
    return MoveSetCatalog(code_name, moves)
