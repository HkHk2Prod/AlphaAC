from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ac_zero.algebra.presentation import BalancedPresentation


class MoveError(ValueError):
    """Raised when an AC move is invalid."""


@dataclass(frozen=True, slots=True)
class MultiplyRelatorsMove:
    """Primitive `AC1`: replace one relator by its product with another."""

    target: int
    source: int
    kind: Literal["AC1"] = "AC1"

    def apply(self, presentation: BalancedPresentation) -> BalancedPresentation:
        """Apply `r_target <- red(r_target r_source)`."""
        _check_index(presentation, self.target)
        _check_index(presentation, self.source)
        if self.target == self.source:
            raise MoveError("AC1 requires distinct target and source")
        new_rel = presentation.relators[self.target].concat(presentation.relators[self.source])
        return presentation.replace_relator(self.target, new_rel)

    def to_json(self) -> dict[str, Any]:
        """Serialize this primitive move to certificate JSON."""
        return {"type": self.kind, "target": self.target, "source": self.source}


@dataclass(frozen=True, slots=True)
class InvertRelatorMove:
    """Primitive `AC2`: replace one relator by its inverse."""

    target: int
    kind: Literal["AC2"] = "AC2"

    def apply(self, presentation: BalancedPresentation) -> BalancedPresentation:
        """Apply `r_target <- red(r_target^-1)`."""
        _check_index(presentation, self.target)
        return presentation.replace_relator(
            self.target, presentation.relators[self.target].inverse()
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize this primitive move to certificate JSON."""
        return {"type": self.kind, "target": self.target}


@dataclass(frozen=True, slots=True)
class ConjugateRelatorMove:
    """Primitive `AC3`: conjugate one relator by a signed generator."""

    target: int
    generator: int
    kind: Literal["AC3"] = "AC3"

    def apply(self, presentation: BalancedPresentation) -> BalancedPresentation:
        """Apply `r_target <- red(g r_target g^-1)`."""
        _check_index(presentation, self.target)
        if self.generator == 0 or abs(self.generator) > presentation.rank:
            raise MoveError("invalid conjugating generator")
        rel = presentation.relators[self.target].conjugate_by_letter(self.generator)
        return presentation.replace_relator(self.target, rel)

    def to_json(self) -> dict[str, Any]:
        """Serialize this primitive move to certificate JSON."""
        return {"type": self.kind, "target": self.target, "generator": self.generator}


@dataclass(frozen=True, slots=True)
class ConcatRelatorMove:
    """Universal concat: replace one relator by its product with another relator.

    Generalizes the strict AC1 `r_target <- r_target r_source` to the three other
    invertible variants: right/left multiplication by the source relator or its
    inverse. The right, non-inverted case is exactly `MultiplyRelatorsMove`, which
    the strict catalog keeps for byte-stable certificates; this class carries the
    other three so that every concat has its inverse concat in the universal set.
    """

    target: int
    source: int
    side: Literal["left", "right"]
    invert_source: bool
    kind: Literal["CONCAT"] = "CONCAT"

    def apply(self, presentation: BalancedPresentation) -> BalancedPresentation:
        """Apply `r_target <- red(r_source^s r_target)` or `red(r_target r_source^s)`."""
        _check_index(presentation, self.target)
        _check_index(presentation, self.source)
        if self.target == self.source:
            raise MoveError("concat requires distinct target and source")
        target = presentation.relators[self.target]
        source = presentation.relators[self.source]
        if self.invert_source:
            source = source.inverse()
        new_rel = source.concat(target) if self.side == "left" else target.concat(source)
        return presentation.replace_relator(self.target, new_rel)

    def to_json(self) -> dict[str, Any]:
        """Serialize this concat move to JSON."""
        return {
            "type": self.kind,
            "target": self.target,
            "source": self.source,
            "side": self.side,
            "invert_source": self.invert_source,
        }


PrimitiveMove = MultiplyRelatorsMove | InvertRelatorMove | ConjugateRelatorMove | ConcatRelatorMove


def move_from_json(data: dict[str, Any]) -> PrimitiveMove:
    """Deserialize one primitive move from certificate/dataset JSON."""
    match data.get("type"):
        case "AC1":
            return MultiplyRelatorsMove(int(data["target"]), int(data["source"]))
        case "AC2":
            return InvertRelatorMove(int(data["target"]))
        case "AC3":
            return ConjugateRelatorMove(int(data["target"]), int(data["generator"]))
        case "CONCAT":
            return ConcatRelatorMove(
                int(data["target"]),
                int(data["source"]),
                data["side"],
                bool(data["invert_source"]),
            )
        case typ:
            raise MoveError(f"unknown primitive move type {typ!r}")


def inverse_move(move: PrimitiveMove) -> PrimitiveMove:
    """Return the single universal move that undoes `move`.

    Every universal move is invertible in one move: inversion (AC2) is its own
    inverse, conjugation (AC3) inverts by negating the generator, and a concat by
    a relator inverts by flipping whether the source is inverted (on the same
    side). The strict right multiply `MultiplyRelatorsMove` and its partner
    `ConcatRelatorMove(right, invert_source=True)` are each other's inverse.
    """
    if isinstance(move, InvertRelatorMove):
        return move
    if isinstance(move, ConjugateRelatorMove):
        return ConjugateRelatorMove(move.target, -move.generator)
    if isinstance(move, MultiplyRelatorsMove):
        return ConcatRelatorMove(move.target, move.source, "right", True)
    if isinstance(move, ConcatRelatorMove):
        if move.side == "right" and move.invert_source:
            return MultiplyRelatorsMove(move.target, move.source)
        return ConcatRelatorMove(move.target, move.source, move.side, not move.invert_source)
    raise TypeError(f"unsupported primitive move {move!r}")


def inverse_primitive_sequence(move: PrimitiveMove) -> tuple[PrimitiveMove, ...]:
    """Expand the inverse of one move into *strict* primitive moves (AC1/AC2/AC3).

    Used to build reverse certificates, which must replay through the strict
    catalog the verifier accepts. Inversion (AC2) is its own inverse and
    conjugation (AC3) inverts by negating the generator, so both undo in a single
    strict move. A strict relator multiply (AC1) has no single strict inverse, so
    it is undone by inverting the source, multiplying, and inverting back -- three
    strict primitives. (For the single-move universal inverse used by the graph
    and annotation, see :func:`inverse_move`.)
    """
    if isinstance(move, InvertRelatorMove):
        return (move,)
    if isinstance(move, ConjugateRelatorMove):
        return (ConjugateRelatorMove(move.target, -move.generator),)
    if isinstance(move, MultiplyRelatorsMove):
        return (
            InvertRelatorMove(move.source),
            MultiplyRelatorsMove(move.target, move.source),
            InvertRelatorMove(move.source),
        )
    raise TypeError(f"unsupported strict primitive move {move!r}")


def _check_index(presentation: BalancedPresentation, index: int) -> None:
    if not 0 <= index < presentation.rank:
        raise MoveError("relator index out of range")
