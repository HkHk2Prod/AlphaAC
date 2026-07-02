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


PrimitiveMove = MultiplyRelatorsMove | InvertRelatorMove | ConjugateRelatorMove


def move_from_json(data: dict[str, Any]) -> PrimitiveMove:
    """Deserialize one strict primitive move from certificate JSON."""
    match data.get("type"):
        case "AC1":
            return MultiplyRelatorsMove(int(data["target"]), int(data["source"]))
        case "AC2":
            return InvertRelatorMove(int(data["target"]))
        case "AC3":
            return ConjugateRelatorMove(int(data["target"]), int(data["generator"]))
        case typ:
            raise MoveError(f"unknown primitive move type {typ!r}")


def inverse_primitive_sequence(move: PrimitiveMove) -> tuple[PrimitiveMove, ...]:
    """Expand the inverse of one primitive move into strict primitive moves.

    Inversion (AC2) is its own inverse, and conjugation (AC3) inverts by negating
    the generator, so both undo in a single move. A relator multiply (AC1) has no
    single-move inverse in the catalog, so it is undone by inverting the source,
    multiplying, and inverting the source back -- three strict primitives.
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
    raise TypeError(f"unsupported primitive move {move!r}")


def _check_index(presentation: BalancedPresentation, index: int) -> None:
    if not 0 <= index < presentation.rank:
        raise MoveError("relator index out of range")
