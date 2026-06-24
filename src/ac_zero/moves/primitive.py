from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ac_zero.algebra.presentation import BalancedPresentation


class MoveError(ValueError):
    """Raised when an AC move is invalid."""


class ACMove(Protocol):
    """Protocol for primitive Andrews-Curtis moves."""

    def apply(self, presentation: BalancedPresentation) -> BalancedPresentation: ...

    def to_json(self) -> dict[str, Any]: ...

    def notation(self) -> str: ...


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

    def notation(self) -> str:
        """Return a human-readable mathematical notation string."""
        return f"AC1(r{self.target + 1} <- r{self.target + 1} r{self.source + 1})"


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

    def notation(self) -> str:
        """Return a human-readable mathematical notation string."""
        return f"AC2(r{self.target + 1} <- r{self.target + 1}^-1)"


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

    def notation(self) -> str:
        """Return a human-readable mathematical notation string."""
        return (
            f"AC3(r{self.target + 1} <- x{self.generator} r{self.target + 1} x{self.generator}^-1)"
        )


PrimitiveMove = MultiplyRelatorsMove | InvertRelatorMove | ConjugateRelatorMove


def move_from_json(data: dict[str, Any]) -> PrimitiveMove:
    """Deserialize one strict primitive move from certificate JSON."""
    typ = data.get("type")
    if typ == "AC1":
        return MultiplyRelatorsMove(int(data["target"]), int(data["source"]))
    if typ == "AC2":
        return InvertRelatorMove(int(data["target"]))
    if typ == "AC3":
        return ConjugateRelatorMove(int(data["target"]), int(data["generator"]))
    raise MoveError(f"unknown primitive move type {typ!r}")


def _check_index(presentation: BalancedPresentation, index: int) -> None:
    if not 0 <= index < presentation.rank:
        raise MoveError("relator index out of range")
