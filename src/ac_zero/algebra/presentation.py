from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ac_zero.algebra.word import FreeGroupWord


class PresentationError(ValueError):
    """Raised when a balanced presentation is malformed."""


@dataclass(frozen=True, slots=True)
class BalancedPresentation:
    """Balanced presentation <x_1,...,x_n | r_1,...,r_n>."""

    rank: int
    relators: tuple[FreeGroupWord, ...]
    generator_names: tuple[str, ...] = field(default_factory=tuple)
    presentation_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    # Lazily memoized content hash. Excluded from init/eq/repr: it is a pure
    # function of the immutable content above, so caching it never changes the
    # value or identity of a presentation, only how often it is recomputed. The
    # grow graph hashes millions of neighbours, so this stays off the hot path.
    _content_hash: str | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize generator names and relators while preserving immutability."""
        if self.rank <= 0:
            raise PresentationError("rank must be positive")
        names = self.generator_names or tuple(f"x{i}" for i in range(1, self.rank + 1))
        if len(names) != self.rank:
            raise PresentationError("generator name count must equal rank")
        if len(self.relators) != self.rank:
            raise PresentationError("balanced presentations require rank relators")
        reduced = tuple(FreeGroupWord(rel.letters, self.rank) for rel in self.relators)
        object.__setattr__(self, "generator_names", tuple(names))
        object.__setattr__(self, "relators", reduced)

    @classmethod
    def from_letters(
        cls,
        rank: int,
        relators: Iterable[Iterable[int]],
        *,
        generator_names: Iterable[str] | None = None,
        presentation_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> BalancedPresentation:
        """Build a presentation from raw signed-integer relator iterables."""
        return cls(
            rank=rank,
            relators=tuple(FreeGroupWord(r, rank) for r in relators),
            generator_names=tuple(generator_names or (f"x{i}" for i in range(1, rank + 1))),
            presentation_id=presentation_id,
            provenance=dict(provenance or {}),
        )

    @property
    def total_length(self) -> int:
        """Total length of the freely reduced ordered relator tuple."""
        return sum(len(r) for r in self.relators)

    @property
    def content_hash(self) -> str:
        """Deterministic SHA-256 hash of the mathematical presentation content."""
        if self._content_hash is None:
            payload = {
                "rank": self.rank,
                "generator_names": self.generator_names,
                "relators": [r.to_json() for r in self.relators],
            }
            blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(blob).hexdigest()
            object.__setattr__(self, "_content_hash", digest)
            return digest
        return self._content_hash

    def replace_relator(self, index: int, relator: FreeGroupWord) -> BalancedPresentation:
        """Return a new presentation with one freely reduced relator replaced."""
        if not 0 <= index < self.rank:
            raise PresentationError("relator index out of range")
        if relator.rank != self.rank:
            raise PresentationError("relator rank mismatch")
        rels = list(self.relators)
        rels[index] = relator.reduced()
        return BalancedPresentation(
            self.rank,
            tuple(rels),
            self.generator_names,
            self.presentation_id,
            self.provenance,
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize to the canonical dataset/certificate presentation object."""
        return {
            "rank": self.rank,
            "generator_names": list(self.generator_names),
            "relators": [r.to_json() for r in self.relators],
            "human_relators": [r.format(self.generator_names) for r in self.relators],
            "presentation_id": self.presentation_id,
            "provenance": self.provenance,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BalancedPresentation:
        """Deserialize a canonical presentation JSON object."""
        return cls.from_letters(
            int(data["rank"]),
            data["relators"],
            generator_names=data.get("generator_names"),
            presentation_id=data.get("presentation_id"),
            provenance=data.get("provenance") or {},
        )

    @classmethod
    def standard(cls, rank: int) -> BalancedPresentation:
        """Return the exact standard trivial presentation of the given rank."""
        return cls.from_letters(rank, ([i] for i in range(1, rank + 1)), presentation_id="standard")

    def format(self) -> str:
        """Return a compact human-readable presentation string."""
        rels = ", ".join(r.format(self.generator_names) for r in self.relators)
        gens = ", ".join(self.generator_names)
        return f"<{gens} | {rels}>"
