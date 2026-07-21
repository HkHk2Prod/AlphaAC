"""Enumerate the Akbulut-Kirby and Miller-Schupp benchmark presentations.

The catalog is defined by two bounds, both recorded in the file it writes:

``max_relator_length``
    No relator of an included presentation may be longer than this, measured
    *after* free reduction. It bounds both families' index -- AK(n) has a
    relator of length ``2n+1`` and MS(n, w) one of length ``2n+3``.

``max_w_length``
    A separate cap on the Miller-Schupp word ``w``. It is needed because
    ``max_relator_length`` alone does not bound the sweep usefully: freely
    reduced words with x-exponent sum zero grow roughly threefold per letter, so
    the ~47 letters a bound of 48 permits describe more words than could ever be
    enumerated. The relator bound still applies to ``w`` on top of this cap.

Entries are deduplicated by content hash -- free reduction collapses distinct
words onto the same presentation -- and ordered smallest-first, so a run that
exhausts its budget partway through has spent it on the easiest entries.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.candidates import akbulut_kirby, miller_schupp

# Generators are x = 1, y = 2, matching ac_zero.datasets.candidates.
_X = 1
_LETTERS = (1, -1, 2, -2)
DEFAULT_MAX_W_LENGTH = 7


def catalog_name(max_relator_length: int, max_w_length: int) -> str:
    """Stable name for the catalog these bounds define (used as its file name)."""
    return f"ak-ms-rel{max_relator_length}-w{max_w_length}"


def miller_schupp_words(max_length: int) -> Iterator[list[int]]:
    """Yield every freely reduced word over x^+-1, y^+-1 with x-exponent sum zero.

    Depth-first over the four letters, pruning two ways: a letter that would
    cancel its predecessor is never appended (free reduction), and a prefix whose
    x-exponent balance can no longer reach zero within the remaining letters is
    abandoned. The empty word is included -- MS(n, e) is a legitimate member.
    """
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    word: list[int] = []

    def walk(balance: int) -> Iterator[list[int]]:
        if balance == 0:
            yield list(word)
        if len(word) == max_length:
            return
        for letter in _LETTERS:
            if word and letter == -word[-1]:
                continue
            step = 1 if letter == _X else -1 if letter == -_X else 0
            word.append(letter)
            # Every remaining slot can shift the balance by at most one, so a
            # balance deeper than the slots left can never return to zero.
            if abs(balance + step) <= max_length - len(word):
                yield from walk(balance + step)
            word.pop()

    yield from walk(0)


def _longest_relator(presentation: BalancedPresentation) -> int:
    return max(len(relator) for relator in presentation.relators)


def benchmark_entries(
    *, max_relator_length: int, max_w_length: int = DEFAULT_MAX_W_LENGTH
) -> list[BalancedPresentation]:
    """Every AK and MS presentation inside the two bounds, smallest-first.

    Akbulut-Kirby members come first so that when free reduction makes an MS
    instance coincide with one, the entry keeps the AK identity.
    """
    if max_relator_length < 1:
        raise ValueError("max_relator_length must be positive")
    if max_w_length < 0:
        raise ValueError("max_w_length must be non-negative")

    entries: list[BalancedPresentation] = []
    seen: set[str] = set()

    def keep(presentation: BalancedPresentation) -> None:
        if _longest_relator(presentation) > max_relator_length:
            return
        if presentation.content_hash in seen:
            return
        seen.add(presentation.content_hash)
        entries.append(presentation)

    # AK(n)'s longest relator is max(2n+1, 6), so it grows without bound in n.
    for n in range(1, (max_relator_length - 1) // 2 + 1):
        keep(akbulut_kirby(n))

    words = list(miller_schupp_words(min(max_w_length, max_relator_length)))
    for n in range(1, (max_relator_length - 3) // 2 + 1):
        for word in words:
            keep(miller_schupp(n, word))

    entries.sort(key=lambda p: (p.total_length, str(p.presentation_id)))
    return entries


@dataclass(frozen=True, slots=True)
class BenchmarkCatalog:
    """A named, bounded set of benchmark presentations."""

    name: str
    max_relator_length: int
    max_w_length: int
    entries: tuple[BalancedPresentation, ...]

    @classmethod
    def build(
        cls, *, max_relator_length: int, max_w_length: int = DEFAULT_MAX_W_LENGTH
    ) -> BenchmarkCatalog:
        return cls(
            name=catalog_name(max_relator_length, max_w_length),
            max_relator_length=max_relator_length,
            max_w_length=max_w_length,
            entries=tuple(
                benchmark_entries(max_relator_length=max_relator_length, max_w_length=max_w_length)
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "benchmark_catalog",
            "name": self.name,
            "rank": 2,
            "max_relator_length": self.max_relator_length,
            "max_w_length": self.max_w_length,
            "count": len(self.entries),
            "families": _family_counts(self.entries),
            "entries": [presentation.to_json() for presentation in self.entries],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BenchmarkCatalog:
        return cls(
            name=str(data["name"]),
            max_relator_length=int(data["max_relator_length"]),
            max_w_length=int(data["max_w_length"]),
            entries=tuple(
                BalancedPresentation.from_json(entry) for entry in data.get("entries", [])
            ),
        )

    def write(self, path: str | Path) -> Path:
        """Write the catalog as JSON, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")
        return target

    @classmethod
    def read(cls, path: str | Path) -> BenchmarkCatalog:
        """Read a catalog written by :meth:`write`."""
        return cls.from_json(json.loads(Path(path).read_text()))


def _family_counts(entries: Sequence[BalancedPresentation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        family = str(entry.provenance.get("family", "unknown"))
        counts[family] = counts.get(family, 0) + 1
    return counts
