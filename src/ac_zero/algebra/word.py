from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


class WordError(ValueError):
    """Raised when a free-group word is malformed."""


def _reduce_letters(letters: Iterable[int]) -> tuple[int, ...]:
    """Freely reduce adjacent inverse pairs using a deterministic stack pass."""
    stack: list[int] = []
    for letter in letters:
        if stack and stack[-1] == -letter:
            stack.pop()
        else:
            stack.append(letter)
    return tuple(stack)


@dataclass(frozen=True, slots=True)
class FreeGroupWord:
    """Immutable word in a free group, represented by signed generator indices."""

    letters: tuple[int, ...]
    rank: int

    def __init__(self, letters: Iterable[int] = (), rank: int = 0, *, reduce: bool = True) -> None:
        """Validate and optionally freely reduce a signed-integer word."""
        object.__setattr__(self, "rank", int(rank))
        raw = tuple(int(x) for x in letters)
        if self.rank < 0:
            raise WordError("rank must be non-negative")
        for letter in raw:
            if letter == 0:
                raise WordError("0 is not a valid free-group letter")
            if self.rank and abs(letter) > self.rank:
                raise WordError(f"letter {letter} exceeds rank {self.rank}")
        object.__setattr__(self, "letters", _reduce_letters(raw) if reduce else raw)

    def __iter__(self) -> Iterator[int]:
        return iter(self.letters)

    def __getitem__(self, index: int) -> int:
        return self.letters[index]

    def __len__(self) -> int:
        return len(self.letters)

    def __bool__(self) -> bool:
        return bool(self.letters)

    def reduced(self) -> FreeGroupWord:
        """Return the freely reduced representative of this word."""
        return FreeGroupWord(self.letters, self.rank)

    def inverse(self) -> FreeGroupWord:
        """Return the inverse word with reversed order and flipped signs."""
        return FreeGroupWord((-x for x in reversed(self.letters)), self.rank)

    def concat(self, other: FreeGroupWord) -> FreeGroupWord:
        """Concatenate two same-rank words and freely reduce the product."""
        self._check_rank(other)
        return FreeGroupWord((*self.letters, *other.letters), self.rank)

    def left_multiply(self, other: FreeGroupWord) -> FreeGroupWord:
        """Return `other * self` with free reduction."""
        return other.concat(self)

    def right_multiply(self, other: FreeGroupWord) -> FreeGroupWord:
        """Return `self * other` with free reduction."""
        return self.concat(other)

    def conjugate_by_letter(self, generator: int) -> FreeGroupWord:
        """Return `g self g^-1` for one signed generator `g`."""
        if generator == 0 or abs(generator) > self.rank:
            raise WordError("conjugating generator is outside the rank")
        return FreeGroupWord((generator, *self.letters, -generator), self.rank)

    def to_json(self) -> list[int]:
        """Serialize to canonical signed-integer JSON form."""
        return list(self.letters)

    @classmethod
    def from_json(cls, data: list[int], rank: int) -> FreeGroupWord:
        """Parse canonical signed-integer JSON form."""
        return cls(data, rank)

    def format(self, generator_names: tuple[str, ...] | None = None) -> str:
        """Format the word using documented tokens such as `x1 x2^-1`."""
        if not self.letters:
            return "1"
        names = generator_names or tuple(f"x{i}" for i in range(1, self.rank + 1))
        parts = []
        for letter in self.letters:
            name = names[abs(letter) - 1]
            parts.append(name if letter > 0 else f"{name}^-1")
        return " ".join(parts)

    @classmethod
    def parse(
        cls, text: str, rank: int, generator_names: tuple[str, ...] | None = None
    ) -> FreeGroupWord:
        """Parse whitespace-separated generator tokens into a free-group word."""
        text = text.strip()
        if text in {"", "1"}:
            return cls((), rank)
        names = generator_names or tuple(f"x{i}" for i in range(1, rank + 1))
        mapping: dict[str, int] = {}
        for idx, name in enumerate(names, start=1):
            mapping[name] = idx
            mapping[f"{name}^-1"] = -idx
            mapping[f"{name}^-"] = -idx
            mapping[f"{name}-1"] = -idx
        try:
            return cls((mapping[token] for token in text.split()), rank)
        except KeyError as exc:
            raise WordError(f"unknown token {exc.args[0]!r}") from exc

    def _check_rank(self, other: FreeGroupWord) -> None:
        if self.rank != other.rank:
            raise WordError("cannot combine words with different ranks")


IDENTITY = FreeGroupWord(())
