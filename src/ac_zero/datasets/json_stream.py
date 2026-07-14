"""Incremental reader for the one large array inside a dataset JSON document.

A grown group file reaches gigabytes, and ``json.loads`` must materialize the
whole document -- roughly six times the file size as Python objects -- before the
caller sees a single group. These helpers walk the top-level members with the
standard library's C scanner and yield the target array's elements one at a
time, so peak memory is one element rather than the entire file.

The reader makes no assumptions about formatting or key order: it works on the
pretty-printed output of :func:`ac_zero.datasets.io.atomic_write_json` and on
compact ``json.dumps`` output alike.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_WHITESPACE = " \t\n\r"
_CHUNK = 1 << 22


class JsonStreamError(ValueError):
    """Raised when a document does not have the expected top-level shape."""


class _Scanner:
    """A sliding text window over a JSON file, decoded one value at a time.

    Consumed text is dropped whenever the window refills, so the buffer stays
    bounded by the chunk size plus the largest single value in the document.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._chunk = _CHUNK
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._index = 0
        self._eof = False

    def _fill(self) -> bool:
        """Drop consumed text and append the next chunk; False once exhausted."""
        if self._eof:
            return False
        more = self._handle.read(self._chunk)
        if not more:
            self._eof = True
            return False
        self._buffer = self._buffer[self._index :] + more
        self._index = 0
        return True

    def peek(self) -> str:
        """Return the next non-whitespace character, or ``""`` at end of document."""
        while True:
            index = self._index
            while index < len(self._buffer) and self._buffer[index] in _WHITESPACE:
                index += 1
            self._index = index
            if index < len(self._buffer):
                return self._buffer[index]
            if not self._fill():
                return ""

    def expect(self, char: str) -> None:
        """Consume ``char``, or raise when the document holds something else."""
        found = self.peek()
        if found != char:
            raise JsonStreamError(f"expected {char!r}, found {found or 'end of document'!r}")
        self._index += 1

    def accept(self, char: str) -> bool:
        """Consume ``char`` when it is next, reporting whether it was there."""
        if self.peek() != char:
            return False
        self._index += 1
        return True

    def value(self) -> Any:
        """Decode the next complete JSON value, refilling until it is whole."""
        while True:
            if self.peek() == "":
                raise JsonStreamError("document ended mid-value")
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._index)
            except ValueError:
                if self._fill():
                    continue  # the value straddles the window; widen it and retry
                raise
            # A number touching the window's end only looks complete -- its
            # remaining digits may sit in the next chunk. Any other trailing
            # character proves the scanner stopped at a real value boundary.
            if end == len(self._buffer) and self._fill():
                continue
            self._index = end
            return value


def read_members_before(path: Path, key: str) -> dict[str, Any]:
    """Return the top-level members that precede the ``key`` array, without reading it.

    Documents are written with sorted keys, so a member named to sort before the
    large array (``expanded`` before ``groups``) can be read back for the price of
    the members ahead of it -- which is how a multi-gigabyte ball is resumed
    without materializing a single group.
    """
    members: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        scanner = _Scanner(handle)
        scanner.expect("{")
        if scanner.accept("}"):
            return members
        while True:
            name = scanner.value()
            scanner.expect(":")
            if name == key:
                return members
            members[str(name)] = scanner.value()
            if not scanner.accept(","):
                return members


def iter_json_array(path: Path, key: str) -> Iterator[Any]:
    """Yield each element of the top-level ``key`` array in the JSON object at ``path``.

    Members preceding ``key`` are decoded and discarded, so ``key`` may sit
    anywhere in the object -- but any member before it must fit in memory, which
    holds for every dataset document (the one large array is what we stream).
    """
    with path.open(encoding="utf-8") as handle:
        scanner = _Scanner(handle)
        scanner.expect("{")
        missing = JsonStreamError(f"{path}: no {key!r} array at the top level")
        if scanner.accept("}"):
            raise missing
        while True:
            name = scanner.value()
            scanner.expect(":")
            if name == key:
                break
            scanner.value()  # a member we do not need; decode it and drop it
            if not scanner.accept(","):
                raise missing
        scanner.expect("[")
        if scanner.accept("]"):
            return
        while True:
            yield scanner.value()
            if not scanner.accept(","):
                break
        scanner.expect("]")
