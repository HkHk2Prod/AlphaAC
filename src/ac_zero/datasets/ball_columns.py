"""Compact, growable storage for a closest-first ball: its columns and dedup index.

The ball is built one breadth-first shell at a time and, at every checkpoint, is
still written out as the ``groups``/``annotations`` JSON documents every consumer
reads. Holding each group as a :class:`BalancedPresentation` cost ~1 KB of RAM --
enough that a rank-2 ball hit a machine's memory before it hit its group target.

:class:`BallColumns` keeps the same per-group facts as flat arrays instead: the
relator letters in one ``int8`` buffer, their bounds as ``int64`` offsets, the
distance as a ``uint8``, the co-optimal forward moves as a bitmask, and the
32-byte content digest raw rather than as a 64-char hex string. That is ~80 bytes
a group, an order of magnitude smaller, and it rebuilds a presentation only when
one is actually asked for.

:class:`DigestIndex` is the growth-time dedup map, content digest -> group index.
It cannot be evicted -- the ball is dominated by its outermost shells, so dedup
stays whole-ball -- so a Python ``dict`` of 100M hex-string keys would itself
outweigh the columns. It is an open-addressing table over the digest's 128-bit
prefix: two ``uint64`` key columns and one ``int64`` value column, ~34 bytes a
group. A 128-bit prefix makes a false match vanishingly unlikely (~1e-23 at 1e8
groups), so no full-digest verification is needed.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence

import numpy as np

# SHA-256 content digest, stored raw. The index keys on its first 16 bytes.
DIGEST_BYTES = 32
_KEY_BYTES = 16


class DigestIndex:
    """Open-addressing map from a 32-byte digest to a group index, keyed on 128 bits.

    Values store ``index + 1`` so a zeroed slot reads as empty. The table doubles
    and rehashes past a 0.7 load factor; linear probing resolves the rare slot
    clash. Digests come from SHA-256, so the top ``uint64`` is already uniform and
    is used directly as the home slot.
    """

    __slots__ = ("_count", "_hi", "_lo", "_mask", "_val")

    def __init__(self, capacity: int = 1 << 12) -> None:
        self._alloc(capacity)
        self._count = 0

    def _alloc(self, capacity: int) -> None:
        self._mask = capacity - 1
        self._hi = np.zeros(capacity, dtype=np.uint64)
        self._lo = np.zeros(capacity, dtype=np.uint64)
        self._val = np.zeros(capacity, dtype=np.int64)

    @staticmethod
    def _key(digest: bytes) -> tuple[int, int]:
        return (
            int.from_bytes(digest[0:8], "big"),
            int.from_bytes(digest[8:_KEY_BYTES], "big"),
        )

    def get(self, digest: bytes) -> int | None:
        """Return the group index stored for ``digest``, or ``None`` if absent."""
        hi, lo = self._key(digest)
        hi_c, lo_c, val_c, mask = self._hi, self._lo, self._val, self._mask
        slot = hi & mask
        while True:
            value = int(val_c[slot])
            if value == 0:
                return None
            if int(hi_c[slot]) == hi and int(lo_c[slot]) == lo:
                return value - 1
            slot = (slot + 1) & mask

    def insert(self, digest: bytes, index: int) -> None:
        """Record ``digest -> index``; the caller has checked the digest is new."""
        if (self._count + 1) * 10 >= (self._mask + 1) * 7:
            self._grow()
        self._place(*self._key(digest), index)
        self._count += 1

    def _place(self, hi: int, lo: int, index: int) -> None:
        val_c, mask = self._val, self._mask
        slot = hi & mask
        while int(val_c[slot]) != 0:
            slot = (slot + 1) & mask
        self._hi[slot] = hi
        self._lo[slot] = lo
        self._val[slot] = index + 1

    def _grow(self) -> None:
        old_hi, old_lo, old_val = self._hi, self._lo, self._val
        self._alloc((self._mask + 1) * 2)
        for slot in np.flatnonzero(old_val):
            self._place(int(old_hi[slot]), int(old_lo[slot]), int(old_val[slot]) - 1)

    def __len__(self) -> int:
        return self._count


class BallColumns:
    """The ball's per-group facts as flat, append-only columns.

    ``letters`` concatenates every relator; ``offsets`` holds one boundary per
    relator (plus a leading 0), so group ``i``'s relators span
    ``offsets[i * rank : (i + 1) * rank + 1]`` -- the layout the instance store
    already memory-maps. ``distance``, ``moves`` (a bitmask over the move set's
    forward moves), and the raw ``digests`` are one entry per group.
    """

    __slots__ = ("_digests", "_distance", "_letters", "_max_length", "_moves", "_offsets", "rank")

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._letters = array("b")
        self._offsets = array("q", [0])
        self._distance = array("B")
        self._moves = array("I")  # 4 bytes on every target platform; holds the move-set width
        self._digests = bytearray()
        self._max_length = 0

    def __len__(self) -> int:
        return len(self._distance)

    def append(
        self, relators: Iterable[Sequence[int]], digest: bytes, distance: int, moves_mask: int
    ) -> int:
        """Append one group's columns and return its index."""
        index = len(self._distance)
        total = 0
        for relator in relators:
            self._letters.extend(relator)
            self._offsets.append(len(self._letters))
            total += len(relator)
        self._distance.append(distance)
        self._moves.append(moves_mask)
        self._digests += digest
        if total > self._max_length:
            self._max_length = total
        return index

    def distance_at(self, index: int) -> int:
        return self._distance[index]

    def or_move(self, index: int, bit: int) -> None:
        self._moves[index] |= bit

    def moves_at(self, index: int) -> int:
        return self._moves[index]

    def digest_at(self, index: int) -> bytes:
        return bytes(self._digests[index * DIGEST_BYTES : (index + 1) * DIGEST_BYTES])

    def relators_at(self, index: int) -> list[list[int]]:
        rank = self.rank
        bounds = self._offsets[index * rank : index * rank + rank + 1]
        return [self._letters[bounds[j] : bounds[j + 1]].tolist() for j in range(rank)]

    def total_length_at(self, index: int) -> int:
        rank = self.rank
        return self._offsets[index * rank + rank] - self._offsets[index * rank]

    def max_length(self) -> int:
        return self._max_length
