"""Look up a value by content hash across millions of groups, without a Python dict.

The group, annotation, and split files all key their entries by a 32-byte content
hash, and joining them means resolving millions of those keys. A ``dict`` of hex
strings costs hundreds of bytes per entry on the heap; these columns cost 32. The
digests are sorted by their leading 8 bytes, so a whole query array resolves in one
vectorized binary search and the full 32 bytes settle the rare prefix collision.

Read big-endian: integer order over the 8-byte prefix is then identical to
lexicographic byte order, so the sort a binary search needs is the natural one.
The prefixes are *stored* in native byte order, though -- see :func:`prefixes`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DIGEST_BYTES = 32
# Stands in for a missing value so a joined column stays a plain int32.
UNKNOWN = -1


def digest_array(digests: bytearray | bytes) -> NDArray[np.uint8]:
    """View a flat run of concatenated digests as an ``(n, 32)`` byte matrix.

    The buffer is viewed, not copied: ``bytes(digests)`` on a multi-million-group
    dataset duplicates a hundred megabytes of digests for no reason. The caller must
    have finished appending -- a ``bytearray`` cannot grow while a view is out.
    """
    return np.frombuffer(memoryview(digests), dtype=np.uint8).reshape(-1, DIGEST_BYTES)


def prefixes(digests: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Read each digest's leading 8 bytes as a big-endian integer, stored natively.

    The big-endian *interpretation* is what makes integer order match lexicographic
    byte order, but the result is cast to native ``uint64`` rather than left as a
    ``>u8`` view. Only the byte order of the storage changes; every value, and so
    the sort order, is identical.

    That cast is not cosmetic. NumPy has no fast path for a byte-swapped array, so
    `searchsorted` against a ``>u8`` column silently degrades from an O(log n)
    binary search into an O(n) pass that byteswaps the whole column on every call
    -- 228 ms per lookup on a 16M-group ball, against 3 us here.
    """
    return np.ascontiguousarray(digests[:, :8]).view(">u8").ravel().astype(np.uint64)


def sorted_lookup(
    digests: NDArray[np.uint8], values: NDArray[np.int32]
) -> tuple[NDArray[np.uint64], NDArray[np.uint8], NDArray[np.int32]]:
    """Prefix-sort `(digest, value)` pairs into binary-searchable lookup columns."""
    order = np.argsort(prefixes(digests), kind="stable")
    ordered = digests[order]
    return prefixes(ordered), ordered, values[order]


def values_for(
    sorted_prefixes: NDArray[np.uint64],
    sorted_digests: NDArray[np.uint8],
    sorted_values: NDArray[np.int32],
    queries: NDArray[np.uint8],
) -> NDArray[np.int32]:
    """Look up each query digest's value, returning `UNKNOWN` where it is absent.

    The first three arguments are the prefix-sorted lookup columns from
    :func:`sorted_lookup`. A vectorized binary search on the 64-bit prefix narrows
    each query to one slot, and the full 32 bytes are then compared -- so a prefix
    collision costs a short scan rather than a wrong answer.
    """
    query_prefixes = prefixes(queries)
    low = np.searchsorted(sorted_prefixes, query_prefixes, "left")
    high = np.searchsorted(sorted_prefixes, query_prefixes, "right")
    found = np.full(len(query_prefixes), UNKNOWN, dtype=np.int32)
    unique = np.flatnonzero(high - low == 1)
    if unique.size:
        matched = unique[(sorted_digests[low[unique]] == queries[unique]).all(axis=1)]
        found[matched] = sorted_values[low[matched]]
    for row in np.flatnonzero(high - low > 1):  # 64-bit prefix collision: scan the tie
        for slot in range(low[row], high[row]):
            if bool((sorted_digests[slot] == queries[row]).all()):
                found[row] = sorted_values[slot]
                break
    return found
