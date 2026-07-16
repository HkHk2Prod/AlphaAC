from __future__ import annotations

import numpy as np

from ac_zero.datasets.digest_index import (
    DIGEST_BYTES,
    UNKNOWN,
    digest_array,
    prefixes,
    sorted_lookup,
    values_for,
)


def _digests(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(count, DIGEST_BYTES), dtype=np.uint8)


def test_digest_array_views_a_flat_buffer_as_rows() -> None:
    raw = bytearray(range(DIGEST_BYTES)) + bytearray(range(DIGEST_BYTES))
    assert digest_array(raw).shape == (2, DIGEST_BYTES)


def test_prefixes_are_stored_in_native_byte_order() -> None:
    """The column must be natively ordered, or `searchsorted` degrades to O(n).

    NumPy has no fast path for a byte-swapped array: a ``>u8`` column turns every
    binary search into a full byteswapping pass over the whole ball. This is the
    regression that made navigation self-play ~4000x slower per distance lookup.
    """
    column = prefixes(_digests(64))
    assert column.dtype == np.dtype(np.uint64)
    assert column.dtype.byteorder in ("=", "|") or column.dtype.isnative


def test_prefixes_preserve_lexicographic_byte_order() -> None:
    """Native storage must not change the ordering the binary search relies on."""
    digests = _digests(512)
    column = prefixes(digests)
    lexicographic = sorted(range(len(digests)), key=lambda i: bytes(digests[i, :8]))
    assert np.array_equal(np.argsort(column, kind="stable"), np.array(lexicographic))


def test_values_for_resolves_each_present_digest() -> None:
    digests = _digests(256)
    values = np.arange(256, dtype=np.int32)
    found = values_for(*sorted_lookup(digests, values), digests)
    assert np.array_equal(found, values)


def test_values_for_reports_unknown_for_absent_digests() -> None:
    known = _digests(32, seed=1)
    absent = _digests(4, seed=2)
    lookup = sorted_lookup(known, np.arange(32, dtype=np.int32))
    assert np.array_equal(values_for(*lookup, absent), np.full(4, UNKNOWN, dtype=np.int32))


def test_values_for_settles_prefix_collisions_on_the_full_digest() -> None:
    """Two digests sharing an 8-byte prefix must still resolve to their own values."""
    digests = np.zeros((2, DIGEST_BYTES), dtype=np.uint8)
    digests[0, DIGEST_BYTES - 1] = 1  # identical prefix, differing tail
    digests[1, DIGEST_BYTES - 1] = 2
    values = np.array([10, 20], dtype=np.int32)
    assert np.array_equal(values_for(*sorted_lookup(digests, values), digests), values)
