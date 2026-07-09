"""A compact, memory-mapped view of a grown group dataset's start states.

Self-play needs only three things from a ``*.groups.json``: the rank, each
group's relators, and -- for the curriculum filter and the potential reward --
its distance to the trivial group. The document carries far more. The universal
move adjacency alone is ~85% of the bytes, and parsing the whole thing costs
roughly six times the file size in Python objects: a 2 GB dataset becomes a
~13 GB peak, paid *once per worker process*, which is enough to exhaust a machine
before the first episode runs.

So the JSON is streamed once into a single binary sidecar of plain numpy arrays
and memory-mapped thereafter. The relators of every group in a multi-gigabyte
dataset compress to a few tens of megabytes of ``int8`` letters, and because the
mapping is read-only every worker in a self-play pool shares one copy through the
page cache instead of parsing its own. Presentations are rebuilt lazily, one per
episode, rather than materialized three million at a time.

The sidecar is a pure derivative of the JSON, fingerprinted on the size and mtime
of both source files, so it is rebuilt whenever they change and never has to be
managed by hand. :mod:`ac_zero.datasets.columnar` owns its on-disk container.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.columnar import ColumnFile, Columns
from ac_zero.datasets.columnar import write as write_columns
from ac_zero.datasets.json_stream import iter_json_array

SCHEMA_VERSION = "aczero-instances-v1"
_DIGEST_BYTES = 32
# Stands in for a missing annotation so the distance column stays a plain int32.
UNKNOWN = -1
# Relator letters are signed generator indices, so an int8 column holds any rank a
# balanced presentation could realistically use.
_MAX_RANK = 127


def sidecar_path(groups_path: Path) -> Path:
    """Return where the compact sidecar for ``groups_path`` lives."""
    return groups_path.with_suffix(groups_path.suffix + ".instances")


def _fingerprints(groups_path: Path, annotations_path: Path | None) -> dict[str, Any]:
    """Identify the source documents, so a sidecar built from older ones is rejected."""
    paths = {"groups": groups_path, "annotations": annotations_path}
    return {
        name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for name, path in paths.items()
        if path is not None
    }


def _current(header: dict[str, Any], sources: dict[str, Any]) -> bool:
    """Whether a mapped sidecar was built by this schema from exactly these sources."""
    return header.get("schema_version") == SCHEMA_VERSION and header.get("sources") == sources


def _prefixes(digests: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """View each digest's leading 8 bytes as a big-endian integer.

    Big-endian makes integer order identical to lexicographic byte order, so a
    sorted prefix column can be binary-searched to locate a digest's candidates.
    """
    return np.ascontiguousarray(digests[:, :8]).view(">u8").ravel()


def _sorted_lookup(
    digests: NDArray[np.uint8], distances: NDArray[np.int32]
) -> tuple[NDArray[np.uint64], NDArray[np.uint8], NDArray[np.int32]]:
    """Prefix-sort `(digest, distance)` pairs into binary-searchable lookup columns."""
    order = np.argsort(_prefixes(digests), kind="stable")
    ordered = digests[order]
    return _prefixes(ordered), ordered, distances[order]


def _distances_for(
    prefixes: NDArray[np.uint64],
    digests: NDArray[np.uint8],
    distances: NDArray[np.int32],
    queries: NDArray[np.uint8],
) -> NDArray[np.int32]:
    """Look up each query digest's distance, returning `UNKNOWN` where it is absent.

    The first three arguments are the prefix-sorted lookup columns from
    :func:`_sorted_lookup`. A vectorized binary search on the 64-bit prefix narrows
    each query to one slot, and the full 32 bytes are then compared -- so a prefix
    collision costs a short scan rather than a wrong distance.
    """
    query_prefixes = _prefixes(queries)
    low = np.searchsorted(prefixes, query_prefixes, "left")
    high = np.searchsorted(prefixes, query_prefixes, "right")
    found = np.full(len(query_prefixes), UNKNOWN, dtype=np.int32)
    unique = np.flatnonzero(high - low == 1)
    if unique.size:
        matched = unique[(digests[low[unique]] == queries[unique]).all(axis=1)]
        found[matched] = distances[low[matched]]
    for row in np.flatnonzero(high - low > 1):  # 64-bit prefix collision: scan the tie
        for slot in range(low[row], high[row]):
            if bool((digests[slot] == queries[row]).all()):
                found[row] = distances[slot]
                break
    return found


def _digest_array(digests: bytearray) -> NDArray[np.uint8]:
    return np.frombuffer(bytes(digests), dtype=np.uint8).reshape(-1, _DIGEST_BYTES)


def _read_annotations(path: Path) -> tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """Stream a `.annotations.json` into the `(digests, distances)` it resolves."""
    digests = bytearray()
    distances = array("q")
    for entry in iter_json_array(path, "annotations"):
        distance = entry.get("distance_to_origin")
        if not isinstance(distance, int):
            continue
        digests += bytes.fromhex(entry["hash"])
        distances.append(distance)
    return _digest_array(digests), np.asarray(distances, dtype=np.int32)


def _read_groups(path: Path) -> tuple[int, Columns, NDArray[np.uint8]]:
    """Stream a `.groups.json` into its rank, relator columns, and group digests.

    ``word_offsets`` bounds every relator inside the flat ``letters`` column, so
    group ``i``'s relators span ``word_offsets[i * rank : (i + 1) * rank + 1]``.

    The rank comes from the first group rather than the document's own ``rank``
    member: keys are written sorted, so that member trails the multi-gigabyte
    ``groups`` array and reaching it would mean buffering the very thing we stream.
    """
    letters = array("b")
    word_offsets = array("q", [0])
    digests = bytearray()
    rank = 0
    for entry in iter_json_array(path, "groups"):
        relators = entry["relators"]
        if not rank:
            rank = int(entry["rank"])
            if not 0 < rank <= _MAX_RANK:
                raise ValueError(f"{path}: rank {rank} is outside the supported range")
        if len(relators) != rank:
            raise ValueError(f"{path}: a group has {len(relators)} relators, expected {rank}")
        for relator in relators:
            letters.extend(relator)
            word_offsets.append(len(letters))
        digests += bytes.fromhex(entry["hash"])
    if not rank:
        raise ValueError(f"{path}: dataset has no groups")
    if len(letters) > np.iinfo(np.int32).max:
        raise ValueError(f"{path}: {len(letters)} relator letters overflow the offset column")
    columns: Columns = {
        "letters": np.frombuffer(letters, dtype=np.int8),
        "word_offsets": np.asarray(word_offsets, dtype=np.int32),
    }
    return rank, columns, _digest_array(digests)


def _lookup_columns(digests: NDArray[np.uint8], distances: NDArray[np.int32]) -> Columns:
    """Store the groups with a known distance as the sidecar's potential lookup."""
    known = np.flatnonzero(distances != UNKNOWN)
    prefixes, ordered, ordered_distances = _sorted_lookup(digests[known], distances[known])
    return {
        "hash_prefixes": prefixes,
        "hash_digests": ordered,
        "hash_distances": ordered_distances,
    }


def build(groups_path: Path, annotations_path: Path | None) -> None:
    """Stream the JSON dataset into its compact sidecar.

    The sidecar is assembled in a sibling temp file and moved into place, so a
    crash mid-build leaves no half-written mapping behind, and two processes
    racing to build simply overwrite each other with identical bytes.
    """
    rank, columns, digests = _read_groups(groups_path)
    if annotations_path is not None:
        lookup = _sorted_lookup(*_read_annotations(annotations_path))
        distances = _distances_for(*lookup, digests)
        columns["distances"] = distances
        columns |= _lookup_columns(digests, distances)
    header = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "count": len(digests),
        "sources": _fingerprints(groups_path, annotations_path),
    }
    write_columns(sidecar_path(groups_path), header, columns)


class InstancePotentials(Mapping[str, int]):
    """Maps a presentation's content hash to its distance to the trivial group.

    Backed by the sidecar's memory-mapped digest columns, so the environment can
    score potential-based shaping against millions of annotated groups without
    any process holding them on its heap.
    """

    def __init__(self, prefixes: Any, digests: Any, distances: Any) -> None:
        self._prefixes = prefixes
        self._digests = digests
        self._distances = distances

    def __getitem__(self, content_hash: str) -> int:
        try:
            digest = bytes.fromhex(content_hash)
        except ValueError:
            raise KeyError(content_hash) from None
        if len(digest) != _DIGEST_BYTES:
            raise KeyError(content_hash)
        query = np.frombuffer(digest, dtype=np.uint8).reshape(1, _DIGEST_BYTES)
        distance = int(_distances_for(self._prefixes, self._digests, self._distances, query)[0])
        if distance == UNKNOWN:
            raise KeyError(content_hash)
        return distance

    def __iter__(self) -> Iterator[str]:
        return (bytes(digest).hex() for digest in self._digests)

    def __len__(self) -> int:
        return len(self._distances)


class InstanceStore:
    """The memory-mapped relator and distance columns of a grown group dataset."""

    def __init__(self, mapped: ColumnFile) -> None:
        """Bind to an already-mapped sidecar container."""
        self.path = mapped.path
        self.rank = int(mapped.header["rank"])
        self.count = int(mapped.header["count"])
        # Held for its lifetime, not its contents: dropping the ColumnFile would
        # close the mapping the columns below are views into.
        self._mapped = mapped
        self._columns = mapped.columns

    @classmethod
    def open(cls, groups_path: Path, annotations_path: Path | None) -> InstanceStore:
        """Map the sidecar for these sources, (re)building it when absent or stale."""
        path = sidecar_path(groups_path)
        sources = _fingerprints(groups_path, annotations_path)
        mapped = ColumnFile.open(path)
        if mapped is None or not _current(mapped.header, sources):
            build(groups_path, annotations_path)
            mapped = ColumnFile.open(path)
        if mapped is None:  # pragma: no cover - a freshly built sidecar always reads back
            raise ValueError(f"{path}: sidecar could not be read after being built")
        return cls(mapped)

    @property
    def distances(self) -> NDArray[np.int32] | None:
        """Per-group distance to the trivial group, `UNKNOWN` where unannotated."""
        return self._columns.get("distances")

    @property
    def potentials(self) -> Mapping[str, int]:
        """The annotated groups as a hash-to-distance map, empty without annotations."""
        if "hash_prefixes" not in self._columns:
            return {}
        return InstancePotentials(
            self._columns["hash_prefixes"],
            self._columns["hash_digests"],
            self._columns["hash_distances"],
        )

    def presentation(self, index: int) -> BalancedPresentation:
        """Rebuild group ``index``'s presentation from its memory-mapped letters."""
        letters = self._columns["letters"]
        bounds = self._columns["word_offsets"][index * self.rank : (index + 1) * self.rank + 1]
        return BalancedPresentation.from_letters(
            self.rank,
            [letters[start:stop].tolist() for start, stop in pairwise(bounds)],
        )
