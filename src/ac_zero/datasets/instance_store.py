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
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.columnar import ColumnFile, Columns
from ac_zero.datasets.columnar import write as write_columns
from ac_zero.datasets.digest_index import (
    DIGEST_BYTES,
    UNKNOWN,
    digest_array,
    sorted_lookup,
    values_for,
)
from ac_zero.datasets.json_stream import iter_json_array

SCHEMA_VERSION = "aczero-instances-v1"
# Relator letters are signed generator indices, so an int8 column holds any rank a
# balanced presentation could realistically use.
_MAX_RANK = 127

__all__ = ["UNKNOWN", "InstanceStore", "build", "read_annotations", "sidecar_path"]


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


def read_annotations(path: Path) -> tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """Stream a `.annotations.json` into the `(digests, distances)` it resolves."""
    digests = bytearray()
    distances = array("q")
    for entry in iter_json_array(path, "annotations"):
        distance = entry.get("distance_to_origin")
        if not isinstance(distance, int):
            continue
        digests += bytes.fromhex(entry["hash"])
        distances.append(distance)
    return digest_array(digests), np.asarray(distances, dtype=np.int32)


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
    return rank, columns, digest_array(digests)


def _lookup_columns(digests: NDArray[np.uint8], distances: NDArray[np.int32]) -> Columns:
    """Store the groups with a known distance as the sidecar's potential lookup."""
    known = np.flatnonzero(distances != UNKNOWN)
    prefixes, ordered, ordered_distances = sorted_lookup(digests[known], distances[known])
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
        lookup = sorted_lookup(*read_annotations(annotations_path))
        distances = values_for(*lookup, digests)
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
        if len(digest) != DIGEST_BYTES:
            raise KeyError(content_hash)
        query = np.frombuffer(digest, dtype=np.uint8).reshape(1, DIGEST_BYTES)
        distance = int(values_for(self._prefixes, self._digests, self._distances, query)[0])
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

    @cached_property
    def longest(self) -> NDArray[np.int32]:
        """Each group's longest relator, in letters.

        A ball is grown with no length cap -- capping it would reroute the shortest
        paths that pass through a long group and cost the distances their optimality --
        so the groups a model's encoder cannot hold are filtered by whoever samples
        them. Derived from the relator bounds rather than stored: the sidecar already
        knows where every word starts and ends.
        """
        bounds = self._columns["word_offsets"].astype(np.int64)
        return np.diff(bounds).reshape(-1, self.rank).max(axis=1).astype(np.int32)

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
