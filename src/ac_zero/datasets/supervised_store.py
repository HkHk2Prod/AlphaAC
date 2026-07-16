"""The supervised labels of a grown dataset, as memory-mapped columns.

The label a supervised move-predictor needs is implicit in any distance-annotated
dataset: applying a move to a group says which group it reaches, and the annotation
file says how far from the origin each group is. Scoring one against the other gives
exactly the quantity the task is defined on --

    delta[group, action] = distance(move takes group here) - distance(group)

-- so a move that steps toward the trivial group scores ``-1``, one that stalls ``0``,
and one that steps away ``+1`` or worse. (Under a non-invertible move set such as
``strict-ac`` a single move can strand the search much further from the origin than
one step, which is why the column stores the real difference and not just its sign.)

An action whose child the dataset knows nothing about -- or that the environment would
refuse to play, being a no-op or overflowing the encoder's ``max_relator_tokens`` --
carries ``DELTA_UNKNOWN`` instead, so the labels a model is trained on are exactly the
moves it will be allowed to make.

The join is done once, offline, and written to a sidecar next to the instance store's:
a ``(groups, actions)`` int16 matrix, the per-group distance, its longest relator, and
the split it belongs to. Training then reads labels straight out of a mapping instead
of re-deriving them, and every worker shares one copy through the page cache. The
sidecar is fingerprinted on all three source files and on the capacity it was built
for, so changing any of them rebuilds it rather than silently training on stale labels.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator
from functools import cached_property
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
from ac_zero.datasets.instance_store import read_annotations
from ac_zero.datasets.json_stream import iter_json_array
from ac_zero.datasets.split import SPLIT_CODES
from ac_zero.encoding.padded import within_capacity
from ac_zero.moves.universal import MoveSetCatalog, moveset_catalog
from ac_zero.system.parallel import describe_worker_pool, imap_ordered, resolve_worker_count

# A build streams multi-gigabyte JSON single-threaded and can run for an hour on a
# 30M-group ball; without a heartbeat the run looks hung. This is how often it reports.
ProgressCallback = Callable[[str, dict[str, Any]], None]
_LOG_EVERY = 1_000_000

# v2 stores `hash_prefixes` in native byte order (see `digest_index.prefixes`); a v1
# sidecar's big-endian column would keep every distance lookup on the O(n) path.
SCHEMA_VERSION = "aczero-supervised-v2"
# No neighbour distance is known for this move: its child is outside the grown region
# (or the group's own distance is unknown). Distinct from any real delta.
DELTA_UNKNOWN = np.iinfo(np.int16).min
_DELTA_LIMIT = int(np.iinfo(np.int16).max)
# Groups joined per pass. Bounds the transient child-digest buffer (rows * actions * 32
# bytes) so a multi-million-group dataset joins in tens of megabytes, not gigabytes.
_CHUNK = 50_000

# The prefix-sorted lookup columns a distance join searches: see `digest_index`.
DistanceIndex = tuple[NDArray[np.uint64], NDArray[np.uint8], NDArray[np.int32]]


def sidecar_path(groups_path: Path, moveset: str) -> Path:
    """Where the supervised sidecar for this dataset and move set lives."""
    return groups_path.with_suffix(f"{groups_path.suffix}.{moveset}.supervised")


def _fingerprints(paths: dict[str, Path]) -> dict[str, Any]:
    """Identify the source documents, so a sidecar built from older ones is rejected."""
    return {
        name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for name, path in paths.items()
    }


def _read_split(path: Path) -> tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """Stream a `.split.json` into the `(digests, split codes)` it assigns."""
    digests = bytearray()
    codes = array("q")
    for entry in iter_json_array(path, "assignments"):
        code = SPLIT_CODES.get(str(entry["split"]))
        if code is None:
            raise ValueError(f"{path}: unknown split {entry['split']!r}")
        digests += bytes.fromhex(entry["hash"])
        codes.append(code)
    return digest_array(digests), np.asarray(codes, dtype=np.int32)


# One group's minimal payload for a worker: its rank, its relators, and its own hash.
_GroupInput = tuple[int, list[list[int]], str]
# The digests and playable flags a worker returns for a chunk: concatenated own digests,
# concatenated child digests (32 bytes per (group, action) slot), and the playable flags.
_EncodedChunk = tuple[bytes, bytes, list[bool]]

# Move application + child hashing is pure-Python and GIL-bound, so it is fanned out to
# worker *processes*. Each holds its own catalog and capacity -- never the gigabyte-scale
# distance index, which stays in the main process -- so a worker carries kilobytes of
# state. Set once per process by the pool initializer to avoid re-pickling per chunk.
_WORKER_CATALOG: MoveSetCatalog | None = None
_WORKER_CAPACITY = 0
# Chunks kept in flight per worker while streaming, bounding the parsed groups held in
# memory to (workers * this * _CHUNK) entries regardless of dataset size.
_WINDOW_CHUNKS = 2


def _init_encode_worker(moveset: str, rank: int, max_relator_tokens: int) -> None:
    """Build the worker's catalog and remember its capacity, once per process."""
    global _WORKER_CATALOG, _WORKER_CAPACITY
    _WORKER_CATALOG = moveset_catalog(moveset, rank)
    _WORKER_CAPACITY = max_relator_tokens


def _encode_chunk(chunk: list[_GroupInput]) -> _EncodedChunk:
    """Apply every move to every group in one chunk -- the CPU-heavy work workers run.

    The children are derived from the moves rather than read out of the group file, so a
    dataset that stores no adjacency at all -- a ``dataset ball``, whose distances are
    proven rather than searched for -- labels exactly like a grown one. An unplayable move
    still contributes its 32 zero bytes so the child rows stay aligned with the
    ``(group, action)`` grid; the ``present`` flag discards it.
    """
    assert _WORKER_CATALOG is not None  # set by _init_encode_worker before any chunk
    moves = _WORKER_CATALOG.moves
    own = bytearray()
    children = bytearray()
    present: list[bool] = []
    for rank, relators, own_hash in chunk:
        presentation = BalancedPresentation.from_letters(rank, [list(r) for r in relators])
        own += bytes.fromhex(own_hash)
        for move in moves:
            child = move.apply(presentation)
            playable = child.content_hash != own_hash and within_capacity(child, _WORKER_CAPACITY)
            present.append(playable)
            children += bytes.fromhex(child.content_hash) if playable else bytes(32)
    return bytes(own), bytes(children), present


def _stream_encoded(
    groups_path: Path, moveset: str, max_relator_tokens: int, workers: int
) -> Iterator[tuple[int, _EncodedChunk]]:
    """Stream groups, fan their move-application out to workers, yield ``(rank, encoded)``.

    Chunks are dispatched in windows so at most ``workers * _WINDOW_CHUNKS`` chunks of
    parsed groups are resident at once; results are yielded in input order (via
    :func:`imap_ordered`), so the delta rows come out in group order and the sidecar is
    identical for any worker count.
    """
    window = max(1, resolve_worker_count(workers)) * _WINDOW_CHUNKS
    rank = 0
    pending: list[list[_GroupInput]] = []
    chunk: list[_GroupInput] = []

    def dispatch() -> Iterator[_EncodedChunk]:
        nonlocal pending
        if not pending:
            return iter(())
        results = imap_ordered(
            _encode_chunk,
            pending,
            workers=workers,
            initializer=_init_encode_worker,
            initargs=(moveset, rank, max_relator_tokens),
        )
        pending = []
        return results

    for entry in iter_json_array(groups_path, "groups"):
        if not rank:
            rank = int(entry["rank"])
        chunk.append((rank, entry["relators"], str(entry["hash"])))
        if len(chunk) >= _CHUNK:
            pending.append(chunk)
            chunk = []
            if len(pending) >= window:
                for encoded in dispatch():
                    yield rank, encoded
    if chunk:
        pending.append(chunk)
    for encoded in dispatch():
        yield rank, encoded


class _DeltaJoin:
    """Turns worker-encoded chunks into delta rows against the distance index (main side)."""

    def __init__(self, distances: DistanceIndex) -> None:
        self._distances = distances  # prefix-sorted (prefixes, digests, distances)
        self.actions = 0  # derived from the first chunk's shape
        self.deltas: list[NDArray[np.int16]] = []
        self.own_distances: list[NDArray[np.int32]] = []
        self.group_digests = bytearray()

    def add(self, encoded: _EncodedChunk) -> None:
        """Resolve one encoded chunk's own/child digests into delta rows."""
        own_bytes, children_bytes, present = encoded
        if not present:
            return
        groups = len(own_bytes) // DIGEST_BYTES
        if not self.actions:
            self.actions = len(present) // groups
        self.group_digests += own_bytes
        own = values_for(*self._distances, digest_array(own_bytes))
        child = values_for(*self._distances, digest_array(children_bytes)).reshape(-1, self.actions)
        known = (
            np.asarray(present, dtype=np.bool_).reshape(child.shape)
            & (child != UNKNOWN)
            & (own != UNKNOWN)[:, None]
        )
        delta = np.clip(
            child.astype(np.int64) - own.astype(np.int64)[:, None], -_DELTA_LIMIT, _DELTA_LIMIT
        )
        self.deltas.append(np.where(known, delta, DELTA_UNKNOWN).astype(np.int16))
        self.own_distances.append(own)


def build(
    groups_path: Path,
    annotations_path: Path,
    split_file: Path,
    moveset: str,
    max_relator_tokens: int,
    *,
    workers: int = 0,
    progress: ProgressCallback | None = None,
) -> None:
    """Join the groups, their distances, and their split into the supervised sidecar.

    ``workers`` fans the per-group move-application across that many worker processes (0 =
    every physical core), turning the build's dominant cost -- hundreds of millions of
    pure-Python move applications on a large ball -- from single-core into parallel work.
    ``progress`` is called with ``(message, metrics)`` at each phase boundary and every
    million groups, so a long build reports that it is alive rather than sitting silent.
    """
    report = progress or (lambda _message, _metrics: None)
    report("reading annotations", {"file": annotations_path.name})
    distances = sorted_lookup(*read_annotations(annotations_path))
    _, worker_message, worker_metrics = describe_worker_pool(workers)
    report(worker_message, worker_metrics)

    join = _DeltaJoin(distances)
    rank = 0
    seen = 0
    logged = 0
    report("scanning groups and applying moves", {"groups": 0})
    for chunk_rank, encoded in _stream_encoded(groups_path, moveset, max_relator_tokens, workers):
        rank = chunk_rank
        join.add(encoded)
        seen += len(encoded[0]) // DIGEST_BYTES
        if seen - logged >= _LOG_EVERY:
            logged = seen
            report("scanning groups and applying moves", {"groups": seen})
    if not seen:
        raise ValueError(f"{groups_path}: dataset has no groups")

    report("reading split", {"file": split_file.name})
    group_digests = digest_array(join.group_digests)
    splits = values_for(*sorted_lookup(*_read_split(split_file)), group_digests)
    report("writing sidecar", {"groups": seen})
    columns: Columns = {
        "deltas": np.concatenate(join.deltas),
        "distances": np.concatenate(join.own_distances),
        "splits": splits.astype(np.int8),
    }
    header = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "moveset": moveset,
        "count": len(group_digests),
        "actions": join.actions,
        "max_relator_tokens": max_relator_tokens,
        "sources": _fingerprints(
            {"groups": groups_path, "annotations": annotations_path, "split": split_file}
        ),
    }
    write_columns(sidecar_path(groups_path, moveset), header, columns)


class SupervisedStore:
    """The memory-mapped supervised labels of one dataset under one move set.

    Row ``i`` is group ``i`` of the group file -- the same order
    :class:`ac_zero.datasets.instance_store.InstanceStore` indexes -- so a training
    example pairs this store's labels with that store's presentation at one index.
    """

    def __init__(self, mapped: ColumnFile) -> None:
        self.path = mapped.path
        self.rank = int(mapped.header["rank"])
        self.moveset = str(mapped.header["moveset"])
        self.count = int(mapped.header["count"])
        self.actions = int(mapped.header["actions"])
        self.max_relator_tokens = int(mapped.header["max_relator_tokens"])
        # Held for its lifetime: dropping it would close the mapping the columns view.
        self._mapped = mapped
        self.deltas: NDArray[np.int16] = mapped.columns["deltas"]
        self.distances: NDArray[np.int32] = mapped.columns["distances"]
        self.splits: NDArray[np.int8] = mapped.columns["splits"]

    @classmethod
    def open(
        cls,
        groups_path: Path,
        annotations_path: Path,
        split_file: Path,
        moveset: str,
        max_relator_tokens: int,
        *,
        workers: int = 0,
        progress: ProgressCallback | None = None,
    ) -> SupervisedStore:
        """Map the sidecar for these sources, (re)building it when absent or stale.

        ``max_relator_tokens`` is the encoder capacity the labels are for -- the same
        bound the dataset was generated under. It decides which moves the environment
        would let a model play, so a run that changes it gets a rebuilt sidecar rather
        than the old one's labels.
        """
        path = sidecar_path(groups_path, moveset)
        sources = _fingerprints(
            {"groups": groups_path, "annotations": annotations_path, "split": split_file}
        )
        mapped = ColumnFile.open(path)
        if (
            mapped is None
            or mapped.header.get("schema_version") != SCHEMA_VERSION
            or mapped.header.get("sources") != sources
            or mapped.header.get("max_relator_tokens") != max_relator_tokens
        ):
            build(
                groups_path,
                annotations_path,
                split_file,
                moveset,
                max_relator_tokens,
                workers=workers,
                progress=progress,
            )
            mapped = ColumnFile.open(path)
        if mapped is None:  # pragma: no cover - a freshly built sidecar always reads back
            raise ValueError(f"{path}: supervised sidecar could not be read after being built")
        return cls(mapped)

    @cached_property
    def _labelled(self) -> NDArray[np.bool_]:
        """Whether each group has any move whose neighbour's distance is known.

        Cached: it is the same question for every split, and answering it scans the
        whole ``(groups, actions)`` delta matrix.
        """
        labelled: NDArray[np.bool_] = (self.deltas != DELTA_UNKNOWN).any(axis=1)
        return labelled

    def trainable(self, split: str) -> NDArray[np.int64]:
        """The rows of ``split`` that carry a usable label.

        A group is trainable when it is in the split, its own distance to the origin is
        known and positive -- the origin itself is the goal, not a state to move out of
        -- and at least one of its moves leads somewhere whose distance is known. No
        group is dropped for being long: the dataset was generated under this run's
        relator bound, so every group in it is one the encoder can hold.
        """
        code = SPLIT_CODES.get(split)
        if code is None:
            raise ValueError(f"unknown split {split!r}; choose from {sorted(SPLIT_CODES)}")
        keep = (self.splits == code) & (self.distances > 0) & self._labelled
        return np.flatnonzero(keep).astype(np.int64)
