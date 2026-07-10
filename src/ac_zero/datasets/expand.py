from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import NamedTuple, Protocol

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.moves.universal import UniversalCatalog
from ac_zero.system.parallel import resolve_worker_count


class NeighbourRecord(NamedTuple):
    """One universal-move neighbour, kept compact for cheap IPC.

    Only what the main process needs to record an adjacency edge and, for a
    genuinely new group, add its node -- without re-deriving anything: the
    ``move_id`` that produced it (its universal catalog ID, the transition key),
    the precomputed ``child_hash`` (so the merge never re-hashes), the freely
    reduced relator ``letters`` (a full presentation is rebuilt only when the
    child is new), and the child's ``total_length``.
    """

    move_id: int
    child_hash: str
    letters: tuple[tuple[int, ...], ...]
    total_length: int


# Per-worker state, built once by the pool initializer so the hot expansion path
# never re-allocates the catalog or re-reads the length cap.
_WORKER_CATALOG: UniversalCatalog | None = None
_WORKER_CAP: int = 0


def _init_expand_worker(rank: int, total_length_cap: int) -> None:
    global _WORKER_CATALOG, _WORKER_CAP
    _WORKER_CATALOG = UniversalCatalog(rank)
    _WORKER_CAP = total_length_cap


def expand_group(presentation: BalancedPresentation) -> list[NeighbourRecord]:
    """Apply every universal move to one group, hashing each neighbour in-worker.

    Returns every non-identity neighbour within the length cap (0 = no cap), keyed by universal
    move ID -- the complete local adjacency. All the expensive per-neighbour work
    (applying the move, freely reducing, hashing) happens here in the worker, so
    the main process only records precomputed edges and adds new nodes.
    """
    assert _WORKER_CATALOG is not None
    base = presentation.content_hash
    records: list[NeighbourRecord] = []
    for move_id, move in enumerate(_WORKER_CATALOG.moves):
        child = move.apply(presentation)
        child_hash = child.content_hash
        if child_hash == base or (_WORKER_CAP > 0 and child.total_length > _WORKER_CAP):
            continue
        letters = tuple(relator.letters for relator in child.relators)
        records.append(NeighbourRecord(move_id, child_hash, letters, child.total_length))
    return records


# A single group's expansion is cheap (well under a millisecond) and each group is
# expanded once, so a short run finishes before a worker pool would even finish
# spawning. Stay inline until this many groups have been expanded, then -- if the
# run is clearly long enough to amortize it -- fan out for the remaining rounds.
_SPAWN_AFTER_GROUPS = 512


def _expand_chunk(presentations: list[BalancedPresentation]) -> list[list[NeighbourRecord]]:
    """Expand a contiguous slice of a batch in one worker task, preserving order."""
    return [expand_group(presentation) for presentation in presentations]


def _contiguous_chunks(
    presentations: list[BalancedPresentation], count: int
) -> list[list[BalancedPresentation]]:
    """Split a batch into `count` order-preserving slices, one task per worker."""
    count = max(1, min(len(presentations), count))
    size = -(-len(presentations) // count)  # ceil, so `count` slices cover everything
    return [presentations[i : i + size] for i in range(0, len(presentations), size)]


class BatchHandle(Protocol):
    """A submitted batch whose per-group neighbour records can be awaited in order."""

    def result(self) -> list[list[NeighbourRecord]]: ...


class _DoneBatch:
    """Inline (single-process) result -- already computed at submit time."""

    def __init__(self, records: list[list[NeighbourRecord]]) -> None:
        self._records = records

    def result(self) -> list[list[NeighbourRecord]]:
        return self._records


class _FuturesBatch:
    """A batch expanding across worker processes; `result` gathers slices in order."""

    def __init__(self, futures: list[Future[list[list[NeighbourRecord]]]]) -> None:
        self._futures = futures

    def result(self) -> list[list[NeighbourRecord]]:
        records: list[list[NeighbourRecord]] = []
        for future in self._futures:
            records.extend(future.result())
        return records


class ExpansionPool:
    """Worker pool for graph expansion, spawned lazily and reused across rounds.

    Two traps this avoids: creating a :class:`ProcessPoolExecutor` per round
    (on ``forkserver`` start methods every round re-spawned workers and rebuilt
    the catalog), and paying that spawn at all on short runs that finish inline in
    a few milliseconds. So expansion starts in-process and only fans out once a
    run has expanded enough groups to repay the spawn; from then on one pool stays
    open. :meth:`submit_batch` submits a whole batch *eagerly* and hands back a
    handle, so the caller can keep several batches in flight -- workers stay busy
    expanding later batches while the main process merges an earlier one. Fanning
    out never changes the output: :func:`expand_group` is pure and per-group
    results are always gathered back in submission order.
    """

    def __init__(self, rank: int, total_length_cap: int, workers: int | None) -> None:
        self._rank = rank
        self._cap = total_length_cap
        self._resolved = resolve_worker_count(workers)
        self._executor: ProcessPoolExecutor | None = None
        self._inline_ready = False
        self._expanded = 0

    def __enter__(self) -> ExpansionPool:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def submit_batch(self, presentations: Sequence[BalancedPresentation]) -> BatchHandle:
        """Kick off expansion of a whole batch and return a handle to await it."""
        self._expanded += len(presentations)
        if self._executor is None and self._resolved > 1 and self._expanded >= _SPAWN_AFTER_GROUPS:
            self._executor = ProcessPoolExecutor(
                max_workers=self._resolved,
                initializer=_init_expand_worker,
                initargs=(self._rank, self._cap),
            )
        if self._executor is None:
            if not self._inline_ready:
                _init_expand_worker(self._rank, self._cap)
                self._inline_ready = True
            return _DoneBatch([expand_group(presentation) for presentation in presentations])
        # One task per worker keeps the batch balanced while minimizing the number
        # of IPC round trips; with several batches in flight the pool stays full.
        chunks = _contiguous_chunks(list(presentations), self._resolved)
        return _FuturesBatch([self._executor.submit(_expand_chunk, chunk) for chunk in chunks])
