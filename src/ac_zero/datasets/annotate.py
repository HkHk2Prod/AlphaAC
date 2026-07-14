from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.io import atomic_write_json
from ac_zero.moves.universal import UniversalCatalog, move_set
from ac_zero.system.parallel import describe_worker_pool, imap_ordered

# Emitted incrementally during long annotation passes: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]

_GROUPS_SUFFIX = ".groups.json"
SCHEMA_VERSION = "aczero-annotations-v1"


@dataclass(frozen=True, slots=True)
class AnnotateConfig:
    """Parameters for one per-move-set annotation pass."""

    moveset: str = "universal"
    max_depth: int = 32  # max moves per per-group shorter-distance search; 0 = unbounded
    workers: int = 0
    # Rewrite the whole file every this many freshly computed shorter-distance
    # entries so an interrupted pass keeps its progress; 0 writes only at the end.
    checkpoint_every: int = 5000


@dataclass(frozen=True, slots=True)
class AnnotateReport:
    """Summary of one annotation pass."""

    moveset: str
    total: int
    reached_origin: int
    with_shorter: int
    computed: int
    max_distance_to_origin: int


def annotation_path(groups_path: str | Path, moveset: str) -> Path:
    """Derive the annotation filename `<base>.<moveset>.annotations.json`."""
    groups = Path(groups_path)
    name = groups.name
    base = name[: -len(_GROUPS_SUFFIX)] if name.endswith(_GROUPS_SUFFIX) else groups.stem
    return groups.with_name(f"{base}.{moveset}.annotations.json")


def annotation_entry(
    content_hash: str,
    *,
    distance_to_origin: int | None,
    moves_to_origin: list[int],
    distance_to_shorter: int | None = None,
    moves_to_shorter: list[int] | None = None,
    shorter_proven: bool = False,
) -> dict[str, Any]:
    """Build one annotation entry -- the shape both this pass and `dataset ball` write.

    ``optimal`` marks an entry whose distance to the origin is a proven shortest
    path. The descent fields default to "no descent search was run", which is what
    a ball generated for one move set carries.
    """
    return {
        "hash": content_hash,
        "distance_to_origin": distance_to_origin,
        "optimal_moves_to_origin": moves_to_origin,
        "distance_to_shorter": distance_to_shorter,
        "optimal_moves_to_shorter": moves_to_shorter or [],
        "shorter_proven": shorter_proven,
        "optimal": distance_to_origin is not None,
    }


def annotate(
    groups_path: str | Path,
    config: AnnotateConfig,
    *,
    progress: ProgressCallback | None = None,
) -> AnnotateReport:
    """Annotate a group dataset with distances under one move set.

    Reads the stored universal transition graph and, for the chosen move set,
    computes each group's distance to the origin (a single BFS from the trivial
    root over the *inverse* move set -- exact because the moves are invertible)
    and its distance to a strictly shorter group (a bounded per-group BFS). Both
    carry their co-optimal first moves. The result is written to a separate
    ``<base>.<moveset>.annotations.json`` file, checkpointed and resumable: a
    later pass only recomputes groups whose shorter-distance is still unresolved.
    """
    data = json.loads(Path(groups_path).read_text(encoding="utf-8"))
    rank = int(data["rank"])
    lengths, adjacency = _load_graph(data)
    catalog = UniversalCatalog(rank)
    selected = move_set(config.moveset, catalog)
    move_ids = selected.ids
    inverse_to_forward = {inv: catalog.inverse_id(inv) for inv in selected.inverse_ids(catalog)}

    origin = BalancedPresentation.standard(rank).content_hash
    dist_origin, moves_origin = _bfs_from_origin(origin, adjacency, inverse_to_forward)

    destination = annotation_path(groups_path, config.moveset)
    existing = _load_existing(destination)
    if progress is not None:
        _, worker_message, worker_metrics = describe_worker_pool(config.workers)
        progress(
            "annotating distances",
            {
                "input": str(groups_path),
                "output": str(destination),
                "moveset": config.moveset,
                "total": len(lengths),
                "reached_origin": len(dist_origin),
            },
        )
        progress(worker_message, worker_metrics)

    todo = sorted(
        (h for h in lengths if not _shorter_resolved(existing.get(h))),
        key=lambda h: (lengths[h], h),
    )
    results = imap_ordered(
        _shorter_for,
        todo,
        workers=config.workers,
        initializer=_init_worker,
        initargs=(adjacency, lengths, frozenset(move_ids), config.max_depth),
    )

    shorter: dict[str, tuple[int | None, list[int], bool]] = {}
    for h, entry in existing.items():
        if _shorter_resolved(entry):
            shorter[h] = (
                entry.get("distance_to_shorter"),
                list(entry.get("optimal_moves_to_shorter", [])),
                bool(entry.get("shorter_proven", False)),
            )
    computed = 0
    checkpointed = 0
    for h, result in zip(todo, results, strict=True):
        shorter[h] = result
        computed += 1
        if config.checkpoint_every > 0 and computed - checkpointed >= config.checkpoint_every:
            _write(destination, rank, config.moveset, lengths, dist_origin, moves_origin, shorter)
            checkpointed = computed
            if progress is not None:
                progress(
                    "checkpoint",
                    {
                        "computed": computed,
                        "total": len(todo),
                        "pct_complete": round(100 * computed / len(todo), 1) if todo else 100.0,
                    },
                )

    _write(destination, rank, config.moveset, lengths, dist_origin, moves_origin, shorter)
    report = AnnotateReport(
        moveset=config.moveset,
        total=len(lengths),
        reached_origin=len(dist_origin),
        with_shorter=sum(1 for v in shorter.values() if v[0] is not None),
        computed=computed,
        max_distance_to_origin=max(dist_origin.values(), default=0),
    )
    if progress is not None:
        progress(
            "annotation complete",
            {
                "moveset": report.moveset,
                "total": report.total,
                "reached_origin": report.reached_origin,
                "with_shorter": report.with_shorter,
                "computed": report.computed,
            },
        )
    return report


def _load_graph(data: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[int, str]]]:
    """Build `hash -> length` and `hash -> {move_id -> target}` from the group file.

    Only expanded groups appear in the adjacency; unexpanded frontier groups still
    contribute their length (they can be the shorter target of a descent) but have
    no outgoing edges to traverse.
    """
    lengths: dict[str, int] = {}
    adjacency: dict[str, dict[int, str]] = {}
    for entry in data.get("groups", []):
        h = entry["hash"]
        lengths[h] = int(entry["total_length"])
        transitions = entry.get("transitions")
        if transitions is not None:
            adjacency[h] = {int(k): str(v) for k, v in transitions.items()}
    return lengths, adjacency


def _bfs_from_origin(
    origin: str,
    adjacency: dict[str, dict[int, str]],
    inverse_to_forward: dict[int, int],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """BFS from the origin over the inverse move set; return distances + first moves.

    Traversing an edge ``cur --inv--> target`` (with ``inv`` in the inverse move
    set) means ``target`` is one move further from the origin, and the forward move
    that steps ``target`` back toward the origin is ``inverse_to_forward[inv]`` --
    a co-optimal first move for ``target`` toward the origin.
    """
    dist = {origin: 0}
    moves: dict[str, set[int]] = {origin: set()}
    queue = deque([origin])
    while queue:
        cur = queue.popleft()
        depth = dist[cur]
        for inv, target in adjacency.get(cur, {}).items():
            forward = inverse_to_forward.get(inv)
            if forward is None:
                continue
            if target not in dist:
                dist[target] = depth + 1
                moves[target] = {forward}
                queue.append(target)
            elif dist[target] == depth + 1:
                moves[target].add(forward)
    return dist, {h: sorted(m) for h, m in moves.items()}


# Per-worker graph, set once by the pool initializer so each shorter-distance
# search is pure dict lookups with no per-item pickling of the graph.
_ADJACENCY: dict[str, dict[int, str]] = {}
_LENGTHS: dict[str, int] = {}
_MOVE_IDS: frozenset[int] = frozenset()
_MAX_DEPTH: int = 0


def _init_worker(
    adjacency: dict[str, dict[int, str]],
    lengths: dict[str, int],
    move_ids: frozenset[int],
    max_depth: int,
) -> None:
    global _ADJACENCY, _LENGTHS, _MOVE_IDS, _MAX_DEPTH
    _ADJACENCY = adjacency
    _LENGTHS = lengths
    _MOVE_IDS = move_ids
    _MAX_DEPTH = max_depth


def _shorter_for(start: str) -> tuple[int | None, list[int], bool]:
    """Fewest move-set moves from `start` to a strictly shorter group.

    A layer-synchronized BFS over the stored graph, so on the first layer that
    reaches a shorter group it can collect *every* co-optimal first move that
    begins such a shortest descent. Returns ``(distance, first_moves, proven)``;
    ``distance`` is ``None`` when no shorter group is reachable, and ``proven`` is
    ``True`` when that answer is exact -- a shorter group was found, or the whole
    reachable region was exhausted within the depth budget.
    """
    length = _LENGTHS[start]
    if start not in _ADJACENCY:
        return (None, [], True)  # frontier group: no descent search possible, but settled
    frontier: dict[str, frozenset[int]] = {start: frozenset()}
    visited = {start}
    depth = 0
    while frontier:
        if _MAX_DEPTH > 0 and depth >= _MAX_DEPTH:
            return (None, [], False)  # cut by the depth budget: not proven
        depth += 1
        nxt: dict[str, set[int]] = {}
        shorter_first: set[int] = set()
        for cur, starts in frontier.items():
            for move_id, target in _ADJACENCY.get(cur, {}).items():
                if move_id not in _MOVE_IDS:
                    continue
                begun = {move_id} if cur == start else starts
                if _LENGTHS.get(target, length) < length:
                    shorter_first |= begun
                elif target not in visited:
                    nxt.setdefault(target, set()).update(begun)
        if shorter_first:
            return (depth, sorted(shorter_first), True)
        visited.update(nxt)
        frontier = {node: frozenset(begun) for node, begun in nxt.items()}
    return (None, [], True)  # exhausted the component with no shorter group: proven


def _shorter_resolved(entry: dict[str, Any] | None) -> bool:
    """Whether a group's shorter-distance is already settled (skip on resume)."""
    return entry is not None and bool(entry.get("shorter_proven", False))


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["hash"]: entry for entry in data.get("annotations", [])}


def _write(
    path: Path,
    rank: int,
    moveset: str,
    lengths: dict[str, int],
    dist_origin: dict[str, int],
    moves_origin: dict[str, list[int]],
    shorter: dict[str, tuple[int | None, list[int], bool]],
) -> None:
    annotations = []
    for h in lengths:
        distance, moves, proven = shorter.get(h, (None, [], False))
        annotations.append(
            annotation_entry(
                h,
                distance_to_origin=dist_origin.get(h),
                moves_to_origin=moves_origin.get(h, []),
                distance_to_shorter=distance,
                moves_to_shorter=moves,
                shorter_proven=proven,
            )
        )
    data = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "moveset": moveset,
        "move_catalog": UniversalCatalog(rank).version,
        "annotations": annotations,
        "provenance": {
            "count": len(annotations),
            "reached_origin": len(dist_origin),
            "with_shorter": sum(1 for v in shorter.values() if v[0] is not None),
        },
    }
    atomic_write_json(path, data)
