from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.io import atomic_write_json
from ac_zero.datasets.labels import known_solution
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import PrimitiveMove, inverse_primitive_sequence, move_from_json
from ac_zero.system.parallel import describe_worker_pool, imap_ordered, resolve_worker_count

SCHEMA_VERSION = "aczero-dataset-v3"
SelectStrategy = Literal["smallest", "weighted-random"]
# Emitted incrementally during long grow runs: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class Edge:
    """A construction step: apply `move` to the parent group to reach a node."""

    parent_hash: str
    move: PrimitiveMove


@dataclass(slots=True)
class GraphNode:
    """One group in the reachable-from-trivial AC construction graph.

    `difficulty` is the minimal *known* construction depth -- the fewest forward
    catalog moves from the trivial root seen so far. It is an upper bound that is
    refined whenever a shorter construction is found. `predecessors` holds every
    edge reaching this node at exactly that depth: the co-optimal construction
    moves kept for supervised learning. Longer (suboptimal) constructions are
    never recorded. `reverse_operations` is the strict-primitive length of the
    shortest known trivialization along a co-optimal chain, and `exhausted` marks
    that every catalog move has been applied here so it yields no new neighbour.
    """

    presentation: BalancedPresentation
    difficulty: int
    reverse_operations: int
    predecessors: list[Edge]
    exhausted: bool = False

    @property
    def content_hash(self) -> str:
        return self.presentation.content_hash

    @property
    def total_length(self) -> int:
        return self.presentation.total_length


@dataclass(frozen=True, slots=True)
class GrowConfig:
    """Parameters for one persistent grow run.

    `select` picks which open group to expand next: ``smallest`` always takes the
    shortest (by total relator length), giving one deterministic canonical
    frontier; ``weighted-random`` samples with a bias toward short groups (steered
    by `short_bias` and `seed`) so independent machines explore divergent paths.
    """

    rank: int
    target: int
    select: SelectStrategy = "smallest"
    seed: int = 0
    total_length_cap: int = 48
    short_bias: float = 2.0
    workers: int = 0
    # Dump the dataset to disk every this many newly added groups so an
    # interrupted long run keeps its progress; 0 dumps only at the end. Each
    # checkpoint rewrites the whole file, so keep it well above a handful.
    checkpoint_every: int = 1000


@dataclass(frozen=True, slots=True)
class GrowReport:
    """Summary of one grow run."""

    total: int
    added: int
    expanded: int
    frontier: int
    max_difficulty: int


def grow_dataset(
    path: str | Path, config: GrowConfig, *, progress: ProgressCallback | None = None
) -> GrowReport:
    """Expand a persistent dataset outward from the trivial group.

    Loads the dataset at `path` (seeding the trivial root on the first run),
    repeatedly expands non-exhausted groups by every catalog move, and records
    each novel group with its co-optimal construction edges until `config.target`
    new groups have been added or the reachable frontier (within the length cap)
    is exhausted. The file is rewritten atomically -- both at the end and every
    `config.checkpoint_every` added groups -- so a run interrupted mid-way resumes
    from its last checkpoint, and every run only ever grows the database.

    Each round claims a batch of open groups -- one per worker -- and expands them
    in parallel worker processes, so no two workers ever study the same group.
    """
    path = Path(path)
    rng = random.Random(config.seed)
    workers = resolve_worker_count(config.workers)
    nodes = _load_or_seed(path, config.rank)
    if progress is not None:
        progress(
            "growing dataset",
            {
                "path": str(path),
                "rank": config.rank,
                "target": config.target,
                "select": config.select,
                "seed": config.seed,
                "start_groups": len(nodes),
            },
        )
        _, message, metrics = describe_worker_pool(config.workers)
        progress(message, metrics)

    added = 0
    expanded = 0
    checkpointed = 0
    while added < config.target:
        batch = _select_batch(_open_nodes(nodes), config.select, rng, workers, config.short_bias)
        if not batch:
            break
        neighbour_lists = imap_ordered(
            _expand_presentation,
            [node.presentation for node in batch],
            workers=config.workers,
            initializer=_init_expand_worker,
            initargs=(config.rank,),
        )
        for parent, neighbours in zip(batch, neighbour_lists, strict=True):
            parent.exhausted = True
            expanded += 1
            for move_json, child in neighbours:
                move = move_from_json(move_json)
                if _relax(nodes, parent, move, child, config.total_length_cap):
                    added += 1
        if progress is not None:
            progress(
                "growing dataset",
                {
                    "added": added,
                    "target": config.target,
                    "expanded": expanded,
                    "groups": len(nodes),
                },
            )
        # Snapshot to disk between rounds (a consistent point) so an interrupted
        # run resumes from the last checkpoint rather than losing everything.
        if config.checkpoint_every > 0 and added - checkpointed >= config.checkpoint_every:
            _write(path, config.rank, nodes)
            checkpointed = added
            if progress is not None:
                progress("checkpoint", {"groups": len(nodes), "added": added})

    _write(path, config.rank, nodes)
    frontier = sum(1 for node in nodes.values() if not node.exhausted)
    max_difficulty = max((node.difficulty for node in nodes.values()), default=0)
    report = GrowReport(len(nodes), added, expanded, frontier, max_difficulty)
    if progress is not None:
        progress(
            "grow complete",
            {"groups": report.total, "added": added, "expanded": expanded, "frontier": frontier},
        )
    return report


def _relax(
    nodes: dict[str, GraphNode],
    parent: GraphNode,
    move: PrimitiveMove,
    child: BalancedPresentation,
    total_length_cap: int,
) -> bool:
    """Record one construction edge ``parent --move--> child``; return True if child is new.

    Enforces "record only optimal constructions": a strictly shorter path replaces
    the child's whole edge set, an equal-depth path is appended as a co-optimal
    alternative, and any longer path is discarded.
    """
    if child.total_length > total_length_cap:
        return False
    depth = parent.difficulty + 1
    reverse_ops = parent.reverse_operations + len(inverse_primitive_sequence(move))
    edge = Edge(parent.content_hash, move)
    existing = nodes.get(child.content_hash)
    if existing is None:
        nodes[child.content_hash] = GraphNode(_identify(child), depth, reverse_ops, [edge])
        return True
    if depth < existing.difficulty:
        # A strictly shorter construction supersedes the old co-optimal set. Re-open
        # the node so the improvement propagates to any neighbours already derived
        # from it (SPFA-style relaxation); difficulties only fall, so this settles.
        existing.difficulty = depth
        existing.reverse_operations = reverse_ops
        existing.predecessors = [edge]
        existing.exhausted = False
    elif depth == existing.difficulty:
        if edge not in existing.predecessors:
            existing.predecessors.append(edge)
        existing.reverse_operations = min(existing.reverse_operations, reverse_ops)
    return False


def _open_nodes(nodes: dict[str, GraphNode]) -> list[GraphNode]:
    return [node for node in nodes.values() if not node.exhausted]


def _select_batch(
    open_nodes: list[GraphNode],
    strategy: SelectStrategy,
    rng: random.Random,
    size: int,
    short_bias: float,
) -> list[GraphNode]:
    """Claim up to `size` distinct open groups to expand this round."""
    size = min(size, len(open_nodes))
    if size <= 0:
        return []
    if strategy == "smallest":
        ordered = sorted(open_nodes, key=lambda node: (node.total_length, node.content_hash))
        return ordered[:size]
    pool = list(open_nodes)
    chosen: list[GraphNode] = []
    for _ in range(size):
        weights = [1.0 / (1.0 + node.total_length) ** short_bias for node in pool]
        index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        chosen.append(pool.pop(index))
    return chosen


# Per-worker catalog, built once by the process-pool initializer so the hot
# expansion path never re-allocates the move objects.
_WORKER_CATALOG: ActionCatalog | None = None


def _init_expand_worker(rank: int) -> None:
    global _WORKER_CATALOG
    _WORKER_CATALOG = ActionCatalog(rank)


def _expand_presentation(
    presentation: BalancedPresentation,
) -> list[tuple[dict[str, Any], BalancedPresentation]]:
    """Apply every catalog move to one group, returning its length-changing neighbours."""
    assert _WORKER_CATALOG is not None
    base = presentation.content_hash
    neighbours: list[tuple[dict[str, Any], BalancedPresentation]] = []
    for move in _WORKER_CATALOG.moves:
        child = move.apply(presentation)
        if child.content_hash != base:
            neighbours.append((move.to_json(), child))
    return neighbours


def _identify(child: BalancedPresentation) -> BalancedPresentation:
    """Re-stamp a neighbour with its own id/provenance (moves inherit the parent's)."""
    return BalancedPresentation.from_letters(
        child.rank,
        [relator.letters for relator in child.relators],
        generator_names=child.generator_names,
        presentation_id=f"grown-r{child.rank}-{child.content_hash[:12]}",
        provenance={"family": "trivial_graph_expansion"},
    )


def _trivial_root(rank: int) -> GraphNode:
    """The trivial standard presentation: depth 0, no predecessors, provably solved."""
    return GraphNode(BalancedPresentation.standard(rank), 0, 0, [])


def _load_or_seed(path: Path, rank: int) -> dict[str, GraphNode]:
    nodes: dict[str, GraphNode] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("instances", []):
            node = _entry_to_node(entry)
            nodes[node.content_hash] = node
    if not nodes:
        root = _trivial_root(rank)
        nodes[root.content_hash] = root
    return nodes


def _entry_to_node(entry: dict[str, Any]) -> GraphNode:
    predecessors = [
        Edge(str(edge["parent_hash"]), move_from_json(edge["move"]))
        for edge in entry.get("predecessors", [])
    ]
    return GraphNode(
        presentation=BalancedPresentation.from_json(entry),
        difficulty=int(entry.get("difficulty", 0)),
        reverse_operations=int(entry.get("minimal_known_operations") or 0),
        predecessors=predecessors,
        exhausted=bool(entry.get("exhausted", False)),
    )


def _node_to_entry(node: GraphNode) -> dict[str, Any]:
    entry = node.presentation.to_json()
    entry["difficulty"] = node.difficulty
    entry["exhausted"] = node.exhausted
    entry["predecessors"] = [
        {"parent_hash": edge.parent_hash, "move": edge.move.to_json()} for edge in node.predecessors
    ]
    # Only the trivial root has no construction edge; its empty trivialization is
    # provably optimal, while every constructed group carries a best-known bound.
    entry.update(known_solution(node.reverse_operations, optimal=not node.predecessors).to_json())
    return entry


def _write(path: Path, rank: int, nodes: dict[str, GraphNode]) -> None:
    entries = [_node_to_entry(node) for node in nodes.values()]
    exhausted = sum(1 for node in nodes.values() if node.exhausted)
    data = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "instances": entries,
        "provenance": {
            "generator": "trivial_graph_expansion",
            "count": len(entries),
            "max_difficulty": max((node.difficulty for node in nodes.values()), default=0),
            "exhausted": exhausted,
            "frontier": len(entries) - exhausted,
        },
    }
    atomic_write_json(path, data)
