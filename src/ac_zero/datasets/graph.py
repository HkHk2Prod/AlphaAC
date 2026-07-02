from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.expand import ChildRecord
from ac_zero.datasets.io import atomic_write_json
from ac_zero.datasets.labels import known_solution
from ac_zero.moves.primitive import PrimitiveMove, move_from_json

SCHEMA_VERSION = "aczero-dataset-v3"
SelectStrategy = Literal["smallest", "weighted-random"]


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


class ConstructionGraph:
    """The persistent set of groups reachable from the trivial root, keyed by hash.

    Owns the co-optimal construction invariant: expansion happens in worker
    processes, but every mutation of the graph -- recording edges, superseding a
    longer construction with a shorter one, re-opening a node so an improvement
    propagates (SPFA-style) -- runs here in a single deterministic pass so the
    result never depends on how many workers produced the neighbours.
    """

    def __init__(self, nodes: dict[str, GraphNode]) -> None:
        self.nodes = nodes

    @classmethod
    def load_or_seed(cls, path: Path, rank: int) -> ConstructionGraph:
        nodes: dict[str, GraphNode] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("instances", []):
                node = _entry_to_node(entry)
                nodes[node.content_hash] = node
        if not nodes:
            root = _trivial_root(rank)
            nodes[root.content_hash] = root
        return cls(nodes)

    def select_batch(
        self,
        strategy: SelectStrategy,
        rng: random.Random,
        size: int,
        short_bias: float,
        claimed: frozenset[str] | set[str] = frozenset(),
    ) -> list[GraphNode]:
        """Claim up to `size` open groups to expand, skipping any already `claimed`.

        `claimed` holds the hashes of groups whose expansion is still in flight in
        the pipeline, so the next batch never re-selects a group already being
        expanded. The `smallest` key uses each node's cached content hash, so
        re-scanning the frontier stays cheap.
        """
        open_nodes = [
            node
            for node in self.nodes.values()
            if not node.exhausted and node.content_hash not in claimed
        ]
        size = min(size, len(open_nodes))
        if size <= 0:
            return []
        if strategy == "smallest":
            ordered = sorted(open_nodes, key=lambda node: (node.total_length, node.content_hash))
            return ordered[:size]
        # weighted-random: never consume the whole frontier in one round, so the
        # seeded weighting always chooses *which* short groups advance. That biased
        # subset is what makes independent seeds diverge; capping at half the
        # frontier keeps divergence true at every scale (the batch-size cap alone
        # does not once the frontier is smaller than a batch).
        size = min(size, max(1, len(open_nodes) // 2))
        pool = list(open_nodes)
        chosen: list[GraphNode] = []
        for _ in range(size):
            weights = [1.0 / (1.0 + node.total_length) ** short_bias for node in pool]
            index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
            chosen.append(pool.pop(index))
        return chosen

    def merge(self, parent: GraphNode, records: list[ChildRecord]) -> int:
        """Fold one group's expansion into the graph; return the count of new groups.

        Enforces "record only optimal constructions": a strictly shorter path
        replaces the child's whole edge set and re-opens it so the improvement
        propagates, an equal-depth path is appended as a co-optimal alternative,
        and any longer path is discarded. Hashes arrive precomputed, so this only
        rebuilds a presentation for genuinely new groups.
        """
        added = 0
        depth = parent.difficulty + 1
        base_ops = parent.reverse_operations
        parent_hash = parent.content_hash
        names = parent.presentation.generator_names
        for record in records:
            reverse_ops = base_ops + record.reverse_delta
            existing = self.nodes.get(record.child_hash)
            if existing is None:
                edge = Edge(parent_hash, move_from_json(record.move))
                self.nodes[record.child_hash] = _grown_node(names, record, depth, reverse_ops, edge)
                added += 1
            elif depth < existing.difficulty:
                existing.difficulty = depth
                existing.reverse_operations = reverse_ops
                existing.predecessors = [Edge(parent_hash, move_from_json(record.move))]
                existing.exhausted = False
            elif depth == existing.difficulty:
                edge = Edge(parent_hash, move_from_json(record.move))
                if edge not in existing.predecessors:
                    existing.predecessors.append(edge)
                existing.reverse_operations = min(existing.reverse_operations, reverse_ops)
        return added

    def frontier(self) -> int:
        return sum(1 for node in self.nodes.values() if not node.exhausted)

    def max_difficulty(self) -> int:
        return max((node.difficulty for node in self.nodes.values()), default=0)

    def write(self, path: Path, rank: int) -> None:
        entries = [_node_to_entry(node) for node in self.nodes.values()]
        exhausted = len(entries) - self.frontier()
        data = {
            "schema_version": SCHEMA_VERSION,
            "rank": rank,
            "instances": entries,
            "provenance": {
                "generator": "trivial_graph_expansion",
                "count": len(entries),
                "max_difficulty": self.max_difficulty(),
                "exhausted": exhausted,
                "frontier": len(entries) - exhausted,
            },
        }
        atomic_write_json(path, data)


def _grown_node(
    names: tuple[str, ...], record: ChildRecord, depth: int, reverse_ops: int, edge: Edge
) -> GraphNode:
    """Build a fresh node for a newly discovered group from its compact record."""
    rank = len(names)
    presentation = BalancedPresentation.from_letters(
        rank,
        record.letters,
        generator_names=names,
        presentation_id=f"grown-r{rank}-{record.child_hash[:12]}",
        provenance={"family": "trivial_graph_expansion"},
    )
    return GraphNode(presentation, depth, reverse_ops, [edge])


def _trivial_root(rank: int) -> GraphNode:
    """The trivial standard presentation: depth 0, no predecessors, provably solved."""
    return GraphNode(BalancedPresentation.standard(rank), 0, 0, [])


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
    # Only the trivial root sits at depth 0; its empty trivialization is provably
    # optimal, while every constructed group carries a best-known upper bound.
    entry.update(known_solution(node.reverse_operations, optimal=node.difficulty == 0).to_json())
    return entry
