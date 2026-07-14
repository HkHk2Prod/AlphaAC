from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.expand import NeighbourRecord
from ac_zero.datasets.io import atomic_write_json
from ac_zero.datasets.json_stream import iter_json_array, read_members_before

SCHEMA_VERSION = "aczero-groups-v1"
MOVE_CATALOG = "universal-v1"
SelectStrategy = Literal["smallest", "weighted-random"]

# The per-relator bound the dataset was generated under lives under this top-level
# member -- in every group file, grown or ball. It is named to sort before ``groups``
# so `read_relator_bound` recovers it from a multi-gigabyte file without decoding a
# single group (documents are written with sorted keys).
BOUNDS_KEY = "bounds"
RELATOR_BOUND = "max_relator_length"

# Provenance strings for the ``source`` field.
SOURCE_TRIVIAL = "trivial"
SOURCE_EXPANSION = "universal_expansion"
SOURCE_ORIGIN_BALL = "origin_ball"


def read_relator_bound(path: str | Path) -> int:
    """The longest relator any group in this dataset may carry; 0 if it is unbounded.

    Every consumer that puts the dataset behind an encoder checks its capacity against
    this: a model bounded differently from its data is trained on moves it cannot play,
    or plays moves whose outcome the data never proved a distance for.
    """
    bounds = read_members_before(Path(path), "groups").get(BOUNDS_KEY, {})
    return int(bounds.get(RELATOR_BOUND, 0))


def check_relator_bound(path: str | Path, stored: int, requested: int, what: str) -> None:
    """Refuse to extend a dataset under a bound other than the one it was grown under.

    A ball grown to `rel48` and then extended unbounded is not a `rel48` ball with more
    groups in it: the groups added past the bound reroute the shortest paths through
    themselves, and every distance already written becomes a claim about a graph the
    file no longer holds.
    """
    if stored == requested:
        return
    raise ValueError(
        f"{path} is a {_describe_bound(stored)} {what}; it cannot be extended as a "
        f"{_describe_bound(requested)} one -- generate that one under its own name"
    )


def _describe_bound(bound: int) -> str:
    return f"max_relator_length={bound}" if bound > 0 else "unbounded"


@dataclass(slots=True)
class GroupNode:
    """One group in the universal construction graph, stored in minimal form.

    ``transitions`` maps each applicable universal move ID to the content hash of
    the group it reaches (the complete local adjacency within the relator bound).
    ``transitions is None`` marks an unexpanded frontier group -- one discovered as
    a neighbour but whose own moves have not been applied yet -- so a run resumes
    from exactly the groups still to expand without a separate flag.
    """

    presentation: BalancedPresentation
    ac_trivial: bool | None
    source: str
    transitions: dict[int, str] | None = None

    @property
    def content_hash(self) -> str:
        return self.presentation.content_hash

    @property
    def total_length(self) -> int:
        return self.presentation.total_length

    @property
    def exhausted(self) -> bool:
        """Whether every universal move has been applied here (adjacency recorded)."""
        return self.transitions is not None


class GroupStore:
    """The persistent set of groups reachable from the trivial root, keyed by hash.

    Generation is pure graph construction: expand a frontier group by every
    universal move and record the resulting adjacency. Because the moves are
    invertible, no reverse-path bookkeeping is needed -- distances are computed
    later by the annotation pass over the stored adjacency.
    """

    def __init__(self, nodes: dict[str, GroupNode], rank: int, max_relator_length: int = 0) -> None:
        self.nodes = nodes
        self.rank = rank
        self.max_relator_length = max_relator_length

    @classmethod
    def load_or_seed(cls, path: Path, rank: int, max_relator_length: int = 0) -> GroupStore:
        """Reopen the stored graph, or seed a fresh one holding just the trivial root.

        The document is streamed rather than parsed: :func:`json.loads` holds the whole
        file as text *and* as a decoded object graph before the first node exists, which
        on a multi-gigabyte dataset costs several times its size. The rank is read off
        the entries, which each carry it -- keys are written sorted, so the document's
        own ``rank`` member trails the ``groups`` array and reaching it would mean
        buffering the very thing being streamed. The relator bound is the exception: it
        is written to sort *before* the groups precisely so a resuming run can check the
        bound it was asked for against the one the file was grown under.
        """
        nodes: dict[str, GroupNode] = {}
        if path.exists():
            check_relator_bound(path, read_relator_bound(path), max_relator_length, "dataset")
            for entry in iter_json_array(path, "groups"):
                rank = int(entry.get("rank", rank))
                node = _entry_to_node(entry, rank)
                nodes[node.content_hash] = node
        if not nodes:
            root = _trivial_root(rank)
            nodes[root.content_hash] = root
        return cls(nodes, rank, max_relator_length)

    def select_batch(
        self,
        strategy: SelectStrategy,
        rng: random.Random,
        size: int,
        short_bias: float,
        claimed: frozenset[str] | set[str] = frozenset(),
    ) -> list[GroupNode]:
        """Claim up to `size` unexpanded groups to expand, skipping any `claimed`.

        `claimed` holds the hashes of groups whose expansion is still in flight, so
        the next batch never re-selects one already being expanded. `smallest`
        gives a deterministic shortest-first frontier; `weighted-random` samples
        with a short-group bias so independent seeds diverge.
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
        # Never consume the whole frontier in one round, so the seeded weighting
        # always chooses *which* short groups advance -- that biased subset is what
        # makes independent seeds diverge at every scale.
        size = min(size, max(1, len(open_nodes) // 2))
        pool = list(open_nodes)
        chosen: list[GroupNode] = []
        for _ in range(size):
            weights = [1.0 / (1.0 + node.total_length) ** short_bias for node in pool]
            index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
            chosen.append(pool.pop(index))
        return chosen

    def merge(self, parent: GroupNode, records: list[NeighbourRecord]) -> int:
        """Record one group's full adjacency; return the count of new groups.

        Every neighbour becomes a `move_id -> child_hash` transition on the parent,
        and any genuinely new group is added as an unexpanded frontier node. A
        group reachable from the trivial root by AC moves is AC-trivial, so grown
        groups carry `ac_trivial=True`.
        """
        added = 0
        transitions: dict[int, str] = {}
        for record in records:
            transitions[record.move_id] = record.child_hash
            if record.child_hash not in self.nodes:
                self.nodes[record.child_hash] = _grown_node(record, self.rank)
                added += 1
        parent.transitions = transitions
        return added

    def frontier(self) -> int:
        return sum(1 for node in self.nodes.values() if not node.exhausted)

    def max_length(self) -> int:
        return max((node.total_length for node in self.nodes.values()), default=0)

    def write(self, path: Path) -> None:
        """Rewrite the graph, streaming the entries so a checkpoint stays bounded."""
        count = len(self.nodes)
        frontier = self.frontier()
        data = {
            BOUNDS_KEY: {RELATOR_BOUND: self.max_relator_length},
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "move_catalog": MOVE_CATALOG,
            "groups": (_node_to_entry(node) for node in self.nodes.values()),
            "provenance": {
                "generator": "universal_graph_expansion",
                "count": count,
                "frontier": frontier,
                "exhausted": count - frontier,
                "max_length": self.max_length(),
            },
        }
        atomic_write_json(path, data)


def _grown_node(record: NeighbourRecord, rank: int) -> GroupNode:
    """Build a fresh unexpanded node for a newly discovered group."""
    presentation = BalancedPresentation.from_letters(rank, record.letters)
    return GroupNode(presentation, ac_trivial=True, source=SOURCE_EXPANSION, transitions=None)


def _trivial_root(rank: int) -> GroupNode:
    """The trivial standard presentation: the origin, provably AC-trivial."""
    return GroupNode(BalancedPresentation.standard(rank), ac_trivial=True, source=SOURCE_TRIVIAL)


def _entry_to_node(entry: dict[str, Any], rank: int) -> GroupNode:
    presentation = BalancedPresentation.from_letters(rank, entry["relators"])
    raw = entry.get("transitions")
    transitions = {int(k): str(v) for k, v in raw.items()} if raw is not None else None
    return GroupNode(
        presentation=presentation,
        ac_trivial=entry.get("ac_trivial"),
        source=str(entry.get("source", "")),
        transitions=transitions,
    )


def group_entry(
    presentation: BalancedPresentation,
    *,
    ac_trivial: bool | None,
    source: str,
    transitions: dict[int, str] | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    """Build one minimal group-dataset entry (also used for curated candidates).

    ``content_hash`` lets a caller that already holds the hash pass it in rather than
    have the presentation derive (and then cache) its own -- which, over millions of
    groups rewritten at every checkpoint, is a hash and a string per group.
    """
    entry: dict[str, Any] = {
        "hash": content_hash or presentation.content_hash,
        "rank": presentation.rank,
        "ac_trivial": ac_trivial,
        "source": source,
        "relators": [list(relator.letters) for relator in presentation.relators],
        "total_length": presentation.total_length,
    }
    if transitions is not None:
        entry["transitions"] = {str(move_id): target for move_id, target in transitions.items()}
    return entry


def _node_to_entry(node: GroupNode) -> dict[str, Any]:
    return group_entry(
        node.presentation,
        ac_trivial=node.ac_trivial,
        source=node.source,
        transitions=node.transitions,
    )
