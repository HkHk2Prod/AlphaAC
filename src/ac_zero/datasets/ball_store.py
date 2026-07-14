"""The persistent state of a closest-first ball: its groups, and how far each one is.

Two documents, written together and read back together. The ``.groups.json`` holds the
presentations in discovery order and the ``.<moveset>.annotations.json`` holds each
one's proven distance to the origin and the co-optimal moves that get it there -- the
same pair of files ``dataset grow`` + ``dataset annotate`` produce, so every consumer
(the instance store, the supervised labels, the split) reads a ball without knowing it
is one.

What a ball does *not* store is the move adjacency. A grown dataset has to keep it --
its distances are searched for over that graph afterwards -- but here the distances are
known at the moment a group is discovered, and the adjacency is 85% of the bytes.

See :mod:`ac_zero.datasets.ball` for why the construction gives exact distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.annotate import SCHEMA_VERSION as ANNOTATIONS_SCHEMA
from ac_zero.datasets.annotate import annotation_entry
from ac_zero.datasets.expand import NeighbourRecord
from ac_zero.datasets.groups import MOVE_CATALOG, SOURCE_ORIGIN_BALL, SOURCE_TRIVIAL, group_entry
from ac_zero.datasets.groups import SCHEMA_VERSION as GROUPS_SCHEMA
from ac_zero.datasets.io import atomic_write_json
from ac_zero.datasets.json_stream import iter_json_array, read_members_before
from ac_zero.moves.universal import UniversalCatalog, move_set

GENERATOR = "origin_ball_bfs"

# The resume state lives under this top-level member. It is named to sort before
# ``groups``, so `read_members_before` can recover it from a multi-gigabyte file
# without decoding a single group (documents are written with sorted keys).
STATE_KEY = "ball"


@dataclass(slots=True)
class BallNode:
    """One group of the ball: how far it is from the origin, and how it gets closer.

    ``content_hash`` is carried rather than taken from the presentation, which would
    hash (and then cache a string for) every group in the dataset at every checkpoint:
    the expansion worker already computed it.
    """

    presentation: BalancedPresentation
    content_hash: str
    distance: int
    # Every forward move of the set that steps this group one closer to the origin --
    # the co-optimal first moves, which are exactly a supervised policy's target.
    optimal_moves: list[int]


class OriginBall:
    """The groups within a known distance of the trivial group, in discovery order.

    The node list doubles as the breadth-first queue: ``expanded`` is how far into it
    the run has got, everything after that is the frontier, and because a group is only
    ever appended when it is first discovered, the list is ordered by distance.
    """

    def __init__(self, rank: int, moveset: str, nodes: list[BallNode], expanded: int = 0) -> None:
        self.rank = rank
        self.moveset = moveset
        self.nodes = nodes
        self.expanded = expanded
        self._index = {node.content_hash: i for i, node in enumerate(nodes)}
        catalog = UniversalCatalog(rank)
        forward_ids = move_set(moveset, catalog).ids
        # Expanding by ``inv`` from a group reaches one that steps back to it by the
        # forward move ``inv`` inverts -- the co-optimal move recorded on the child.
        self.inverse_ids = frozenset(catalog.inverse_id(i) for i in forward_ids)
        self._forward_for = {catalog.inverse_id(i): i for i in forward_ids}

    @classmethod
    def load_or_seed(
        cls, groups_path: Path, annotations_path: Path, rank: int, moveset: str
    ) -> OriginBall:
        """Reopen an existing ball, or seed a fresh one holding just the origin."""
        if not groups_path.exists():
            origin = BalancedPresentation.standard(rank)
            return cls(rank, moveset, [BallNode(origin, origin.content_hash, 0, [])])
        state = read_members_before(groups_path, "groups").get(STATE_KEY, {})
        stored = str(state.get("moveset", ""))
        if stored != moveset:
            raise ValueError(
                f"{groups_path} is a {stored!r} ball; it cannot be extended under "
                f"{moveset!r} -- generate that one under its own name"
            )
        if not annotations_path.exists():
            raise FileNotFoundError(f"{groups_path} has no distances at {annotations_path}")
        labels = {
            entry["hash"]: entry for entry in iter_json_array(annotations_path, "annotations")
        }
        nodes = [
            BallNode(
                BalancedPresentation.from_letters(rank, entry["relators"]),
                str(entry["hash"]),
                int(labels[entry["hash"]]["distance_to_origin"]),
                list(labels[entry["hash"]]["optimal_moves_to_origin"]),
            )
            for entry in iter_json_array(groups_path, "groups")
        ]
        return cls(rank, moveset, nodes, int(state.get("expanded", 0)))

    def merge(self, parent: int, records: list[NeighbourRecord]) -> int:
        """Record one group's inverse-move neighbours; return the count of new groups.

        A neighbour seen for the first time is one move further from the origin than its
        parent, and that is its exact distance: breadth-first, nothing reaches it sooner.
        One already at that distance gains another co-optimal move -- the same group is
        often one step from several groups in the shell above it.
        """
        added = 0
        distance = self.nodes[parent].distance + 1
        for record in records:
            forward = self._forward_for[record.move_id]
            index = self._index.get(record.child_hash)
            if index is None:
                self._index[record.child_hash] = len(self.nodes)
                self.nodes.append(
                    BallNode(
                        BalancedPresentation.from_letters(self.rank, record.letters),
                        record.child_hash,
                        distance,
                        [forward],
                    )
                )
                added += 1
                continue
            node = self.nodes[index]
            if node.distance == distance and forward not in node.optimal_moves:
                node.optimal_moves.append(forward)
        return added

    @property
    def complete_depth(self) -> int:
        """The deepest distance whose shell is *entirely* in the dataset.

        Groups are expanded in distance order, so every group closer to the origin than
        the first unexpanded one has already contributed its neighbours -- which is what
        makes the shell that first unexpanded group belongs to complete.
        """
        if self.expanded >= len(self.nodes):
            return self.nodes[-1].distance
        return self.nodes[self.expanded].distance

    def max_distance(self) -> int:
        return self.nodes[-1].distance

    def max_length(self) -> int:
        return max(node.presentation.total_length for node in self.nodes)

    def _provenance(self) -> dict[str, Any]:
        return {
            "generator": GENERATOR,
            "moveset": self.moveset,
            "count": len(self.nodes),
            "expanded": self.expanded,
            "complete_depth": self.complete_depth,
            "max_distance": self.max_distance(),
        }

    def write(self, groups_path: Path, annotations_path: Path) -> None:
        """Rewrite both documents: the groups, and the distances that label them."""
        provenance = self._provenance()
        atomic_write_json(
            groups_path,
            {
                STATE_KEY: {"moveset": self.moveset, "expanded": self.expanded},
                "schema_version": GROUPS_SCHEMA,
                "rank": self.rank,
                "move_catalog": MOVE_CATALOG,
                "groups": [
                    group_entry(
                        node.presentation,
                        ac_trivial=True,
                        source=SOURCE_TRIVIAL if node.distance == 0 else SOURCE_ORIGIN_BALL,
                        content_hash=node.content_hash,
                    )
                    for node in self.nodes
                ],
                "provenance": {**provenance, "max_length": self.max_length()},
            },
        )
        atomic_write_json(
            annotations_path,
            {
                "schema_version": ANNOTATIONS_SCHEMA,
                "rank": self.rank,
                "moveset": self.moveset,
                "move_catalog": UniversalCatalog(self.rank).version,
                "annotations": [
                    annotation_entry(
                        node.content_hash,
                        distance_to_origin=node.distance,
                        moves_to_origin=sorted(node.optimal_moves),
                    )
                    for node in self.nodes
                ],
                "provenance": {**provenance, "reached_origin": len(self.nodes)},
            },
        )
