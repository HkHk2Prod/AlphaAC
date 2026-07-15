"""The persistent state of a closest-first ball: its groups, and how far each one is.

Two documents, written together and read back together. The ``.groups.json`` holds the
presentations in discovery order and the ``.<moveset>.annotations.json`` holds each
one's proven distance to the origin and the co-optimal moves that get it there -- the
same pair of files ``dataset grow`` + ``dataset annotate`` produce, so every consumer
(the instance store, the supervised labels, the split) reads a ball without knowing it
is one.

In memory the groups live in compact columns rather than as presentation objects (see
:mod:`ac_zero.datasets.ball_columns`): a rank-2 ball reaches tens of millions of groups,
and one Python object per group runs a machine out of memory long before it runs out of
groups to add. The columns rebuild a presentation only for the group about to be
expanded, and the digest-keyed dedup index is an open-addressing table rather than a
dict of hex strings.

What a ball does *not* store is the move adjacency. A grown dataset has to keep it --
its distances are searched for over that graph afterwards -- but here the distances are
known at the moment a group is discovered, and the adjacency is 85% of the bytes.

See :mod:`ac_zero.datasets.ball` for why the construction gives exact distances.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.annotate import SCHEMA_VERSION as ANNOTATIONS_SCHEMA
from ac_zero.datasets.annotate import annotation_entry
from ac_zero.datasets.ball_columns import BallColumns, DigestIndex
from ac_zero.datasets.expand import NeighbourRecord
from ac_zero.datasets.groups import (
    BOUNDS_KEY,
    MOVE_CATALOG,
    RELATOR_BOUND,
    SOURCE_ORIGIN_BALL,
    SOURCE_TRIVIAL,
    check_relator_bound,
    read_relator_bound,
)
from ac_zero.datasets.groups import SCHEMA_VERSION as GROUPS_SCHEMA
from ac_zero.datasets.io import atomic_write_json
from ac_zero.datasets.json_stream import iter_json_array, read_members_before
from ac_zero.moves.universal import UniversalCatalog, move_set

GENERATOR = "origin_ball_bfs"

# The resume state lives under this top-level member. It is named to sort before
# ``groups``, so `read_members_before` can recover it from a multi-gigabyte file
# without decoding a single group (documents are written with sorted keys).
STATE_KEY = "ball"


class OriginBall:
    """The groups within a known distance of the trivial group, in discovery order.

    The column list doubles as the breadth-first queue: ``expanded`` is how far into
    it the run has got, everything after that is the frontier, and because a group is
    only ever appended when it is first discovered, the list is ordered by distance.
    """

    def __init__(
        self,
        rank: int,
        moveset: str,
        max_relator_length: int = 0,
        expanded: int = 0,
    ) -> None:
        self.rank = rank
        self.moveset = moveset
        self.max_relator_length = max_relator_length
        self.expanded = expanded
        self._columns = BallColumns(rank)
        self._index = DigestIndex()
        catalog = UniversalCatalog(rank)
        forward_ids = move_set(moveset, catalog).ids
        # Expanding by ``inv`` from a group reaches one that steps back to it by the
        # forward move ``inv`` inverts -- the co-optimal move recorded on the child.
        self.inverse_ids = frozenset(catalog.inverse_id(i) for i in forward_ids)
        self._forward_for = {catalog.inverse_id(i): i for i in forward_ids}
        # A group's co-optimal moves are a subset of the set's forward moves, so they
        # pack into a bitmask: bit ``i`` is the ``i``-th forward move in sorted order.
        self._forward_sorted = sorted(forward_ids)
        self._bit_for = {move_id: 1 << i for i, move_id in enumerate(self._forward_sorted)}

    def __len__(self) -> int:
        return len(self._columns)

    @classmethod
    def load_or_seed(
        cls,
        groups_path: Path,
        annotations_path: Path,
        rank: int,
        moveset: str,
        max_relator_length: int = 0,
    ) -> OriginBall:
        """Reopen an existing ball, or seed a fresh one holding just the origin."""
        if not groups_path.exists():
            ball = cls(rank, moveset, max_relator_length)
            origin = BalancedPresentation.standard(rank)
            ball._append(
                [list(relator.letters) for relator in origin.relators],
                bytes.fromhex(origin.content_hash),
                0,
                0,
            )
            return ball
        state = read_members_before(groups_path, "groups").get(STATE_KEY, {})
        stored = str(state.get("moveset", ""))
        if stored != moveset:
            raise ValueError(
                f"{groups_path} is a {stored!r} ball; it cannot be extended under "
                f"{moveset!r} -- generate that one under its own name"
            )
        check_relator_bound(
            groups_path, read_relator_bound(groups_path), max_relator_length, "ball"
        )
        if not annotations_path.exists():
            raise FileNotFoundError(f"{groups_path} has no distances at {annotations_path}")
        ball = cls(rank, moveset, max_relator_length, expanded=int(state.get("expanded", 0)))
        # The two documents are written together from the one column list, so they are
        # read back in lockstep: pairing them through a hash-keyed dict of every
        # annotation would hold a second copy of the dataset beside the ball being built.
        pairs = zip(
            iter_json_array(groups_path, "groups"),
            iter_json_array(annotations_path, "annotations"),
            strict=True,
        )
        for group, label in pairs:
            ball._load_pair(group, label)
        return ball

    def _load_pair(self, group: dict[str, Any], label: dict[str, Any]) -> None:
        """Append one group from the group entry and the annotation entry beside it."""
        if group["hash"] != label["hash"]:
            raise ValueError(
                "the groups and their distances have drifted out of order: "
                f"group {group['hash']} is labelled {label['hash']}. Both files are "
                "written by the same run -- regenerate them as the pair they are."
            )
        mask = 0
        for move_id in label["optimal_moves_to_origin"]:
            mask |= self._bit_for[move_id]
        self._append(
            group["relators"],
            bytes.fromhex(group["hash"]),
            int(label["distance_to_origin"]),
            mask,
        )

    def _append(
        self, relators: Iterable[Sequence[int]], digest: bytes, distance: int, moves_mask: int
    ) -> int:
        index = self._columns.append(relators, digest, distance, moves_mask)
        self._index.insert(digest, index)
        return index

    def presentation(self, index: int) -> BalancedPresentation:
        """Rebuild group ``index``'s presentation from its columns, for expansion."""
        return BalancedPresentation.from_letters(self.rank, self._columns.relators_at(index))

    def merge(self, parent: int, records: list[NeighbourRecord]) -> int:
        """Record one group's inverse-move neighbours; return the count of new groups.

        A neighbour seen for the first time is one move further from the origin than its
        parent, and that is its exact distance: breadth-first, nothing reaches it sooner.
        One already at that distance gains another co-optimal move -- the same group is
        often one step from several groups in the shell above it.
        """
        columns = self._columns
        added = 0
        distance = columns.distance_at(parent) + 1
        for record in records:
            bit = self._bit_for[self._forward_for[record.move_id]]
            digest = bytes.fromhex(record.child_hash)
            index = self._index.get(digest)
            if index is None:
                self._append(record.letters, digest, distance, bit)
                added += 1
            elif columns.distance_at(index) == distance:
                columns.or_move(index, bit)
        return added

    @property
    def complete_depth(self) -> int:
        """The deepest distance whose shell is *entirely* in the dataset.

        Groups are expanded in distance order, so every group closer to the origin than
        the first unexpanded one has already contributed its neighbours -- which is what
        makes the shell that first unexpanded group belongs to complete.
        """
        columns = self._columns
        if self.expanded >= len(columns):
            return columns.distance_at(len(columns) - 1)
        return columns.distance_at(self.expanded)

    def max_distance(self) -> int:
        return self._columns.distance_at(len(self._columns) - 1)

    def max_length(self) -> int:
        return self._columns.max_length()

    def _moves_to_origin(self, index: int) -> list[int]:
        """Decode a group's co-optimal forward moves from its bitmask, sorted."""
        mask = self._columns.moves_at(index)
        return [move for i, move in enumerate(self._forward_sorted) if mask >> i & 1]

    def _provenance(self) -> dict[str, Any]:
        return {
            "generator": GENERATOR,
            "moveset": self.moveset,
            "max_relator_length": self.max_relator_length,
            "count": len(self),
            "expanded": self.expanded,
            "complete_depth": self.complete_depth,
            "max_distance": self.max_distance(),
        }

    def _group_entries(self) -> Any:
        columns = self._columns
        for index in range(len(columns)):
            distance = columns.distance_at(index)
            yield {
                "hash": columns.digest_at(index).hex(),
                "rank": self.rank,
                "ac_trivial": True,
                "source": SOURCE_TRIVIAL if distance == 0 else SOURCE_ORIGIN_BALL,
                "relators": columns.relators_at(index),
                "total_length": columns.total_length_at(index),
            }

    def _annotation_entries(self) -> Any:
        columns = self._columns
        for index in range(len(columns)):
            yield annotation_entry(
                columns.digest_at(index).hex(),
                distance_to_origin=columns.distance_at(index),
                moves_to_origin=self._moves_to_origin(index),
            )

    def write(self, groups_path: Path, annotations_path: Path) -> None:
        """Rewrite both documents: the groups, and the distances that label them.

        Both arrays are handed to the writer as generators, so a checkpoint encodes one
        entry at a time rather than materializing millions of them beside the ball it is
        already holding.
        """
        provenance = self._provenance()
        atomic_write_json(
            groups_path,
            {
                STATE_KEY: {"moveset": self.moveset, "expanded": self.expanded},
                BOUNDS_KEY: {RELATOR_BOUND: self.max_relator_length},
                "schema_version": GROUPS_SCHEMA,
                "rank": self.rank,
                "move_catalog": MOVE_CATALOG,
                "groups": self._group_entries(),
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
                "annotations": self._annotation_entries(),
                "provenance": {**provenance, "reached_origin": len(self)},
            },
        )
