"""Tests for the supervised label sidecar: per-move distance deltas, split, capacity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.annotate import AnnotateConfig, annotate, annotation_path
from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.datasets.instance_store import InstanceStore
from ac_zero.datasets.split import SplitConfig, split_path, write_split
from ac_zero.datasets.supervised_store import DELTA_UNKNOWN, SupervisedStore, sidecar_path
from ac_zero.moves.universal import moveset_catalog

MOVESET = "strict-ac"


def _labelled(tmp_path: Path, target: int = 300, capacity: int = 0) -> tuple[Path, SupervisedStore]:
    groups = tmp_path / "toy.groups.json"
    grow_dataset(groups, GrowConfig(rank=2, target=target, total_length_cap=10, workers=1))
    annotate(groups, AnnotateConfig(moveset=MOVESET, workers=1))
    write_split(groups, SplitConfig())
    store = SupervisedStore.open(
        groups, annotation_path(groups, MOVESET), split_path(groups), MOVESET, capacity
    )
    return groups, store


def _annotations(groups: Path) -> dict[str, dict]:
    data = json.loads(annotation_path(groups, MOVESET).read_text())
    return {entry["hash"]: entry for entry in data["annotations"]}


def _group_hashes(groups: Path) -> list[str]:
    return [entry["hash"] for entry in json.loads(groups.read_text())["groups"]]


def test_sidecar_naming(tmp_path: Path) -> None:
    path = sidecar_path(tmp_path / "train.groups.json", "strict-ac")
    assert path.name == "train.groups.json.strict-ac.supervised"


def test_shape_and_capacity_match_the_dataset(tmp_path: Path) -> None:
    groups, store = _labelled(tmp_path)
    hashes = _group_hashes(groups)

    assert store.count == len(hashes)
    assert store.deltas.shape == (len(hashes), store.actions)
    assert store.actions == 3 * store.rank**2  # the strict-AC catalog
    # The capacity is the longest relator in the data, so nothing is truncated.
    longest = max(
        len(relator)
        for entry in json.loads(groups.read_text())["groups"]
        for relator in entry["relators"]
    )
    assert store.longest_relator == longest


def test_delta_is_the_change_in_distance_to_the_origin(tmp_path: Path) -> None:
    """Every delta equals `distance(child) - distance(group)`, the child re-derived here.

    The store applies the moves rather than reading a stored adjacency, so this checks
    it against the moves: a no-op or a child the dataset has no distance for is
    unlabelled, and everything else is the real difference.
    """
    groups, store = _labelled(tmp_path)
    annotations = _annotations(groups)
    entries = json.loads(groups.read_text())["groups"]
    catalog = moveset_catalog(MOVESET, store.rank)

    checked = 0
    for row, entry in enumerate(entries):
        own = annotations[entry["hash"]]["distance_to_origin"]
        presentation = BalancedPresentation.from_letters(store.rank, entry["relators"])
        for action, move in enumerate(catalog.moves):
            child = move.apply(presentation)
            child_annotation = annotations.get(child.content_hash, {})
            child_distance = child_annotation.get("distance_to_origin")
            if own is None or child_distance is None or child.content_hash == entry["hash"]:
                assert store.deltas[row, action] == DELTA_UNKNOWN
                continue
            assert store.deltas[row, action] == child_distance - own
            checked += 1
    assert checked > 0


def test_a_frontier_group_is_labelled_from_its_moves(tmp_path: Path) -> None:
    """A group the grow never expanded still gets labels: the store applies the moves.

    Under the old join these rows were empty -- no stored transitions, no label -- which
    silently discarded every group on the frontier, the bulk of a freshly grown dataset.
    """
    groups, store = _labelled(tmp_path)
    entries = json.loads(groups.read_text())["groups"]
    unexpanded = [row for row, entry in enumerate(entries) if not entry.get("transitions")]
    assert unexpanded

    labelled = [row for row in unexpanded if (store.deltas[row] != DELTA_UNKNOWN).any()]
    assert labelled


def test_the_capacity_unlabels_the_moves_the_environment_would_mask(tmp_path: Path) -> None:
    """A move whose child overflows `max_relator_tokens` is no label: the env masks it."""
    groups, roomy = _labelled(tmp_path, capacity=0)
    tight = SupervisedStore.open(
        groups, annotation_path(groups, MOVESET), split_path(groups), MOVESET, 4
    )
    assert tight.max_relator_tokens == 4

    # Every move the tight capacity still labels is one the roomy one labelled the same
    # way, and it drops at least one that overflows.
    labelled = tight.deltas != DELTA_UNKNOWN
    assert bool((tight.deltas[labelled] == roomy.deltas[labelled]).all())
    assert int(labelled.sum()) < int((roomy.deltas != DELTA_UNKNOWN).sum())
    # ...and no group it trains on has a relator the encoder could not hold.
    rows = tight.trainable("train")
    assert rows.size
    assert bool((tight.longest[rows] <= 4).all())


def test_every_trainable_group_has_at_least_one_descent_move(tmp_path: Path) -> None:
    """The label is learnable: a shortest path exists out of every group we train on.

    A group at distance `d > 0` was reached from the origin over the stored graph, so
    one of its own moves steps to a group at `d - 1`. Without this the target would be
    asking the model for a move that is not there.
    """
    _, store = _labelled(tmp_path)
    rows = store.trainable("train")
    assert rows.size
    assert bool((store.deltas[rows] == -1).any(axis=1).all())


def test_trainable_excludes_the_origin_and_the_unlabelled(tmp_path: Path) -> None:
    groups, store = _labelled(tmp_path)
    origin = BalancedPresentation.standard(2).content_hash
    origin_row = _group_hashes(groups).index(origin)

    rows = np.concatenate([store.trainable(split) for split in ("train", "val", "test")])
    assert origin_row not in set(rows.tolist())  # distance 0: the goal, not a problem
    assert bool((store.distances[rows] > 0).all())
    # A group whose every move lands somewhere the dataset has no distance for is
    # unlabelled: its target would be uniform over nothing.
    assert bool((store.deltas[rows] != DELTA_UNKNOWN).any(axis=1).all())


def test_splits_partition_the_groups(tmp_path: Path) -> None:
    _, store = _labelled(tmp_path)
    rows = [set(store.trainable(split).tolist()) for split in ("train", "val", "test")]
    train, val, test = rows
    assert not train & val and not train & test and not val & test
    assert len(train) > len(val) and len(train) > len(test)


def test_rows_align_with_the_instance_store(tmp_path: Path) -> None:
    """Row `i` of the labels is group `i` of the presentations -- the pairing training uses."""
    groups, store = _labelled(tmp_path)
    instances = InstanceStore.open(groups, annotation_path(groups, MOVESET))
    hashes = _group_hashes(groups)

    for row in (0, store.count // 2, store.count - 1):
        assert instances.presentation(row).content_hash == hashes[row]


def test_an_unknown_split_is_rejected(tmp_path: Path) -> None:
    _, store = _labelled(tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        store.trainable("validation")


def test_the_sidecar_is_rebuilt_when_a_source_changes(tmp_path: Path) -> None:
    groups, store = _labelled(tmp_path, target=200)
    before = store.count

    grow_dataset(groups, GrowConfig(rank=2, target=200, total_length_cap=10, workers=1))
    annotate(groups, AnnotateConfig(moveset=MOVESET, workers=1))
    write_split(groups, SplitConfig())
    rebuilt = SupervisedStore.open(
        groups, annotation_path(groups, MOVESET), split_path(groups), MOVESET, 0
    )
    assert rebuilt.count > before
