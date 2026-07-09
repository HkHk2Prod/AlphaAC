"""Tests for the memory-mapped sidecar built from a grown group dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets import instance_store
from ac_zero.datasets.instance_store import UNKNOWN, InstanceStore, build, sidecar_path

# Two digests sharing their leading 8 bytes: the prefix index cannot separate
# them, so the lookup must fall back to comparing all 32.
COLLIDING = ("aa" * 8 + "01" * 24, "aa" * 8 + "02" * 24)


def _relators(index: int) -> list[list[int]]:
    return [[1] * (index + 1), [2, 1, -2]]


def _write_groups(path: Path, hashes: list[str], rank: int = 2) -> Path:
    groups = [
        {"hash": digest, "rank": rank, "relators": _relators(index)}
        for index, digest in enumerate(hashes)
    ]
    path.write_text(json.dumps({"rank": rank, "groups": groups}), encoding="utf-8")
    return path


def _write_annotations(path: Path, distances: dict[str, int | None]) -> Path:
    annotations = [{"hash": h, "distance_to_origin": d} for h, d in distances.items()]
    path.write_text(json.dumps({"annotations": annotations}), encoding="utf-8")
    return path


def _digests(count: int) -> list[str]:
    return [f"{index:064x}" for index in range(1, count + 1)]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return _write_groups(tmp_path / "train.groups.json", _digests(4))


def test_open_builds_the_sidecar_beside_the_groups_file(dataset: Path) -> None:
    store = InstanceStore.open(dataset, None)
    assert store.path == sidecar_path(dataset) == dataset.parent / "train.groups.json.instances"
    assert store.path.is_file()
    assert not list(dataset.parent.glob("*.tmp"))


def test_presentations_round_trip_through_the_mapping(dataset: Path) -> None:
    store = InstanceStore.open(dataset, None)
    assert (store.rank, store.count) == (2, 4)
    for index in range(4):
        expected = BalancedPresentation.from_letters(2, _relators(index))
        assert store.presentation(index).content_hash == expected.content_hash


def test_without_annotations_there_are_no_distances_or_potentials(dataset: Path) -> None:
    store = InstanceStore.open(dataset, None)
    assert store.distances is None
    assert store.potentials == {}


def test_distances_follow_group_order_with_unknowns_marked(tmp_path: Path) -> None:
    hashes = _digests(4)
    dataset = _write_groups(tmp_path / "train.groups.json", hashes)
    # Annotations are written out of group order, and one group is unresolved.
    annotations = _write_annotations(
        tmp_path / "train.annotations.json",
        {hashes[2]: 7, hashes[0]: 3, hashes[3]: None, hashes[1]: 5},
    )
    store = InstanceStore.open(dataset, annotations)
    assert store.distances is not None
    assert list(store.distances) == [3, 5, 7, UNKNOWN]


def test_annotations_for_absent_groups_are_ignored(tmp_path: Path) -> None:
    hashes = _digests(2)
    dataset = _write_groups(tmp_path / "train.groups.json", hashes)
    annotations = _write_annotations(
        tmp_path / "train.annotations.json", {hashes[0]: 1, "f" * 64: 9}
    )
    store = InstanceStore.open(dataset, annotations)
    assert store.potentials == {hashes[0]: 1}


def test_potentials_expose_the_annotated_groups(tmp_path: Path) -> None:
    hashes = _digests(3)
    dataset = _write_groups(tmp_path / "train.groups.json", hashes)
    annotations = _write_annotations(
        tmp_path / "train.annotations.json", {hashes[0]: 4, hashes[1]: 0, hashes[2]: None}
    )
    potentials = InstanceStore.open(dataset, annotations).potentials

    assert potentials == {hashes[0]: 4, hashes[1]: 0}
    assert len(potentials) == 2
    assert sorted(potentials) == sorted([hashes[0], hashes[1]])
    assert potentials[hashes[0]] == 4
    assert potentials.get(hashes[2]) is None


@pytest.mark.parametrize("key", ["e" * 64, "not hex at all!" + "0" * 49, "abcd", ""])
def test_potentials_reject_absent_and_malformed_hashes(tmp_path: Path, key: str) -> None:
    hashes = _digests(1)
    dataset = _write_groups(tmp_path / "train.groups.json", hashes)
    annotations = _write_annotations(tmp_path / "train.annotations.json", {hashes[0]: 4})
    potentials = InstanceStore.open(dataset, annotations).potentials

    assert potentials.get(key) is None
    with pytest.raises(KeyError):
        potentials[key]


def test_colliding_digest_prefixes_resolve_to_their_own_distances(tmp_path: Path) -> None:
    dataset = _write_groups(tmp_path / "train.groups.json", list(COLLIDING))
    annotations = _write_annotations(
        tmp_path / "train.annotations.json", {COLLIDING[0]: 11, COLLIDING[1]: 22}
    )
    store = InstanceStore.open(dataset, annotations)

    assert list(store.distances) == [11, 22]  # type: ignore[arg-type]
    assert store.potentials == {COLLIDING[0]: 11, COLLIDING[1]: 22}
    # A third digest sharing the same prefix is still absent from the map.
    assert store.potentials.get("aa" * 8 + "03" * 24) is None


def test_a_current_sidecar_is_reused_rather_than_rebuilt(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = _count_builds(monkeypatch)
    InstanceStore.open(dataset, None)
    assert builds == [1]
    InstanceStore.open(dataset, None)
    assert builds == [1]


def test_a_changed_groups_file_rebuilds_the_sidecar(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = _count_builds(monkeypatch)
    assert InstanceStore.open(dataset, None).count == 4

    _write_groups(dataset, _digests(6))
    store = InstanceStore.open(dataset, None)
    assert builds == [2]
    assert store.count == 6


def test_adding_annotations_rebuilds_the_sidecar(tmp_path: Path) -> None:
    hashes = _digests(2)
    dataset = _write_groups(tmp_path / "train.groups.json", hashes)
    assert InstanceStore.open(dataset, None).distances is None

    annotations = _write_annotations(tmp_path / "train.annotations.json", {hashes[1]: 6})
    store = InstanceStore.open(dataset, annotations)
    assert list(store.distances) == [UNKNOWN, 6]  # type: ignore[arg-type]


def test_a_corrupt_sidecar_is_rebuilt(dataset: Path) -> None:
    build(dataset, None)
    sidecar_path(dataset).write_bytes(b"not a sidecar")
    assert InstanceStore.open(dataset, None).count == 4


def test_a_sidecar_from_another_schema_is_rebuilt(dataset: Path) -> None:
    build(dataset, None)
    # Same length, so only the schema check -- not the header's own framing -- can
    # reject it.
    written = sidecar_path(dataset).read_bytes()
    sidecar_path(dataset).write_bytes(
        written.replace(b"aczero-instances-v1", b"aczero-instances-v9")
    )
    assert InstanceStore.open(dataset, None).count == 4


def test_a_dataset_with_no_groups_is_rejected(tmp_path: Path) -> None:
    dataset = _write_groups(tmp_path / "train.groups.json", [])
    with pytest.raises(ValueError, match="no groups"):
        InstanceStore.open(dataset, None)


def test_an_unbalanced_group_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "train.groups.json"
    groups = [{"hash": _digests(1)[0], "rank": 2, "relators": [[1]]}]
    dataset.write_text(json.dumps({"groups": groups}), encoding="utf-8")
    with pytest.raises(ValueError, match="1 relators, expected 2"):
        InstanceStore.open(dataset, None)


@pytest.mark.parametrize("rank", [0, 128])
def test_an_unsupported_rank_is_rejected(tmp_path: Path, rank: int) -> None:
    dataset = tmp_path / "train.groups.json"
    groups = [{"hash": _digests(1)[0], "rank": rank, "relators": [[1]] * max(rank, 1)}]
    dataset.write_text(json.dumps({"groups": groups}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the supported range"):
        InstanceStore.open(dataset, None)


def _count_builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace `build` with a counting passthrough, returning its call tally."""
    calls = [0]
    real = instance_store.build

    def counted(*args: Any) -> None:
        calls[0] += 1
        real(*args)

    monkeypatch.setattr(instance_store, "build", counted)
    return calls
