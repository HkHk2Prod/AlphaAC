"""Tests for the deterministic train/val/test split of a group dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.datasets.split import SplitConfig, assign, split_path, write_split


def _dataset(tmp_path: Path, target: int = 400) -> Path:
    path = tmp_path / "toy.groups.json"
    grow_dataset(path, GrowConfig(rank=2, target=target, total_length_cap=10, workers=1))
    return path


def test_split_path_naming(tmp_path: Path) -> None:
    assert split_path(tmp_path / "train.groups.json").name == "train.split.json"
    assert split_path(tmp_path / "other.json").name == "other.split.json"


def test_assignment_depends_only_on_the_content_hash() -> None:
    config = SplitConfig()
    digest = "ab" * 32
    assert assign(digest, config) == assign(digest, config)
    # A different salt is a different split -- that is the whole point of having one.
    other = SplitConfig(salt="second-draw")
    reshuffled = [
        d for d in (f"{i:064x}" for i in range(500)) if assign(d, config) != assign(d, other)
    ]
    assert reshuffled


def test_ratios_come_out_on_the_population() -> None:
    counts = Counter(assign(f"{index:064x}", SplitConfig()) for index in range(20_000))
    assert counts["train"] / 20_000 == pytest.approx(0.8, abs=0.02)
    assert counts["val"] / 20_000 == pytest.approx(0.1, abs=0.01)
    assert counts["test"] / 20_000 == pytest.approx(0.1, abs=0.01)


def test_custom_ratios_are_honoured() -> None:
    config = SplitConfig(train=0.5, val=0.25, test=0.25)
    counts = Counter(assign(f"{index:064x}", config) for index in range(20_000))
    assert counts["train"] / 20_000 == pytest.approx(0.5, abs=0.02)


def test_write_split_covers_every_group_exactly_once(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    report = write_split(groups, SplitConfig())

    data = json.loads(Path(report.path).read_text())
    hashes = [entry["hash"] for entry in data["assignments"]]
    stored = {entry["hash"] for entry in json.loads(groups.read_text())["groups"]}
    assert set(hashes) == stored
    assert len(hashes) == len(stored)  # no duplicates
    assert report.total == report.train + report.val + report.test == len(stored)
    assert data["provenance"]["train"] == report.train


def test_a_grown_dataset_keeps_every_existing_assignment(tmp_path: Path) -> None:
    """The database only grows, so re-splitting must not move a group already scored."""
    groups = _dataset(tmp_path, target=200)
    before = {
        entry["hash"]: entry["split"]
        for entry in json.loads(Path(write_split(groups, SplitConfig()).path).read_text())[
            "assignments"
        ]
    }

    grow_dataset(groups, GrowConfig(rank=2, target=200, total_length_cap=10, workers=1))
    after = {
        entry["hash"]: entry["split"]
        for entry in json.loads(Path(write_split(groups, SplitConfig()).path).read_text())[
            "assignments"
        ]
    }

    assert len(after) > len(before)  # the grow really did add groups
    assert all(after[digest] == split for digest, split in before.items())


def test_ratios_must_be_a_valid_distribution() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        SplitConfig(train=0.5, val=0.1, test=0.1).validate()
    with pytest.raises(ValueError, match="positive share"):
        SplitConfig(train=1.0, val=0.0, test=0.0).validate()
    with pytest.raises(ValueError, match="non-negative"):
        SplitConfig(train=1.2, val=-0.1, test=-0.1).validate()


def test_an_empty_dataset_cannot_be_split(tmp_path: Path) -> None:
    empty = tmp_path / "empty.groups.json"
    empty.write_text(json.dumps({"rank": 2, "groups": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no groups"):
        write_split(empty, SplitConfig())
