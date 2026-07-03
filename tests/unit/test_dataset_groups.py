"""Tests for group-dataset generation (``dataset grow``) and its schema."""

from __future__ import annotations

import json
from pathlib import Path

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.groups import MOVE_CATALOG, SCHEMA_VERSION
from ac_zero.datasets.grow import GrowConfig, grow_dataset


def _grow(path: Path, target: int, **kwargs: object) -> dict:
    grow_dataset(path, GrowConfig(rank=2, target=target, total_length_cap=10, workers=1, **kwargs))
    return json.loads(path.read_text())


def test_grow_writes_minimal_group_schema(tmp_path: Path) -> None:
    data = _grow(tmp_path / "toy.groups.json", target=30)
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["move_catalog"] == MOVE_CATALOG
    entry = data["groups"][0]
    allowed = {"hash", "rank", "ac_trivial", "source", "relators", "total_length", "transitions"}
    assert set(entry) <= allowed
    # The minimal form drops presentation_id / human_relators / generator_names.
    assert "human_relators" not in entry and "predecessors" not in entry


def test_grow_seeds_the_trivial_root(tmp_path: Path) -> None:
    data = _grow(tmp_path / "toy.groups.json", target=10)
    roots = [e for e in data["groups"] if e["source"] == "trivial"]
    assert len(roots) == 1
    assert roots[0]["hash"] == BalancedPresentation.standard(2).content_hash
    assert roots[0]["ac_trivial"] is True


def test_every_grown_group_is_ac_trivial(tmp_path: Path) -> None:
    data = _grow(tmp_path / "toy.groups.json", target=40)
    assert all(e["ac_trivial"] is True for e in data["groups"])


def test_transitions_are_complete_and_point_at_nodes(tmp_path: Path) -> None:
    data = _grow(tmp_path / "toy.groups.json", target=40)
    hashes = {e["hash"] for e in data["groups"]}
    expanded = [e for e in data["groups"] if "transitions" in e]
    assert expanded, "at least the root should be expanded"
    for entry in expanded:
        for move_id, target in entry["transitions"].items():
            assert move_id.isdigit()  # integer move ids serialized as strings
            assert target != entry["hash"]  # no-op moves are dropped
            assert target in hashes  # within-cap neighbours are always nodes


def test_grow_only_ever_grows_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "toy.groups.json"
    before = {e["hash"] for e in _grow(path, target=20)["groups"]}
    after = {e["hash"] for e in _grow(path, target=20)["groups"]}
    assert before <= after
    # Still exactly one trivial root after resuming.
    data = json.loads(path.read_text())
    assert sum(1 for e in data["groups"] if e["source"] == "trivial") == 1
