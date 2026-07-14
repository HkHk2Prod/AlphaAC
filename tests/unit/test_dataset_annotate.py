"""Tests for per-move-set distance annotation over a group dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.annotate import (
    AnnotateConfig,
    _init_worker,
    _shorter_for,
    annotate,
    annotation_path,
)
from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.moves.universal import UniversalCatalog, move_set


def _dataset(tmp_path: Path, target: int = 60) -> Path:
    path = tmp_path / "toy.groups.json"
    grow_dataset(path, GrowConfig(rank=2, target=target, max_relator_length=6, workers=1))
    return path


def _annotate(path: Path, moveset: str) -> dict[str, dict[str, Any]]:
    annotate(path, AnnotateConfig(moveset=moveset, workers=1))
    data = json.loads(annotation_path(path, moveset).read_text())
    return {a["hash"]: a for a in data["annotations"]}


def test_annotation_path_naming(tmp_path: Path) -> None:
    groups = tmp_path / "train.groups.json"
    assert annotation_path(groups, "strict-ac").name == "train.strict-ac.annotations.json"


def test_origin_has_distance_zero_and_no_moves(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    ann = _annotate(path, "universal")
    origin = BalancedPresentation.standard(2).content_hash
    assert ann[origin]["distance_to_origin"] == 0
    assert ann[origin]["optimal_moves_to_origin"] == []
    assert ann[origin]["optimal"] is True


def test_optimal_moves_step_toward_origin(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    data = json.loads(path.read_text())
    graph = {e["hash"]: e for e in data["groups"]}
    catalog = UniversalCatalog(2)
    move_ids = move_set("universal", catalog).ids
    ann = _annotate(path, "universal")
    checked = 0
    for h, entry in ann.items():
        distance = entry["distance_to_origin"]
        if not distance:  # skip origin and unreached
            continue
        transitions = graph[h].get("transitions")
        if transitions is None:  # frontier group: verified via BFS predecessors, not edges
            continue
        for move_id in entry["optimal_moves_to_origin"]:
            assert move_id in move_ids
            target = transitions[str(move_id)]
            assert ann[target]["distance_to_origin"] == distance - 1
            checked += 1
    assert checked > 0


def test_strict_ac_reaches_a_subset_of_universal(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    universal = _annotate(path, "universal")
    strict = _annotate(path, "strict-ac")
    uni_reached = {h for h, a in universal.items() if a["distance_to_origin"] is not None}
    strict_reached = {h for h, a in strict.items() if a["distance_to_origin"] is not None}
    assert strict_reached <= uni_reached
    # The universal set is a superset, so distances can only shrink.
    for h in strict_reached:
        assert strict[h]["distance_to_origin"] >= universal[h]["distance_to_origin"]


def test_distance_to_shorter_reaches_a_smaller_group(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    data = json.loads(path.read_text())
    lengths = {e["hash"]: e["total_length"] for e in data["groups"]}
    ann = _annotate(path, "universal")
    found = [a for a in ann.values() if isinstance(a["distance_to_shorter"], int)]
    assert found, "some group should be able to shorten"
    for entry in found:
        assert entry["shorter_proven"] is True
        assert entry["distance_to_shorter"] >= 1
        # The trivial root (length 2) is a global minimum -> no shorter group.
    origin = BalancedPresentation.standard(2).content_hash
    assert ann[origin]["distance_to_shorter"] is None
    assert min(lengths.values()) == lengths[origin]


def test_shorter_search_max_depth_zero_is_unbounded() -> None:
    """max_depth=0 removes the depth cut, so a multi-move shortening still settles."""
    # A(3) -> B(3) -> C(2): the only shortening is two moves away from A.
    adjacency = {"A": {0: "B"}, "B": {0: "C", 1: "A"}}
    lengths = {"A": 3, "B": 3, "C": 2}
    moves = frozenset({0, 1})
    # A shallow budget cannot reach C within one layer and reports "not proven".
    _init_worker(adjacency, lengths, moves, 1)
    assert _shorter_for("A") == (None, [], False)
    # max_depth=0 -> unbounded: it reaches C at distance 2 and proves the shortening.
    _init_worker(adjacency, lengths, moves, 0)
    assert _shorter_for("A") == (2, [0], True)


def test_checkpoint_progress_reports_percent_complete(tmp_path: Path) -> None:
    path = _dataset(tmp_path, target=30)
    events: list[dict[str, Any]] = []
    annotate(
        path,
        AnnotateConfig(moveset="universal", workers=1, checkpoint_every=1),
        progress=lambda message, metrics: (
            events.append(metrics) if message == "checkpoint" else None
        ),
    )
    assert events, "expected at least one checkpoint progress event"
    for metrics in events:
        assert metrics["pct_complete"] == round(100 * metrics["computed"] / metrics["total"], 1)
        assert 0 < metrics["pct_complete"] <= 100


def test_resume_skips_already_resolved_groups(tmp_path: Path) -> None:
    path = _dataset(tmp_path, target=30)
    first = annotate(path, AnnotateConfig(moveset="universal", workers=1))
    assert first.computed > 0
    # A second pass recomputes nothing: every group's shorter-distance is settled.
    second = annotate(path, AnnotateConfig(moveset="universal", workers=1))
    assert second.computed == 0
