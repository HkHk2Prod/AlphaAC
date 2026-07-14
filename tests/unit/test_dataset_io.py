"""Tests for the streaming atomic JSON writer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from ac_zero.datasets.io import atomic_write_json
from ac_zero.datasets.json_stream import iter_json_array, read_members_before


def test_writes_an_object_with_sorted_members(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    atomic_write_json(path, {"rank": 2, "count": 3, "nested": {"b": 1, "a": 2}})
    assert json.loads(path.read_text()) == {"rank": 2, "count": 3, "nested": {"b": 1, "a": 2}}
    assert list(json.loads(path.read_text())) == ["count", "nested", "rank"]


def test_streams_an_iterator_member(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    atomic_write_json(path, {"groups": iter([{"a": 1}, {"a": 2}]), "rank": 2})
    assert json.loads(path.read_text()) == {"groups": [{"a": 1}, {"a": 2}], "rank": 2}
    assert list(iter_json_array(path, "groups")) == [{"a": 1}, {"a": 2}]


def test_encodes_one_element_at_a_time(tmp_path: Path) -> None:
    """The whole array must never exist at once: that is the point of streaming it."""
    live = 0

    def entries() -> Iterator[dict[str, int]]:
        nonlocal live
        for index in range(100):
            live += 1  # an element the writer has not yet consumed would keep this rising
            yield {"index": index}
            live -= 1

    atomic_write_json(tmp_path / "d.json", {"groups": entries()})
    assert live == 0
    assert len(json.loads((tmp_path / "d.json").read_text())["groups"]) == 100


def test_streams_an_empty_iterator(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    atomic_write_json(path, {"groups": iter([])})
    assert json.loads(path.read_text()) == {"groups": []}
    assert list(iter_json_array(path, "groups")) == []


def test_a_member_before_the_array_is_readable_without_decoding_it(tmp_path: Path) -> None:
    """What lets a multi-gigabyte ball resume from its state member alone."""
    path = tmp_path / "d.json"
    atomic_write_json(
        path, {"ball": {"expanded": 7}, "groups": iter([{"a": 1}]), "schema_version": "v1"}
    )
    assert read_members_before(path, "groups") == {"ball": {"expanded": 7}}


def test_a_failed_write_leaves_no_file_behind(tmp_path: Path) -> None:
    def entries() -> Iterator[dict[str, int]]:
        yield {"a": 1}
        raise RuntimeError("expansion died mid-checkpoint")

    with pytest.raises(RuntimeError):
        atomic_write_json(tmp_path / "d.json", {"groups": entries()})
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_the_previous_document_intact(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    atomic_write_json(path, {"groups": iter([{"a": 1}])})

    def entries() -> Iterator[dict[str, int]]:
        yield {"a": 2}
        raise RuntimeError("the disk filled up half way through the checkpoint")

    with pytest.raises(RuntimeError):
        atomic_write_json(path, {"groups": entries()})
    assert json.loads(path.read_text()) == {"groups": [{"a": 1}]}
    assert [entry.name for entry in tmp_path.iterdir()] == ["d.json"]
