"""Tests for the incremental reader over a dataset document's large array."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ac_zero.datasets import json_stream
from ac_zero.datasets.json_stream import JsonStreamError, iter_json_array


@pytest.fixture(autouse=True)
def tiny_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read a few bytes at a time, so every test straddles the window boundary."""
    monkeypatch.setattr(json_stream, "_CHUNK", 4)


def _write(path: Path, document: object, indent: int | None = None) -> Path:
    path.write_text(json.dumps(document, indent=indent), encoding="utf-8")
    return path


def test_yields_elements_of_a_compact_document(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {"groups": [{"a": 1}, {"a": 2}, {"a": 3}]})
    assert list(iter_json_array(path, "groups")) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_yields_elements_of_a_pretty_printed_document(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {"groups": [{"a": 1}, {"a": 2}]}, indent=2)
    assert list(iter_json_array(path, "groups")) == [{"a": 1}, {"a": 2}]


def test_skips_members_preceding_the_target_array(tmp_path: Path) -> None:
    # The real group files sort `groups` first; the test fixtures do not. Both work.
    document = {"schema_version": "v1", "rank": 2, "groups": [{"a": 1}], "trailing": {"x": [1, 2]}}
    path = _write(tmp_path / "d.json", document)
    assert list(iter_json_array(path, "groups")) == [{"a": 1}]


def test_reads_an_array_of_scalars_and_nested_containers(tmp_path: Path) -> None:
    elements = [1, "two", None, True, [1, [2, [3]]], {"k": {"n": [4]}}]
    path = _write(tmp_path / "d.json", {"items": elements})
    assert list(iter_json_array(path, "items")) == elements


def test_strings_containing_json_punctuation_do_not_confuse_the_scanner(tmp_path: Path) -> None:
    elements = ['{"not": "json"}', "],[", 'quote\\" brace}', "back\\slash"]
    path = _write(tmp_path / "d.json", {"items": elements})
    assert list(iter_json_array(path, "items")) == elements


def test_multi_digit_numbers_survive_the_window_boundary(tmp_path: Path) -> None:
    # A number touching the end of the window looks complete but is not; the
    # scanner must refill before trusting it. With 4-byte chunks these all split.
    elements = [1, 12, 123456789, -987654321, 1.5, 1e10, 0]
    path = _write(tmp_path / "d.json", {"items": elements})
    assert list(iter_json_array(path, "items")) == elements


def test_a_trailing_number_at_end_of_document_is_complete(tmp_path: Path) -> None:
    (tmp_path / "d.json").write_text('{"items":[1,234567]}', encoding="utf-8")
    assert list(iter_json_array(tmp_path / "d.json", "items")) == [1, 234567]


def test_empty_array_yields_nothing(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {"groups": []})
    assert list(iter_json_array(path, "groups")) == []


def test_elements_are_yielded_lazily(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {"groups": [{"a": 1}, {"a": 2}, {"a": 3}]})
    stream = iter_json_array(path, "groups")
    assert next(stream) == {"a": 1}
    stream.close()


def test_missing_key_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {"rank": 2, "other": [1]})
    with pytest.raises(JsonStreamError, match="no 'groups' array"):
        list(iter_json_array(path, "groups"))


def test_empty_object_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {})
    with pytest.raises(JsonStreamError, match="no 'groups' array"):
        list(iter_json_array(path, "groups"))


def test_non_object_document_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", [1, 2])
    with pytest.raises(JsonStreamError, match=r"expected '\{'"):
        list(iter_json_array(path, "groups"))


def test_target_member_that_is_not_an_array_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.json", {"groups": {"a": 1}})
    with pytest.raises(JsonStreamError, match=r"expected '\['"):
        list(iter_json_array(path, "groups"))


def test_truncated_document_raises(tmp_path: Path) -> None:
    (tmp_path / "d.json").write_text('{"groups": [{"a": 1}, {"a":', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_json_array(tmp_path / "d.json", "groups"))
