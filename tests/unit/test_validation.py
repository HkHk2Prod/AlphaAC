"""Tests for validating the group and annotation dataset schemas."""

from __future__ import annotations

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.groups import MOVE_CATALOG, SCHEMA_VERSION, group_entry
from ac_zero.datasets.validation import validate_mapping

_ANNOTATIONS_SCHEMA = "aczero-annotations-v1"


def _group_document(entries: list) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "rank": 2,
        "move_catalog": MOVE_CATALOG,
        "groups": entries,
    }


def _entry() -> dict:
    return group_entry(BalancedPresentation.standard(2), ac_trivial=True, source="trivial")


def test_valid_group_document_passes() -> None:
    report = validate_mapping(_group_document([_entry()]))
    assert report.ok
    assert report.entries == 1


def test_unknown_schema_version_fails() -> None:
    document = _group_document([_entry()])
    document["schema_version"] = "aczero-dataset-v9"
    assert not validate_mapping(document).ok


def test_hash_mismatch_is_detected() -> None:
    entry = _entry()
    entry["hash"] = "deadbeef"
    report = validate_mapping(_group_document([entry]))
    assert not report.ok
    assert any("hash" in error for error in report.errors)


def test_total_length_mismatch_is_detected() -> None:
    entry = _entry()
    entry["total_length"] = 999
    report = validate_mapping(_group_document([entry]))
    assert any("total_length" in error for error in report.errors)


def test_bad_ac_trivial_is_rejected() -> None:
    entry = _entry()
    entry["ac_trivial"] = "maybe"
    assert not validate_mapping(_group_document([entry])).ok


def test_groups_must_be_a_list() -> None:
    assert not validate_mapping({"schema_version": SCHEMA_VERSION, "rank": 2}).ok


def test_valid_annotation_document_passes() -> None:
    document = {
        "schema_version": _ANNOTATIONS_SCHEMA,
        "rank": 2,
        "moveset": "universal",
        "annotations": [
            {
                "hash": "abc",
                "distance_to_origin": 3,
                "optimal_moves_to_origin": [1, 4],
                "distance_to_shorter": None,
                "optimal_moves_to_shorter": [],
                "shorter_proven": True,
                "optimal": True,
            }
        ],
    }
    assert validate_mapping(document).ok


def test_annotation_bad_distance_is_rejected() -> None:
    document = {
        "schema_version": _ANNOTATIONS_SCHEMA,
        "rank": 2,
        "annotations": [
            {
                "hash": "abc",
                "distance_to_origin": -1,
                "optimal_moves_to_origin": [],
                "distance_to_shorter": None,
                "optimal_moves_to_shorter": [],
                "shorter_proven": False,
                "optimal": False,
            }
        ],
    }
    report = validate_mapping(document)
    assert not report.ok
    assert any("distance_to_origin" in error for error in report.errors)
