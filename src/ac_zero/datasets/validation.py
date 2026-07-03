from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation

_GROUPS_SCHEMA = "aczero-groups-v1"
_ANNOTATIONS_SCHEMA = "aczero-annotations-v1"
_TRISTATE = (True, False, None)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating a dataset document against an AC-Zero schema."""

    ok: bool
    entries: int
    errors: list[str] = field(default_factory=list)


def validate_dataset(path: str | Path) -> ValidationReport:
    """Validate a group or annotation dataset JSON file on disk."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationReport(ok=False, entries=0, errors=[f"unreadable dataset: {exc}"])
    return validate_mapping(data)


def validate_mapping(data: Any) -> ValidationReport:
    """Validate an already-parsed dataset document, dispatching on its schema.

    Group documents have their relators reparsed and content hashes recomputed, so
    corrupted relators or stale hashes are caught; annotation documents have their
    distance and move-list fields shape-checked.
    """
    if not isinstance(data, dict):
        return ValidationReport(ok=False, entries=0, errors=["document must be a JSON object"])
    schema = data.get("schema_version")
    if not isinstance(data.get("rank"), int) or data.get("rank", 0) < 1:
        base = ["rank must be a positive integer"]
    else:
        base = []
    if schema == _GROUPS_SCHEMA:
        return _validate_list(data, "groups", _validate_group_entry, base)
    if schema == _ANNOTATIONS_SCHEMA:
        return _validate_list(data, "annotations", _validate_annotation_entry, base)
    return ValidationReport(ok=False, entries=0, errors=[f"unknown schema_version {schema!r}"])


def _validate_list(
    data: dict[str, Any],
    key: str,
    validate_entry: Any,
    errors: list[str],
) -> ValidationReport:
    entries = data.get(key)
    if not isinstance(entries, list):
        return ValidationReport(ok=False, entries=0, errors=[*errors, f"{key} must be a list"])
    for index, entry in enumerate(entries):
        errors.extend(f"{key[:-1]} {index}: {message}" for message in validate_entry(entry))
    return ValidationReport(ok=not errors, entries=len(entries), errors=errors)


def _validate_group_entry(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return ["entry must be a JSON object"]
    problems: list[str] = []
    if entry.get("ac_trivial") not in _TRISTATE:
        problems.append("ac_trivial must be true, false, or null")
    if not isinstance(entry.get("source"), str):
        problems.append("source must be a string")
    transitions = entry.get("transitions")
    if transitions is not None and not (
        isinstance(transitions, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in transitions.items())
    ):
        problems.append("transitions must be a map of move-id strings to hash strings")
    try:
        presentation = BalancedPresentation.from_letters(int(entry["rank"]), entry["relators"])
    except Exception as exc:
        return [*problems, f"invalid presentation: {exc}"]
    if entry.get("hash") != presentation.content_hash:
        problems.append("hash does not match the relators")
    if entry.get("total_length") != presentation.total_length:
        problems.append("total_length does not match the relators")
    return problems


def _validate_annotation_entry(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return ["entry must be a JSON object"]
    problems: list[str] = []
    if not isinstance(entry.get("hash"), str):
        problems.append("hash must be a string")
    for field_name in ("distance_to_origin", "distance_to_shorter"):
        value = entry.get(field_name)
        if value is not None and (not isinstance(value, int) or value < 0):
            problems.append(f"{field_name} must be a non-negative integer or null")
    for field_name in ("optimal_moves_to_origin", "optimal_moves_to_shorter"):
        moves = entry.get(field_name)
        if not (isinstance(moves, list) and all(isinstance(m, int) for m in moves)):
            problems.append(f"{field_name} must be a list of integer move ids")
    for field_name in ("shorter_proven", "optimal"):
        if not isinstance(entry.get(field_name), bool):
            problems.append(f"{field_name} must be a boolean")
    return problems
