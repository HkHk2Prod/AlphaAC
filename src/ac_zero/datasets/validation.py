from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation

_SCHEMA_VERSIONS = {
    "aczero-dataset-v1",
    "aczero-dataset-v2",
    "aczero-dataset-v3",
    "aczero-candidates-v1",
}
_TRISTATE = (True, False, None)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating a dataset document against the AC-Zero schema."""

    ok: bool
    instances: int
    errors: list[str] = field(default_factory=list)


def validate_dataset(path: str | Path) -> ValidationReport:
    """Validate a dataset JSON file on disk."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationReport(ok=False, instances=0, errors=[f"unreadable dataset: {exc}"])
    return validate_mapping(data)


def validate_mapping(data: Any) -> ValidationReport:
    """Validate an already-parsed dataset document, checking structure and labels.

    Beyond shape, every instance is reparsed as a presentation and its stored
    content hash is recomputed, so corrupted relators or stale hashes are caught.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return ValidationReport(ok=False, instances=0, errors=["document must be a JSON object"])
    if data.get("schema_version") not in _SCHEMA_VERSIONS:
        errors.append(f"unknown schema_version {data.get('schema_version')!r}")
    if not isinstance(data.get("rank"), int) or data.get("rank", 0) < 1:
        errors.append("rank must be a positive integer")
    instances = data.get("instances")
    if not isinstance(instances, list):
        return ValidationReport(ok=False, instances=0, errors=[*errors, "instances must be a list"])

    for index, entry in enumerate(instances):
        errors.extend(f"instance {index}: {message}" for message in _validate_entry(entry))
    return ValidationReport(ok=not errors, instances=len(instances), errors=errors)


def _validate_entry(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return ["entry must be a JSON object"]
    problems: list[str] = []
    problems.extend(_check_label(entry))
    problems.extend(_check_graph_fields(entry))
    difficulty = entry.get("difficulty")
    if difficulty is not None and (not isinstance(difficulty, int) or difficulty < 0):
        problems.append("difficulty must be a non-negative integer or absent")
    problems.extend(_check_descent(entry))
    try:
        presentation = BalancedPresentation.from_json(entry)
    except Exception as exc:
        return [*problems, f"invalid presentation: {exc}"]
    stored = entry.get("content_hash")
    if stored is not None and stored != presentation.content_hash:
        problems.append("content_hash does not match the relators")
    return problems


def _check_graph_fields(entry: dict[str, Any]) -> list[str]:
    """Validate the v3 construction-graph fields when present (older files omit them)."""
    problems: list[str] = []
    if "exhausted" in entry and not isinstance(entry["exhausted"], bool):
        problems.append("exhausted must be a boolean")
    predecessors = entry.get("predecessors")
    if predecessors is None:
        return problems
    if not isinstance(predecessors, list):
        return [*problems, "predecessors must be a list"]
    for index, edge in enumerate(predecessors):
        if not isinstance(edge, dict):
            problems.append(f"predecessor {index} must be an object")
            continue
        if not isinstance(edge.get("parent_hash"), str):
            problems.append(f"predecessor {index}: parent_hash must be a string")
        if not isinstance(edge.get("move"), dict):
            problems.append(f"predecessor {index}: move must be an object")
    return problems


def _check_descent(entry: dict[str, Any]) -> list[str]:
    """Validate the length-descent annotation fields when present (grow omits them).

    ``descent_distance`` is the fewest moves that strictly shorten the
    presentation: a positive integer, or null when none is known. ``descent_proven``
    flags whether that answer is exact, so it is meaningful only alongside a
    written distance field.
    """
    problems: list[str] = []
    distance = entry.get("descent_distance")
    if distance is not None and (not isinstance(distance, int) or distance < 1):
        problems.append("descent_distance must be a positive integer or null")
    proven = entry.get("descent_proven")
    if proven is not None and not isinstance(proven, bool):
        problems.append("descent_proven must be a boolean or absent")
    if proven is not None and "descent_distance" not in entry:
        problems.append("descent_proven requires a descent_distance field")
    return problems


def _check_label(entry: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if entry.get("ac_trivial") not in _TRISTATE:
        problems.append("ac_trivial must be true, false, or null")
    operations = entry.get("minimal_known_operations")
    if operations is not None and (not isinstance(operations, int) or operations < 0):
        problems.append("minimal_known_operations must be a non-negative integer or null")
    if entry.get("optimal") not in _TRISTATE:
        problems.append("optimal must be true, false, or null")
    if entry.get("optimal") is True and operations is None:
        problems.append("optimal cannot be true without minimal_known_operations")
    return problems
