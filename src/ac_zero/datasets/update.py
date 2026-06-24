from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ac_zero.agents.greedy import GreedyBestFirstConfig, GreedyBestFirstSearch
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.labels import UNKNOWN, TrivializationLabel, known_solution, merge_labels
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.search.breadth_first import BreadthFirstConfig, BreadthFirstSearch

_LABEL_FIELDS = ("ac_trivial", "minimal_known_operations", "optimal")


def label_from_entry(entry: dict[str, Any]) -> TrivializationLabel:
    """Read the trivialization label already stored on a dataset entry."""
    return TrivializationLabel(
        ac_trivial=entry.get("ac_trivial"),
        minimal_known_operations=entry.get("minimal_known_operations"),
        optimal=entry.get("optimal"),
    )


def apply_label(entry: dict[str, Any], label: TrivializationLabel) -> dict[str, Any]:
    """Write a trivialization label onto a dataset entry in place."""
    entry.update(label.to_json())
    return entry


def dedupe_entries(entries: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse entries sharing a content hash, merging labels and keeping difficulty min.

    Returns the unique entries in first-seen order plus the number of duplicates
    that were merged away. Labels are combined with :func:`merge_labels`, so a
    duplicate can only ever improve the surviving entry's known information.
    """
    by_hash: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    duplicates = 0
    for entry in entries:
        content = entry.get("content_hash") or BalancedPresentation.from_json(entry).content_hash
        if content in by_hash:
            duplicates += 1
            survivor = by_hash[content]
            apply_label(survivor, merge_labels(label_from_entry(survivor), label_from_entry(entry)))
            survivor["difficulty"] = _min_optional(
                survivor.get("difficulty"), entry.get("difficulty")
            )
            continue
        by_hash[content] = entry
        order.append(content)
    return [by_hash[content] for content in order], duplicates


class SearchStrategy(Protocol):
    """A search that maps a presentation to a (possibly unknown) trivialization label."""

    def label(
        self, presentation: BalancedPresentation, *, certificate_path: Path
    ) -> TrivializationLabel: ...


@dataclass(frozen=True, slots=True)
class BreadthFirstStrategy:
    """Shortest-path search; reports proven-optimal labels when BFS completes."""

    name: str = "breadth_first"
    max_moves: int = 12
    total_length_cap: int = 48
    max_expansions: int = 3_000
    max_generated: int = 30_000

    def label(
        self, presentation: BalancedPresentation, *, certificate_path: Path
    ) -> TrivializationLabel:
        env = ACEnvironment(
            presentation,
            ACEnvironmentConfig(max_moves=self.max_moves, total_length_cap=self.total_length_cap),
        )
        result = BreadthFirstSearch(
            BreadthFirstConfig(self.max_expansions, self.max_generated)
        ).solve(presentation, env_template=env, certificate_path=certificate_path)
        if not result.success:
            return UNKNOWN
        return known_solution(len(result.path), optimal=result.metrics["proved_optimal"] >= 1.0)


@dataclass(frozen=True, slots=True)
class GreedyBestFirstStrategy:
    """Length-ordered heuristic search; yields a known but non-optimal upper bound."""

    name: str = "greedy_best_first"
    max_moves: int = 24
    total_length_cap: int = 64
    max_expansions: int = 2_000
    max_generated: int = 50_000

    def label(
        self, presentation: BalancedPresentation, *, certificate_path: Path
    ) -> TrivializationLabel:
        env = ACEnvironment(
            presentation,
            ACEnvironmentConfig(max_moves=self.max_moves, total_length_cap=self.total_length_cap),
        )
        result = GreedyBestFirstSearch(
            GreedyBestFirstConfig(self.max_expansions, self.max_generated)
        ).solve(presentation, env_template=env, certificate_path=certificate_path)
        if not result.success:
            return UNKNOWN
        return known_solution(len(result.path), optimal=False)


@dataclass(frozen=True, slots=True)
class ImproveReport:
    """Summary of one dataset-improvement pass."""

    total: int
    duplicates_merged: int
    searched: int
    solved: int
    improved: int
    proved_optimal: int


def improve_dataset(
    path: str | Path,
    *,
    strategies: Sequence[SearchStrategy],
    output: str | Path | None = None,
    max_difficulty: int | None = None,
) -> ImproveReport:
    """Search every entry for a better trivialization and merge improvements in place.

    Entries are deduplicated by content hash first. For each entry the strategies
    run and their labels are merged into the existing label with
    :func:`merge_labels`, which never replaces a shorter known solution with a
    longer one or demotes a known triviality result. The file is written
    atomically so an interrupted run cannot corrupt the dataset.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("instances", [])
    entries, duplicates = dedupe_entries(raw)

    searched = solved = improved = optimal = 0
    with tempfile.TemporaryDirectory() as tmp:
        certificate = Path(tmp) / "candidate.json"
        for entry in entries:
            optimal += 1 if label_from_entry(entry).optimal else 0
            if not _should_search(entry, max_difficulty):
                continue
            searched += 1
            presentation = BalancedPresentation.from_json(entry)
            before = label_from_entry(entry)
            merged = before
            found = False
            for strategy in strategies:
                label = strategy.label(presentation, certificate_path=certificate)
                if label.minimal_known_operations is not None:
                    found = True
                merged = merge_labels(merged, label)
            if found:
                solved += 1
            if merged != before:
                improved += 1
                if merged.optimal and not before.optimal:
                    optimal += 1
                apply_label(entry, merged)

    data["instances"] = entries
    _refresh_provenance(data, entries)
    _atomic_write(Path(output) if output is not None else Path(path), data)
    return ImproveReport(
        total=len(entries),
        duplicates_merged=duplicates,
        searched=searched,
        solved=solved,
        improved=improved,
        proved_optimal=optimal,
    )


def _should_search(entry: dict[str, Any], max_difficulty: int | None) -> bool:
    # A proven-optimal entry cannot be improved, so re-searching it only wastes
    # work; skipping makes repeated improvement passes cheap and idempotent.
    if entry.get("optimal") is True:
        return False
    if max_difficulty is None:
        return True
    difficulty = entry.get("difficulty")
    if difficulty is None:
        return True
    return int(difficulty) <= max_difficulty


def _refresh_provenance(data: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        return
    provenance["count"] = len(entries)
    solved_lengths = [
        entry["minimal_known_operations"]
        for entry in entries
        if entry.get("minimal_known_operations") is not None
    ]
    if solved_lengths:
        provenance["min_known_operations"] = min(solved_lengths)
        provenance["max_known_operations"] = max(solved_lengths)


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False, encoding="utf-8"
    )
    try:
        with handle:
            handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _min_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None
