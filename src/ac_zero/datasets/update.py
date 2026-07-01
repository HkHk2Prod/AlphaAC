from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ac_zero.agents.greedy import GreedyBestFirstConfig, GreedyBestFirstSearch
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.labels import UNKNOWN, TrivializationLabel, known_solution, merge_labels
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.search.breadth_first import BreadthFirstConfig, BreadthFirstSearch
from ac_zero.system.parallel import describe_worker_pool, imap_ordered

# Emitted incrementally during long improvement passes: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]

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


@dataclass(frozen=True, slots=True)
class _EntryResult:
    """Outcome of searching one entry, computed in a (possibly remote) worker."""

    label: TrivializationLabel
    found: bool
    improved: bool
    newly_optimal: bool


# Per-worker state populated once by the process-pool initializer: the strategies
# to run and a scratch certificate path private to this process, so parallel
# searches never write the same file.
_WORKER_STRATEGIES: Sequence[SearchStrategy] = ()
_WORKER_CERT: Path | None = None


def _init_search_worker(strategies: Sequence[SearchStrategy]) -> None:
    global _WORKER_STRATEGIES, _WORKER_CERT
    _WORKER_STRATEGIES = strategies
    tmp = tempfile.mkdtemp(prefix="aczero-improve-")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    _WORKER_CERT = Path(tmp) / "candidate.json"


def _search_entry(entry: dict[str, Any]) -> _EntryResult:
    """Run every strategy on one entry and merge the results into its label."""
    assert _WORKER_CERT is not None
    presentation = BalancedPresentation.from_json(entry)
    before = label_from_entry(entry)
    merged = before
    found = False
    for strategy in _WORKER_STRATEGIES:
        label = strategy.label(presentation, certificate_path=_WORKER_CERT)
        if label.minimal_known_operations is not None:
            found = True
        merged = merge_labels(merged, label)
    return _EntryResult(
        label=merged,
        found=found,
        improved=merged != before,
        newly_optimal=bool(merged.optimal and not before.optimal),
    )


def improve_dataset(
    path: str | Path,
    *,
    strategies: Sequence[SearchStrategy],
    output: str | Path | None = None,
    max_difficulty: int | None = None,
    workers: int = 0,
    progress: ProgressCallback | None = None,
) -> ImproveReport:
    """Search every entry for a better trivialization and merge improvements in place.

    Entries are deduplicated by content hash first. For each entry the strategies
    run and their labels are merged into the existing label with
    :func:`merge_labels`, which never replaces a shorter known solution with a
    longer one or demotes a known triviality result. The file is written
    atomically so an interrupted run cannot corrupt the dataset.

    Per-entry searches are independent, so they fan out across ``workers``
    processes; the default ``0`` autodetects and uses every CPU core (set 1 for
    in-process, or a negative count to leave that many free). Results are merged
    back in entry order, so the output is identical regardless of the worker count.
    """
    if progress is not None:
        # Open with a full description of the task so the run is reproducible
        # from its log: the input, where it writes, the search strategies, and the
        # difficulty gate.
        _, worker_message, worker_metrics = describe_worker_pool(workers)
        progress(
            "improving dataset",
            {
                "input": str(path),
                "output": str(output if output is not None else path),
                "strategies": ", ".join(getattr(s, "name", type(s).__name__) for s in strategies),
                "max_difficulty": "all" if max_difficulty is None else max_difficulty,
            },
        )
        progress(worker_message, worker_metrics)

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("instances", [])
    entries, duplicates = dedupe_entries(raw)
    if progress is not None:
        progress(
            "deduplicated entries",
            {"unique": len(entries), "duplicates_merged": duplicates},
        )

    total = len(entries)
    # Report progress at ~10 checkpoints so long search passes stay visible.
    interval = max(1, total // 10)
    searched = solved = improved = optimal = 0
    search_positions = [
        i for i, entry in enumerate(entries) if _should_search(entry, max_difficulty)
    ]
    results = imap_ordered(
        _search_entry,
        [entries[i] for i in search_positions],
        workers=workers,
        initializer=_init_search_worker,
        initargs=(tuple(strategies),),
    )
    pending = set(search_positions)
    for index, entry in enumerate(entries, start=1):
        optimal += 1 if label_from_entry(entry).optimal else 0
        if (index - 1) in pending:
            searched += 1
            result = next(results)
            if result.found:
                solved += 1
            if result.improved:
                improved += 1
                if result.newly_optimal:
                    optimal += 1
                apply_label(entry, result.label)
        if progress is not None and (index % interval == 0 or index == total):
            progress(
                "improving dataset",
                {
                    "processed": index,
                    "total": total,
                    "searched": searched,
                    "solved": solved,
                    "improved": improved,
                },
            )

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
