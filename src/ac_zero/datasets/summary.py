from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Markdown reports live beside their dataset by stem: ``train_rank2.summary.md``.
SUMMARY_SUFFIX = ".summary.md"


@dataclass(frozen=True, slots=True)
class Distribution:
    """An integer-keyed histogram over the groups, with summary statistics.

    `counts` maps each observed value (a difficulty, a size, ...) to the number
    of groups having it, sorted by value so the rendered table reads in order.
    """

    counts: dict[int, int]

    @property
    def population(self) -> int:
        return sum(self.counts.values())

    @property
    def minimum(self) -> int | None:
        return min(self.counts) if self.counts else None

    @property
    def maximum(self) -> int | None:
        return max(self.counts) if self.counts else None

    @property
    def mean(self) -> float | None:
        if not self.counts:
            return None
        return sum(value * n for value, n in self.counts.items()) / self.population


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Aggregate statistics over one generated dataset."""

    rank: int | None
    generator: str
    total_groups: int
    roots: int
    frontier: int
    exhausted: int
    optimal: int
    difficulty: Distribution
    total_length: Distribution
    predecessors: Distribution
    known_operations: Distribution


def _distribution(values: Iterable[int]) -> Distribution:
    return Distribution(dict(sorted(Counter(values).items())))


def summarize(data: dict[str, Any]) -> DatasetSummary:
    """Compute distribution statistics from a loaded dataset document.

    Reads the flat per-instance fields written by ``dataset grow`` -- difficulty,
    relator sizes, co-optimal ``predecessors``, known trivialization length, and
    the ``exhausted`` frontier flag -- and buckets them into histograms.
    """
    instances = data.get("instances", [])
    provenance = data.get("provenance", {})
    difficulties: list[int] = []
    lengths: list[int] = []
    predecessors: list[int] = []
    operations: list[int] = []
    roots = frontier = exhausted = optimal = 0
    for entry in instances:
        difficulty = int(entry.get("difficulty", 0))
        difficulties.append(difficulty)
        lengths.append(sum(len(relator) for relator in entry.get("relators", [])))
        predecessors.append(len(entry.get("predecessors", [])))
        ops = entry.get("minimal_known_operations")
        if ops is not None:
            operations.append(int(ops))
        roots += difficulty == 0
        exhausted += bool(entry.get("exhausted"))
        frontier += not entry.get("exhausted")
        optimal += entry.get("optimal") is True
    return DatasetSummary(
        rank=data.get("rank"),
        generator=str(provenance.get("generator", "unknown")),
        total_groups=len(instances),
        roots=roots,
        frontier=frontier,
        exhausted=exhausted,
        optimal=optimal,
        difficulty=_distribution(difficulties),
        total_length=_distribution(lengths),
        predecessors=_distribution(predecessors),
        known_operations=_distribution(operations),
    )


def _section(title: str, unit: str, dist: Distribution) -> list[str]:
    """Render one histogram as a stats line plus a value/count Markdown table."""
    lines = [f"## {title}", ""]
    if dist.population == 0:
        return [*lines, "_No data._", ""]
    assert dist.mean is not None  # population > 0 guarantees a mean
    lines += [
        f"- min: {dist.minimum} | max: {dist.maximum} | mean: {dist.mean:.2f}",
        "",
        f"| {unit} | groups |",
        "| --- | --- |",
    ]
    lines += [f"| {value} | {count} |" for value, count in dist.counts.items()]
    lines.append("")
    return lines


def render_markdown(summary: DatasetSummary, *, name: str) -> str:
    """Render a human-readable Markdown report for one dataset summary."""
    known = summary.known_operations.population
    header = [
        f"# Dataset summary: {name}",
        "",
        f"- Generator: `{summary.generator}`",
        f"- Rank: {summary.rank}",
        f"- Total groups: {summary.total_groups}",
        f"- Roots (difficulty 0): {summary.roots}",
        f"- Frontier (open): {summary.frontier} | Exhausted: {summary.exhausted}",
        f"- Proven optimal: {summary.optimal}",
        f"- With known trivialization: {known} | unknown: {summary.total_groups - known}",
        "",
    ]
    body = (
        _section("By construction difficulty (depth from trivial)", "depth", summary.difficulty)
        + _section("By size (total relator length)", "length", summary.total_length)
        + _section("By co-optimal construction moves", "moves", summary.predecessors)
        + _section("By known trivialization length", "operations", summary.known_operations)
    )
    return "\n".join(header + body) + "\n"


def summary_path_for(dataset_path: Path, summary_dir: Path) -> Path:
    """The Markdown report path for a dataset: ``<summary_dir>/<stem>.summary.md``."""
    return summary_dir / f"{dataset_path.stem}{SUMMARY_SUFFIX}"


def write_dataset_summary(dataset_path: str | Path, summary_dir: str | Path) -> Path:
    """Summarize the dataset at `dataset_path`, writing its Markdown report under `summary_dir`.

    Returns the path of the report written.
    """
    dataset_path = Path(dataset_path)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = render_markdown(summarize(data), name=dataset_path.name)
    target = summary_path_for(dataset_path, Path(summary_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    return target
