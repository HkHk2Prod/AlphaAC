from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Markdown reports live beside their dataset by stem: ``train.groups.summary.md``.
SUMMARY_SUFFIX = ".summary.md"


@dataclass(frozen=True, slots=True)
class Distribution:
    """An integer-keyed histogram over the groups, with summary statistics.

    `counts` maps each observed value (a size, a transition degree, ...) to the
    number of groups having it, sorted by value so the rendered table reads in
    order.
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
    """Aggregate statistics over one generated group dataset."""

    rank: int | None
    generator: str
    total_groups: int
    frontier: int
    exhausted: int
    ac_trivial: int
    ac_unknown: int
    by_source: dict[str, int]
    total_length: Distribution
    transition_degree: Distribution


def _distribution(values: Iterable[int]) -> Distribution:
    return Distribution(dict(sorted(Counter(values).items())))


def summarize(data: dict[str, Any]) -> DatasetSummary:
    """Compute distribution statistics from a loaded group dataset document.

    Reads the minimal per-group fields written by ``dataset grow`` -- total relator
    length, the ``transitions`` adjacency (present only on expanded groups), the
    ``source`` provenance, and known AC-triviality -- and buckets them.
    """
    groups = data.get("groups", [])
    provenance = data.get("provenance", {})
    lengths: list[int] = []
    degrees: list[int] = []
    sources: Counter[str] = Counter()
    frontier = exhausted = ac_trivial = ac_unknown = 0
    for entry in groups:
        lengths.append(int(entry.get("total_length", 0)))
        transitions = entry.get("transitions")
        if transitions is None:
            frontier += 1
        else:
            exhausted += 1
            degrees.append(len(transitions))
        sources[str(entry.get("source", "unknown"))] += 1
        if entry.get("ac_trivial") is True:
            ac_trivial += 1
        elif entry.get("ac_trivial") is None:
            ac_unknown += 1
    return DatasetSummary(
        rank=data.get("rank"),
        generator=str(provenance.get("generator", "unknown")),
        total_groups=len(groups),
        frontier=frontier,
        exhausted=exhausted,
        ac_trivial=ac_trivial,
        ac_unknown=ac_unknown,
        by_source=dict(sources.most_common()),
        total_length=_distribution(lengths),
        transition_degree=_distribution(degrees),
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
    header = [
        f"# Dataset summary: {name}",
        "",
        f"- Generator: `{summary.generator}`",
        f"- Rank: {summary.rank}",
        f"- Total groups: {summary.total_groups}",
        f"- Frontier (unexpanded): {summary.frontier} | Expanded: {summary.exhausted}",
        f"- AC-trivial: {summary.ac_trivial} | unknown: {summary.ac_unknown}",
        "",
        "## By source",
        "",
        "| source | groups |",
        "| --- | --- |",
        *[f"| {source} | {count} |" for source, count in summary.by_source.items()],
        "",
    ]
    body = _section("By size (total relator length)", "length", summary.total_length) + _section(
        "By transition degree (universal neighbours within the cap)",
        "degree",
        summary.transition_degree,
    )
    return "\n".join(header + body) + "\n"


def summary_path_for(dataset_path: Path, summary_dir: Path) -> Path:
    """The Markdown report path for a dataset: ``<summary_dir>/<stem>.summary.md``."""
    return summary_dir / f"{dataset_path.stem}{SUMMARY_SUFFIX}"


def write_dataset_summary(dataset_path: str | Path, summary_dir: str | Path) -> Path:
    """Summarize the group dataset at `dataset_path`, writing Markdown under `summary_dir`.

    Returns the path of the report written.
    """
    dataset_path = Path(dataset_path)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = render_markdown(summarize(data), name=dataset_path.name)
    target = summary_path_for(dataset_path, Path(summary_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    return target
