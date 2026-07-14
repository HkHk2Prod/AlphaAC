from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Markdown reports live beside their dataset by stem: ``train.groups.summary.md``.
SUMMARY_SUFFIX = ".summary.md"

# Summaries are published to their own folder in the Hugging Face bucket, keyed by
# the dataset name so they are easy to find: the generation and annotation Kaggle
# notebooks push ``datasets_summaries/train_rank2.groups.summary.md`` and friends.
SUMMARIES_PREFIX = "datasets_summaries"


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
    # A ball stores no adjacency to read expansion off -- expanding a group there means
    # recording its distances, not its edges -- so it states the expanded count instead.
    if "expanded" in provenance:
        exhausted = int(provenance["expanded"])
        frontier = len(groups) - exhausted
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


@dataclass(frozen=True, slots=True)
class AnnotationSummary:
    """Aggregate statistics over one move-set annotation file."""

    rank: int | None
    moveset: str
    move_catalog: str
    total: int
    reached_origin: int
    with_shorter: int
    proven: int
    unresolved: int
    # Every group within this distance of the origin is present, so its distances are
    # complete as well as optimal. Only a closest-first `dataset ball` can claim it.
    complete_depth: int | None
    distance_to_origin: Distribution
    distance_to_shorter: Distribution

    @property
    def has_descent(self) -> bool:
        """Whether a descent pass ever ran over this file."""
        return bool(self.with_shorter or self.proven)


def summarize_annotations(data: dict[str, Any]) -> AnnotationSummary:
    """Compute distance distributions from a loaded annotation document.

    Buckets each group's distance to the origin (present only for groups that
    reach the trivial group) and its length-descent distance (present only where a
    strictly shorter group was found), and counts how many entries are proven
    settled versus still unresolved -- an interrupted or depth-capped pass leaves
    the tail of groups without a proven shorter-distance.
    """
    annotations = data.get("annotations", [])
    to_origin: list[int] = []
    to_shorter: list[int] = []
    reached_origin = with_shorter = proven = 0
    for entry in annotations:
        d_origin = entry.get("distance_to_origin")
        if d_origin is not None:
            reached_origin += 1
            to_origin.append(int(d_origin))
        d_shorter = entry.get("distance_to_shorter")
        if d_shorter is not None:
            with_shorter += 1
            to_shorter.append(int(d_shorter))
        if entry.get("shorter_proven") is True:
            proven += 1
    depth = data.get("provenance", {}).get("complete_depth")
    return AnnotationSummary(
        rank=data.get("rank"),
        moveset=str(data.get("moveset", "unknown")),
        move_catalog=str(data.get("move_catalog", "unknown")),
        total=len(annotations),
        reached_origin=reached_origin,
        with_shorter=with_shorter,
        proven=proven,
        unresolved=len(annotations) - proven,
        complete_depth=None if depth is None else int(depth),
        distance_to_origin=_distribution(to_origin),
        distance_to_shorter=_distribution(to_shorter),
    )


def render_annotation_markdown(summary: AnnotationSummary, *, name: str) -> str:
    """Render a human-readable Markdown report for one annotation summary."""
    header = [
        f"# Annotation summary: {name}",
        "",
        f"- Move set: `{summary.moveset}`",
        f"- Move catalog: `{summary.move_catalog}`",
        f"- Rank: {summary.rank}",
        f"- Total groups: {summary.total}",
        f"- Reach the origin (optimal): {summary.reached_origin}",
    ]
    if summary.complete_depth is not None:
        header.append(
            f"- Complete through distance {summary.complete_depth}: every group that "
            f"close to the origin is in the dataset"
        )
    if summary.has_descent:
        header += [
            f"- With a shorter descent: {summary.with_shorter}",
            f"- Proven settled: {summary.proven} | unresolved: {summary.unresolved}",
        ]
    header.append("")
    body = _section(
        "By distance to origin (moves to the trivial group)",
        "distance",
        summary.distance_to_origin,
    )
    if summary.has_descent:
        body += _section(
            "By descent distance (moves to a strictly shorter group)",
            "distance",
            summary.distance_to_shorter,
        )
    return "\n".join(header + body) + "\n"


def write_annotation_summary(annotation_path: str | Path, summary_dir: str | Path) -> Path:
    """Summarize the annotation file at `annotation_path`, writing Markdown under `summary_dir`.

    Returns the path of the report written. Its name starts with the annotation
    file's stem (e.g. ``train_rank2.groups.strict-ac.annotations.summary.md``) so
    it sorts next to the dataset it describes.
    """
    annotation_path = Path(annotation_path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    report = render_annotation_markdown(summarize_annotations(data), name=annotation_path.name)
    target = summary_path_for(annotation_path, Path(summary_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    return target


def summary_remote_name(summary_path: str | Path) -> str:
    """The Hugging Face bucket object name for a summary: ``datasets_summaries/<file>``."""
    return f"{SUMMARIES_PREFIX}/{Path(summary_path).name}"
