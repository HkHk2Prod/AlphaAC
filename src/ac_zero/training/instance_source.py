"""Where a self-play episode's starting presentation comes from.

Both training backends play episodes from a seeded starting presentation. By
default that is a random scramble of the standard presentation
(:class:`ScrambleSource`, wrapping :func:`generate_solvable`). When the run is
configured with a grown dataset (``dataset.path`` in the experiment config, or a
dataset pulled from the Hugging Face bucket by the notebook/CLI), episodes are
instead seeded from that dataset's guaranteed-solvable presentations
(:class:`DatasetSource`).

Sampling is keyed on the per-episode seed, so an episode's instance is identical
whether it runs in-process or in a worker pool -- the same determinism guarantee
the self-play loop already relies on.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable
from ac_zero.training.pipeline_config import TrainingPipelineConfig

Summary = Mapping[str, float | int | bool | str]


class InstanceSource(Protocol):
    """Supplies the starting presentation for one episode, given its seed."""

    def sample(self, seed: int) -> BalancedPresentation:
        """Return the presentation to start the episode seeded by ``seed``."""
        ...

    @property
    def potentials(self) -> Mapping[str, int]:
        """Map a presentation hash to its distance to the trivial group, if known."""
        ...

    def describe(self) -> Summary:
        """Return a log-friendly summary of what episodes will start from."""
        ...


@dataclass(frozen=True, slots=True)
class ScrambleSource:
    """Seeded random scrambles of the standard presentation -- the default source."""

    rank: int
    depth: int

    def sample(self, seed: int) -> BalancedPresentation:
        return generate_solvable(self.rank, self.depth, seed).presentation

    @property
    def potentials(self) -> Mapping[str, int]:
        # Scrambles carry no annotations, so the potential reward falls back to
        # total length for every state (see `ACEnvironment._potential`).
        return {}

    def describe(self) -> Summary:
        return {"source": "scramble", "rank": self.rank, "depth": self.depth}


class DatasetSource:
    """Draws episode start states from a grown group dataset, keyed by episode seed."""

    def __init__(
        self,
        presentations: list[BalancedPresentation],
        potentials: Mapping[str, int] | None = None,
        summary: Summary | None = None,
    ) -> None:
        if not presentations:
            raise ValueError("dataset instance source has no presentations")
        self._presentations = presentations
        self._potentials = dict(potentials or {})
        self._summary: dict[str, float | int | bool | str] = dict(summary or {})

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        annotations_path: str | Path | None = None,
        max_difficulty: int | None = None,
        require_potential: bool = False,
    ) -> DatasetSource:
        """Load a grown group dataset, filtered from its companion annotations file.

        Presentations come from the ``.groups.json`` at ``path``. The per-group
        distances live in the separate ``annotations_path`` file: ``max_difficulty``
        caps ``distance_to_origin`` (a coarse curriculum knob, ``None`` = all), and
        ``require_potential`` keeps only groups whose ``distance_to_origin`` is known
        (the potential reward's start states). The known distances are also exposed
        via :attr:`potentials` so the environment can score potential-based shaping.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rank = int(data["rank"])
        annotations = _load_annotations(annotations_path)
        groups = data.get("groups", [])
        potentials: dict[str, int] = {}
        presentations: list[BalancedPresentation] = []
        annotated = 0
        used_distances: list[int] = []
        for entry in groups:
            distance = annotations.get(entry["hash"], {}).get("distance_to_origin")
            if isinstance(distance, int):
                potentials[entry["hash"]] = distance
                annotated += 1
            if max_difficulty is not None and (
                not isinstance(distance, int) or distance > max_difficulty
            ):
                continue
            if require_potential and not isinstance(distance, int):
                continue
            presentations.append(BalancedPresentation.from_letters(rank, entry["relators"]))
            if isinstance(distance, int):
                used_distances.append(distance)
        if not presentations:
            constraints = []
            if max_difficulty is not None:
                constraints.append(f"distance_to_origin <= {max_difficulty}")
            elif require_potential:
                constraints.append("a known distance_to_origin")
            suffix = f" with {' and '.join(constraints)}" if constraints else ""
            raise ValueError(f"group dataset at {path} has no groups{suffix}")
        summary = _dataset_summary(
            path, rank, len(groups), len(presentations), annotated, used_distances, max_difficulty
        )
        return cls(presentations, potentials, summary)

    def sample(self, seed: int) -> BalancedPresentation:
        return random.Random(seed).choice(self._presentations)

    @property
    def potentials(self) -> Mapping[str, int]:
        return self._potentials

    def describe(self) -> Summary:
        return dict(self._summary)


def _load_annotations(path: str | Path | None) -> dict[str, dict[str, object]]:
    """Load a `.annotations.json` file as a `hash -> annotation entry` map."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["hash"]: entry for entry in data.get("annotations", [])}


def _dataset_summary(
    path: str | Path,
    rank: int,
    total: int,
    used: int,
    annotated: int,
    used_distances: list[int],
    max_difficulty: int | None,
) -> dict[str, float | int | bool | str]:
    """Build the log summary for a loaded group dataset.

    `total` groups live in the file, `annotated` of them carry a known distance
    to origin, and `used` remain after the `max_difficulty` curriculum filter.
    """
    summary: dict[str, float | int | bool | str] = {
        "source": "dataset",
        "path": str(path),
        "rank": rank,
        "groups_total": total,
        "groups_used": used,
        "annotated": annotated,
        "annotated_pct": round(100.0 * annotated / total, 1) if total else 0.0,
    }
    if max_difficulty is not None:
        summary["max_difficulty"] = max_difficulty
    if used_distances:
        summary["distance_min"] = min(used_distances)
        summary["distance_max"] = max(used_distances)
    return summary


def build_instance_source(config: TrainingPipelineConfig) -> InstanceSource:
    """Pick the episode instance source the run's config asks for.

    A configured ``dataset_path`` seeds episodes from that grown group dataset;
    otherwise episodes fall back to seeded scrambles of the standard presentation.
    """
    if config.dataset_path:
        return DatasetSource.from_file(
            config.dataset_path,
            config.dataset_annotations_path,
            config.dataset_max_difficulty,
            require_potential=config.reward_mode == "potential",
        )
    return ScrambleSource(config.rank, config.scramble_depth)
