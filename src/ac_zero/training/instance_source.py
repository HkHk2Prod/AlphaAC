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
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable
from ac_zero.training.pipeline_config import TrainingPipelineConfig


class InstanceSource(Protocol):
    """Supplies the starting presentation for one episode, given its seed."""

    def sample(self, seed: int) -> BalancedPresentation:
        """Return the presentation to start the episode seeded by ``seed``."""
        ...


@dataclass(frozen=True, slots=True)
class ScrambleSource:
    """Seeded random scrambles of the standard presentation -- the default source."""

    rank: int
    depth: int

    def sample(self, seed: int) -> BalancedPresentation:
        return generate_solvable(self.rank, self.depth, seed).presentation


class DatasetSource:
    """Draws episode start states from a grown group dataset, keyed by episode seed."""

    def __init__(self, presentations: list[BalancedPresentation]) -> None:
        if not presentations:
            raise ValueError("dataset instance source has no presentations")
        self._presentations = presentations

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        annotations_path: str | Path | None = None,
        max_difficulty: int | None = None,
        *,
        require_descent: bool = False,
        moveset: str | None = None,
    ) -> DatasetSource:
        """Load a grown group dataset, filtered/annotated from its companion file.

        Presentations come from the ``.groups.json`` at ``path``. The per-group
        distances live in the separate ``annotations_path`` file: ``max_difficulty``
        caps ``distance_to_origin`` (a coarse curriculum knob, ``None`` = all), and
        ``require_descent`` (the ``descent`` reward) keeps only groups with a
        *proven* ``distance_to_shorter`` -- the known minimal moves N -- stamping
        that N onto each kept presentation's provenance for the environment to read.
        When ``require_descent``, ``moveset`` must match the annotation file's own
        ``moveset`` field: N is only a valid descent distance for the move set the
        environment actually plays.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rank = int(data["rank"])
        annotations = _load_annotations(
            annotations_path, expected_moveset=moveset if require_descent else None
        )
        presentations: list[BalancedPresentation] = []
        for entry in data.get("groups", []):
            annotation = annotations.get(entry["hash"], {})
            if max_difficulty is not None:
                distance = annotation.get("distance_to_origin")
                if not isinstance(distance, int) or distance > max_difficulty:
                    continue
            descent_distance = annotation.get("distance_to_shorter")
            if require_descent and not (
                annotation.get("shorter_proven") is True and isinstance(descent_distance, int)
            ):
                continue
            presentation = BalancedPresentation.from_letters(rank, entry["relators"])
            if require_descent:
                presentation.provenance["descent_distance"] = descent_distance
            presentations.append(presentation)
        if not presentations:
            constraints = []
            if max_difficulty is not None:
                constraints.append(f"distance_to_origin <= {max_difficulty}")
            if require_descent:
                constraints.append("a proven distance_to_shorter")
            suffix = f" with {' and '.join(constraints)}" if constraints else ""
            raise ValueError(f"group dataset at {path} has no groups{suffix}")
        return cls(presentations)

    def sample(self, seed: int) -> BalancedPresentation:
        return random.Random(seed).choice(self._presentations)


def _load_annotations(
    path: str | Path | None, *, expected_moveset: str | None = None
) -> dict[str, dict[str, object]]:
    """Load a `.annotations.json` file as a `hash -> annotation entry` map.

    When ``expected_moveset`` is given, rejects a file annotated under a different
    move set -- its distances (e.g. the descent distance N) aren't valid for an
    environment that plays a different move set.
    """
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_moveset is not None and data.get("moveset") != expected_moveset:
        raise ValueError(
            f"annotations at {path} were computed under move set "
            f"{data.get('moveset')!r}, but the environment is configured for "
            f"{expected_moveset!r} -- the descent distance N would not match "
            "what the environment can actually play"
        )
    return {entry["hash"]: entry for entry in data.get("annotations", [])}


def build_instance_source(config: TrainingPipelineConfig) -> InstanceSource:
    """Pick the episode instance source the run's config asks for.

    A configured ``dataset_path`` seeds episodes from that grown group dataset;
    otherwise episodes fall back to seeded scrambles of the standard presentation.
    The ``descent`` reward needs each start state's known minimal descent distance
    N, which only a group's annotations carry, so it rejects the scramble fallback.
    """
    descent = config.reward_mode == "descent"
    if config.dataset_path:
        return DatasetSource.from_file(
            config.dataset_path,
            config.dataset_annotations_path,
            config.dataset_max_difficulty,
            require_descent=descent,
            moveset=config.moveset,
        )
    if descent:
        raise ValueError(
            "reward_mode 'descent' needs a group dataset with annotations (set dataset.path "
            "and dataset.annotations); random scrambles have no known descent distance N"
        )
    return ScrambleSource(config.rank, config.scramble_depth)
