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
    """Draws episode start states from a grown dataset file, keyed by episode seed."""

    def __init__(self, presentations: list[BalancedPresentation]) -> None:
        if not presentations:
            raise ValueError("dataset instance source has no presentations")
        self._presentations = presentations

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        max_difficulty: int | None = None,
        *,
        require_descent: bool = False,
    ) -> DatasetSource:
        """Load a grown dataset, optionally keeping only instances within a difficulty.

        ``max_difficulty`` caps the construction depth of the instances used, so a
        run can train on the easier part of the dataset (a coarse curriculum knob);
        ``None`` uses every instance. ``require_descent`` (used by the ``descent``
        reward) keeps only instances carrying a *proven* ``descent_distance`` -- the
        known minimal moves N that the reward needs -- and stamps that N onto each
        kept presentation's provenance so the environment can read it.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        presentations: list[BalancedPresentation] = []
        for entry in data.get("instances", []):
            if max_difficulty is not None and int(entry.get("difficulty", 0)) > max_difficulty:
                continue
            if require_descent:
                distance = entry.get("descent_distance")
                if entry.get("descent_proven") is not True or not isinstance(distance, int):
                    continue
                presentation = BalancedPresentation.from_json(entry)
                presentation.provenance["descent_distance"] = distance
                presentations.append(presentation)
            else:
                presentations.append(BalancedPresentation.from_json(entry))
        if not presentations:
            constraints = []
            if max_difficulty is not None:
                constraints.append(f"difficulty <= {max_difficulty}")
            if require_descent:
                constraints.append("a proven descent_distance")
            suffix = f" with {' and '.join(constraints)}" if constraints else ""
            raise ValueError(f"dataset at {path} has no instances{suffix}")
        return cls(presentations)

    def sample(self, seed: int) -> BalancedPresentation:
        return random.Random(seed).choice(self._presentations)


def build_instance_source(config: TrainingPipelineConfig) -> InstanceSource:
    """Pick the episode instance source the run's config asks for.

    A configured ``dataset_path`` seeds episodes from that grown dataset; otherwise
    episodes fall back to seeded scrambles of the standard presentation. The
    ``descent`` reward needs each start state's known minimal descent distance N,
    which only a descent-annotated dataset carries, so it rejects the scramble
    fallback.
    """
    descent = config.reward_mode == "descent"
    if config.dataset_path:
        return DatasetSource.from_file(
            config.dataset_path, config.dataset_max_difficulty, require_descent=descent
        )
    if descent:
        raise ValueError(
            "reward_mode 'descent' needs a descent-annotated dataset (set dataset.path); "
            "random scrambles have no known descent distance N"
        )
    return ScrambleSource(config.rank, config.scramble_depth)
