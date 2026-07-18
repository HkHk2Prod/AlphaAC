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

A grown dataset is far too large to parse into Python objects once per worker, so
:class:`DatasetSource` reads it through the memory-mapped sidecar built by
:mod:`ac_zero.datasets.instance_store` and rebuilds one presentation per episode.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable
from ac_zero.datasets.groups import read_relator_bound
from ac_zero.datasets.instance_store import UNKNOWN, InstanceStore
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig

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
        self, store: InstanceStore, selected: NDArray[np.int64] | None, summary: Summary
    ) -> None:
        """Bind the source to a mapped dataset and the group indices it may sample.

        ``selected`` is ``None`` when no filter applies, so an unfiltered run does
        not hand every worker its own copy of the identity permutation.
        """
        self._store = store
        self._selected = selected
        self._count = store.count if selected is None else int(selected.size)
        self._summary: dict[str, float | int | bool | str] = dict(summary)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        annotations_path: str | Path | None = None,
        max_difficulty: int | None = None,
        require_potential: bool = False,
        *,
        max_relator_tokens: int,
    ) -> DatasetSource:
        """Map a group dataset, filtered by its companion annotations file.

        Presentations come from the ``.groups.json`` at ``path``. The per-group
        distances live in the separate ``annotations_path`` file: ``max_difficulty``
        caps ``distance_to_origin`` (``None`` = all), and
        ``require_potential`` keeps only groups whose ``distance_to_origin`` is known
        (the potential reward's start states). ``max_relator_tokens`` is not a filter
        but a contract: the dataset must have been *generated* under that bound, or
        the run is refused. The known distances are also exposed via :attr:`potentials`
        so the environment can score potential-based shaping.
        """
        groups = Path(path)
        require_dataset_bound(groups, max_relator_tokens)
        annotations = None if annotations_path is None else Path(annotations_path)
        store = InstanceStore.open(groups, annotations)
        selected = _select(store, max_difficulty, require_potential)
        if selected is not None and not selected.size:
            constraint = _constraint(max_difficulty, require_potential)
            raise ValueError(f"group dataset at {groups} has no groups{constraint}")
        return cls(store, selected, _summarize(groups, store, selected, max_difficulty))

    def sample(self, seed: int) -> BalancedPresentation:
        # `Random.choice` over a sequence draws exactly this index, so seeding is
        # unchanged from when the selection was a materialized list of presentations.
        index = random.Random(seed).randrange(self._count)
        if self._selected is not None:
            index = int(self._selected[index])
        return self._store.presentation(index)

    @property
    def potentials(self) -> Mapping[str, int]:
        return self._store.potentials

    def describe(self) -> Summary:
        return dict(self._summary)


def _select(
    store: InstanceStore,
    max_difficulty: int | None,
    require_potential: bool,
) -> NDArray[np.int64] | None:
    """Return the indices of the groups an episode may start from, `None` for all.

    Both filters are stated in terms of a group's distance to origin, so without an
    annotations file neither can be satisfied by any group. Nothing is filtered by
    length: the dataset was generated under this run's relator bound (checked when it
    is opened), so every group in it is one the encoder can hold.
    """
    if max_difficulty is None and not require_potential:
        return None
    distances = store.distances
    if distances is None:
        return np.empty(0, dtype=np.int64)
    keep = distances != UNKNOWN
    if max_difficulty is not None:
        keep &= distances <= max_difficulty
    return np.flatnonzero(keep)


def require_dataset_bound(groups_path: Path, max_relator_tokens: int) -> None:
    """Refuse to train a model on a dataset generated under a different relator bound.

    The bound is what makes the dataset's distances true *for this model*: a ball grown
    to `rel48` proves shortest paths through the graph a 48-token encoder moves in. Put
    a 32-token model on it and the labels point down descents whose next group it cannot
    represent and whose move the environment masks; grow the ball unbounded and every
    distance may route through a group no bounded model can enter. Neither is a filter
    away -- dropping the offending groups leaves the *remaining* distances wrong, since
    they were proven over paths that ran through the dropped ones.
    """
    stored = read_relator_bound(groups_path)
    if stored == max_relator_tokens:
        return
    generated = f"max_relator_length={stored}" if stored else "unbounded"
    raise ValueError(
        f"{groups_path} was generated {generated}, but this run's max_relator_tokens is "
        f"{max_relator_tokens}. Its distances are shortest paths through a different "
        f"graph than the one this model moves in. Train on a dataset grown under this "
        f"bound (`aczero dataset ball --max-relator-length {max_relator_tokens}`), or "
        f"set max_relator_tokens to match the dataset."
    )


def _constraint(max_difficulty: int | None, require_potential: bool) -> str:
    """Describe the filter that emptied a dataset, for the error that reports it."""
    if max_difficulty is not None:
        return f" with distance_to_origin <= {max_difficulty}"
    if require_potential:
        return " with a known distance_to_origin"
    return ""


def _summarize(
    path: Path,
    store: InstanceStore,
    selected: NDArray[np.int64] | None,
    max_difficulty: int | None,
) -> dict[str, float | int | bool | str]:
    """Build the log summary for a mapped group dataset.

    ``store.count`` groups live in the file and ``annotated`` of them carry a known
    distance to origin; ``selected`` are the ones left after the difficulty filter,
    or ``None`` when it kept them all.
    """
    distances = store.distances
    total = store.count
    annotated = 0 if distances is None else int(np.count_nonzero(distances != UNKNOWN))
    summary: dict[str, float | int | bool | str] = {
        "source": "dataset",
        "path": str(path),
        "rank": store.rank,
        "groups_total": total,
        "groups_used": total if selected is None else int(selected.size),
        "annotated": annotated,
        "annotated_pct": round(100.0 * annotated / total, 1) if total else 0.0,
        "instances": str(store.path),
    }
    if max_difficulty is not None:
        summary["max_difficulty"] = max_difficulty
    if distances is not None:
        used = distances if selected is None else distances[selected]
        used = used[used != UNKNOWN]
        if used.size:
            summary["distance_min"] = int(used.min())
            summary["distance_max"] = int(used.max())
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
            require_potential=config.reward_mode in ("potential", "navigation"),
            max_relator_tokens=config.max_relator_tokens,
        )
    return ScrambleSource(config.rank, config.scramble_depth)
