"""Tests for the self-play instance source (scramble vs. grown-dataset seeding)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable
from ac_zero.datasets.groups import MOVE_CATALOG, SCHEMA_VERSION, group_entry
from ac_zero.training.pipeline.instance_source import (
    DatasetSource,
    ScrambleSource,
    build_instance_source,
)
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig

_ANNOTATIONS_SCHEMA = "aczero-annotations-v1"


def _write_groups(path: Path, presentations: list[BalancedPresentation]) -> None:
    groups = [group_entry(p, ac_trivial=True, source="universal_expansion") for p in presentations]
    document = {
        "schema_version": SCHEMA_VERSION,
        "rank": 2,
        "move_catalog": MOVE_CATALOG,
        "groups": groups,
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_annotations(path: Path, rows: dict[str, dict], moveset: str = "strict-ac") -> None:
    annotations = [
        {
            "hash": h,
            "distance_to_origin": row.get("origin"),
            "optimal_moves_to_origin": [],
            "distance_to_shorter": row.get("shorter"),
            "optimal_moves_to_shorter": [],
            "shorter_proven": row.get("proven", False),
            "optimal": row.get("origin") is not None,
        }
        for h, row in rows.items()
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": _ANNOTATIONS_SCHEMA,
                "rank": 2,
                "moveset": moveset,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )


def _presentations(depths: list[int]) -> list[BalancedPresentation]:
    return [
        generate_solvable(rank=2, depth=max(1, d), seed=i).presentation
        for i, d in enumerate(depths)
    ]


def test_scramble_source_is_seed_deterministic() -> None:
    source = ScrambleSource(rank=2, depth=4)
    assert source.sample(7).content_hash == source.sample(7).content_hash
    assert isinstance(source.sample(1), BalancedPresentation)


def test_dataset_source_samples_only_dataset_groups(tmp_path: Path) -> None:
    presentations = _presentations([1, 2, 3, 4])
    dataset = tmp_path / "train.groups.json"
    _write_groups(dataset, presentations)
    hashes = {p.content_hash for p in presentations}
    source = DatasetSource.from_file(dataset)

    seen = {source.sample(seed).content_hash for seed in range(50)}
    assert seen <= hashes
    assert source.sample(3).content_hash == source.sample(3).content_hash


def test_dataset_source_respects_max_difficulty(tmp_path: Path) -> None:
    presentations = _presentations([1, 2, 5, 8])
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, presentations)
    # Distance to origin mirrors the scramble depth here.
    _write_annotations(
        annotations,
        {p.content_hash: {"origin": d} for p, d in zip(presentations, [1, 2, 5, 8], strict=True)},
        moveset="universal",
    )
    easy = {p.content_hash for p, d in zip(presentations, [1, 2, 5, 8], strict=True) if d <= 2}
    source = DatasetSource.from_file(dataset, annotations, max_difficulty=2)

    seen = {source.sample(seed).content_hash for seed in range(50)}
    assert seen == easy


def test_dataset_source_rejects_empty_selection(tmp_path: Path) -> None:
    presentations = _presentations([5, 8])
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, presentations)
    _write_annotations(
        annotations, {p.content_hash: {"origin": 5} for p in presentations}, moveset="universal"
    )
    with pytest.raises(ValueError):
        DatasetSource.from_file(dataset, annotations, max_difficulty=2)


def test_dataset_source_exposes_known_distances_as_potentials(tmp_path: Path) -> None:
    present, absent = _presentations([2])[0], _presentations([3])[0]
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, [present, absent])
    # Only one group has a known distance to origin; the other is left unresolved.
    _write_annotations(
        annotations,
        {present.content_hash: {"origin": 4}, absent.content_hash: {"origin": None}},
        moveset="universal",
    )
    source = DatasetSource.from_file(dataset, annotations)
    assert source.potentials == {present.content_hash: 4}


def test_dataset_source_require_potential_drops_unannotated_groups(tmp_path: Path) -> None:
    present, absent = _presentations([2])[0], _presentations([3])[0]
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, [present, absent])
    _write_annotations(
        annotations,
        {present.content_hash: {"origin": 4}, absent.content_hash: {"origin": None}},
        moveset="universal",
    )
    source = DatasetSource.from_file(dataset, annotations, require_potential=True)
    seen = {source.sample(seed).content_hash for seed in range(50)}
    assert seen == {present.content_hash}


def test_potential_reward_mode_seeds_only_from_known_distance_groups(tmp_path: Path) -> None:
    present, absent = _presentations([2])[0], _presentations([3])[0]
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, [present, absent])
    _write_annotations(
        annotations,
        {present.content_hash: {"origin": 4}, absent.content_hash: {"origin": None}},
        moveset="universal",
    )
    source = build_instance_source(
        TrainingPipelineConfig(
            rank=2,
            reward_mode="potential",
            dataset_path=str(dataset),
            dataset_annotations_path=str(annotations),
        )
    )
    assert isinstance(source, DatasetSource)
    assert source.potentials == {present.content_hash: 4}
    assert {source.sample(seed).content_hash for seed in range(50)} == {present.content_hash}


def _distance_dataset(tmp_path: Path, distances: list[int | None]) -> DatasetSource:
    """A dataset whose four groups carry the given distances to origin (None = absent)."""
    presentations = _presentations([1, 2, 3, 4])
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, presentations)
    _write_annotations(
        annotations,
        {p.content_hash: {"origin": d} for p, d in zip(presentations, distances, strict=True)},
        moveset="universal",
    )
    source = DatasetSource.from_file(dataset, annotations)
    source._by_hash = {p.content_hash: d for p, d in zip(presentations, distances, strict=True)}
    return source


def test_max_distance_sampling_only_returns_problems_at_or_below_lmax(tmp_path: Path) -> None:
    # Sanity checks 3 & 4: the sampler caps L at max_distance and imposes no lower
    # bound -- every distance from 1 up to L_max is eligible.
    source = _distance_dataset(tmp_path, [1, 2, 3, 8])
    seen = {source.sample(seed, max_distance=3).content_hash for seed in range(80)}
    distances = {source._by_hash[h] for h in seen}
    assert distances == {1, 2, 3}  # 1 (well below L_max) included; 8 (> L_max) excluded


def test_max_distance_sampling_excludes_unreachable_and_trivial(tmp_path: Path) -> None:
    # Distance 0 is the destination itself (trivial) and None is unreachable; both
    # must be excluded even though they satisfy L <= L_max.
    source = _distance_dataset(tmp_path, [0, None, 2, 3])
    seen = {source.sample(seed, max_distance=5).content_hash for seed in range(80)}
    assert {source._by_hash[h] for h in seen} == {2, 3}


def test_max_distance_sampling_raises_when_no_eligible_problems(tmp_path: Path) -> None:
    source = _distance_dataset(tmp_path, [5, 6, 7, 8])
    with pytest.raises(ValueError, match="0 < distance_to_origin <= 3"):
        source.sample(0, max_distance=3)


def test_max_distance_sampling_is_seed_deterministic(tmp_path: Path) -> None:
    source = _distance_dataset(tmp_path, [1, 2, 3, 4])
    assert (
        source.sample(11, max_distance=4).content_hash
        == source.sample(11, max_distance=4).content_hash
    )


def test_scramble_source_rejects_distance_curriculum_sampling() -> None:
    with pytest.raises(ValueError, match="distance-annotated dataset"):
        ScrambleSource(rank=2, depth=3).sample(0, max_distance=2)


def test_scramble_source_has_no_potentials() -> None:
    assert ScrambleSource(rank=2, depth=3).potentials == {}


def test_scramble_source_describes_itself() -> None:
    assert ScrambleSource(rank=2, depth=6).describe() == {
        "source": "scramble",
        "rank": 2,
        "depth": 6,
    }


def test_dataset_source_describe_reports_group_and_annotation_stats(tmp_path: Path) -> None:
    presentations = _presentations([1, 2, 5, 8])
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, presentations)
    # Three of four groups carry a known distance; one is left unresolved.
    rows = {p.content_hash: {"origin": d} for p, d in zip(presentations, [1, 2, 5], strict=False)}
    rows[presentations[3].content_hash] = {"origin": None}
    _write_annotations(annotations, rows, moveset="universal")

    summary = DatasetSource.from_file(dataset, annotations, max_difficulty=2).describe()
    assert summary["source"] == "dataset"
    assert summary["path"] == str(dataset)
    assert summary["groups_total"] == 4
    # max_difficulty=2 keeps only the two easiest groups.
    assert summary["groups_used"] == 2
    # Annotation coverage is over the whole file, not the filtered subset.
    assert summary["annotated"] == 3
    assert summary["annotated_pct"] == 75.0
    assert summary["max_difficulty"] == 2
    assert summary["distance_min"] == 1
    assert summary["distance_max"] == 2


def test_build_instance_source_switches_on_config(tmp_path: Path) -> None:
    presentations = _presentations([1, 2])
    dataset = tmp_path / "train.groups.json"
    annotations = tmp_path / "train.universal.annotations.json"
    _write_groups(dataset, presentations)
    _write_annotations(
        annotations, {p.content_hash: {"origin": 1} for p in presentations}, moveset="universal"
    )

    scramble = build_instance_source(TrainingPipelineConfig(rank=2, scramble_depth=3))
    assert isinstance(scramble, ScrambleSource)

    seeded = build_instance_source(
        TrainingPipelineConfig(
            rank=2,
            dataset_path=str(dataset),
            dataset_annotations_path=str(annotations),
            dataset_max_difficulty=2,
        )
    )
    assert isinstance(seeded, DatasetSource)
