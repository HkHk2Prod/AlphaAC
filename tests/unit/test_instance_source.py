"""Tests for the self-play instance source (scramble vs. grown-dataset seeding)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.training.instance_source import (
    DatasetSource,
    ScrambleSource,
    build_instance_source,
)
from ac_zero.training.pipeline_config import TrainingPipelineConfig


def _write_dataset(path: Path, difficulties: list[int]) -> set[str]:
    """Write a minimal grown-dataset file; return the set of instance content hashes.

    Each instance is a distinct scramble so the hashes differ, letting a test
    assert which instances a source can return.
    """
    from ac_zero.datasets.generator import generate_solvable

    instances = []
    hashes = set()
    for index, difficulty in enumerate(difficulties):
        pres = generate_solvable(rank=2, depth=max(1, difficulty), seed=index).presentation
        entry = pres.to_json()
        entry["difficulty"] = difficulty
        instances.append(entry)
        hashes.add(pres.content_hash)
    path.write_text(json.dumps({"instances": instances}), encoding="utf-8")
    return hashes


def test_scramble_source_is_seed_deterministic() -> None:
    source = ScrambleSource(rank=2, depth=4)
    assert source.sample(7).content_hash == source.sample(7).content_hash
    # The standard presentation is AC-trivial and so are its scrambles.
    assert isinstance(source.sample(1), BalancedPresentation)


def test_dataset_source_samples_only_dataset_instances(tmp_path: Path) -> None:
    dataset = tmp_path / "train_rank2.json"
    hashes = _write_dataset(dataset, difficulties=[1, 2, 3, 4])
    source = DatasetSource.from_file(dataset)

    seen = {source.sample(seed).content_hash for seed in range(50)}
    assert seen <= hashes
    # A fixed seed always yields the same instance regardless of call order.
    assert source.sample(3).content_hash == source.sample(3).content_hash


def test_dataset_source_respects_max_difficulty(tmp_path: Path) -> None:
    dataset = tmp_path / "train_rank2.json"
    _write_dataset(dataset, difficulties=[1, 2, 5, 8])
    easy_hashes = {
        BalancedPresentation.from_json(entry).content_hash
        for entry in json.loads(dataset.read_text())["instances"]
        if entry["difficulty"] <= 2
    }
    source = DatasetSource.from_file(dataset, max_difficulty=2)

    seen = {source.sample(seed).content_hash for seed in range(50)}
    assert seen <= easy_hashes


def test_dataset_source_rejects_empty_selection(tmp_path: Path) -> None:
    dataset = tmp_path / "train_rank2.json"
    _write_dataset(dataset, difficulties=[5, 8])
    with pytest.raises(ValueError):
        DatasetSource.from_file(dataset, max_difficulty=2)


def test_build_instance_source_switches_on_config(tmp_path: Path) -> None:
    dataset = tmp_path / "train_rank2.json"
    _write_dataset(dataset, difficulties=[1, 2])

    scramble = build_instance_source(TrainingPipelineConfig(rank=2, scramble_depth=3))
    assert isinstance(scramble, ScrambleSource)

    seeded = build_instance_source(
        TrainingPipelineConfig(rank=2, dataset_path=str(dataset), dataset_max_difficulty=2)
    )
    assert isinstance(seeded, DatasetSource)


def _write_descent_dataset(path: Path) -> None:
    """Write a dataset where only some instances carry a proven descent distance."""
    from ac_zero.datasets.generator import generate_solvable

    instances = []
    for index, (distance, proven) in enumerate([(3, True), (None, False), (1, True)]):
        entry = generate_solvable(rank=2, depth=index + 1, seed=index).presentation.to_json()
        entry["difficulty"] = index + 1
        entry["descent_distance"] = distance
        entry["descent_proven"] = proven
        instances.append(entry)
    path.write_text(json.dumps({"instances": instances}), encoding="utf-8")


def test_descent_source_keeps_only_proven_and_stamps_distance(tmp_path: Path) -> None:
    dataset = tmp_path / "train_rank2.json"
    _write_descent_dataset(dataset)

    source = DatasetSource.from_file(dataset, require_descent=True)
    distances = {source.sample(seed).provenance["descent_distance"] for seed in range(50)}
    # Only the two proven-integer entries survive; the unproven one is dropped.
    assert distances == {3, 1}


def test_descent_source_rejects_dataset_without_proven_descent(tmp_path: Path) -> None:
    dataset = tmp_path / "train_rank2.json"
    _write_dataset(dataset, difficulties=[1, 2])  # no descent annotation at all
    with pytest.raises(ValueError, match="proven descent_distance"):
        DatasetSource.from_file(dataset, require_descent=True)


def test_build_instance_source_rejects_descent_without_dataset() -> None:
    with pytest.raises(ValueError, match="descent"):
        build_instance_source(TrainingPipelineConfig(rank=2, reward_mode="descent"))


def test_build_instance_source_wires_descent_distance_from_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "train_rank2.json"
    _write_descent_dataset(dataset)

    source = build_instance_source(
        TrainingPipelineConfig(rank=2, dataset_path=str(dataset), reward_mode="descent")
    )
    assert isinstance(source, DatasetSource)
    assert source.sample(0).provenance["descent_distance"] in {1, 3}
