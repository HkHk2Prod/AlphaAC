import json

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_dataset, generate_solvable, write_dataset


def test_generate_solvable_reports_difficulty() -> None:
    instance = generate_solvable(rank=2, depth=6, seed=3)
    assert 0 <= instance.difficulty <= 6
    assert instance.presentation.provenance["difficulty"] == instance.difficulty


def test_generate_dataset_is_identical_across_worker_counts() -> None:
    sequential = generate_dataset(rank=2, count=40, depths=[2, 3, 4], seed=0, min_total_length=3)
    parallel = generate_dataset(
        rank=2, count=40, depths=[2, 3, 4], seed=0, min_total_length=3, workers=4
    )
    # Candidates are built out of order across processes but re-sorted, so the
    # deduplicated, length-filtered dataset is byte-identical to the serial run.
    assert [inst.presentation.to_json() for inst in sequential] == [
        inst.presentation.to_json() for inst in parallel
    ]
    assert [inst.difficulty for inst in sequential] == [inst.difficulty for inst in parallel]


def test_generate_dataset_is_deduplicated_and_nontrivial() -> None:
    instances = generate_dataset(rank=2, count=60, depth=10, seed=0)
    assert len(instances) == 60
    trivial = BalancedPresentation.standard(2).content_hash
    hashes = {inst.presentation.content_hash for inst in instances}
    assert len(hashes) == 60
    assert trivial not in hashes
    assert all(1 <= inst.difficulty <= 10 for inst in instances)


def test_generate_dataset_depths_span_a_difficulty_range() -> None:
    instances = generate_dataset(rank=2, count=80, depths=[2, 6, 12], seed=0)
    difficulties = {inst.difficulty for inst in instances}
    assert min(difficulties) <= 3
    assert max(difficulties) >= 8
    assert len({inst.presentation.content_hash for inst in instances}) == 80


def test_generate_dataset_respects_length_constraints() -> None:
    instances = generate_dataset(
        rank=2, count=10, depth=12, seed=5, min_total_length=10, min_relator_length=2
    )
    for instance in instances:
        lengths = [len(relator) for relator in instance.presentation.relators]
        assert sum(lengths) >= 10
        assert min(lengths) >= 2


def test_generate_dataset_raises_when_budget_exhausted() -> None:
    with pytest.raises(ValueError, match="distinct presentations"):
        generate_dataset(rank=2, count=100, depth=1, seed=0, max_attempts=20)


def test_write_dataset_emits_v2_with_difficulty_labels(tmp_path) -> None:
    path = tmp_path / "set.json"
    write_dataset(path, rank=2, count=12, depth=9, seed=1)
    data = json.loads(path.read_text())
    assert data["schema_version"] == "aczero-dataset-v2"
    assert len(data["instances"]) == 12
    assert all("difficulty" in instance for instance in data["instances"])
    assert data["provenance"]["count"] == 12
    assert data["provenance"]["min_difficulty"] <= data["provenance"]["max_difficulty"]
    for instance in data["instances"]:
        # every instance round-trips through the presentation parser
        BalancedPresentation.from_json(instance)
        # generated instances are known AC-trivial with a known, non-optimal solution
        assert instance["ac_trivial"] is True
        assert instance["minimal_known_operations"] >= 1
        assert instance["optimal"] is False


def test_generate_dataset_reports_progress() -> None:
    events: list[tuple[str, dict]] = []
    generate_dataset(
        rank=2, count=20, depth=10, seed=0, progress=lambda m, k: events.append((m, k))
    )

    messages = [message for message, _ in events]
    assert "generating instances" in messages
    # the final summary always fires and accounts for every accepted instance
    completion = next(metrics for message, metrics in events if message == "generation complete")
    assert completion["generated"] == 20
    assert completion["attempts"] >= 20


def test_write_dataset_reports_start_and_write(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    write_dataset(
        tmp_path / "set.json",
        rank=2,
        count=8,
        depth=9,
        seed=1,
        progress=lambda m, k: events.append((m, k)),
    )

    messages = [message for message, _ in events]
    # The run opens with a full task description naming every shaping parameter.
    assert messages[0] == "generating dataset"
    descriptor = events[0][1]
    assert descriptor["rank"] == 2
    assert descriptor["count"] == 8
    assert descriptor["depths"] == "[9]"
    assert descriptor["seed"] == 1
    assert messages[-1] == "dataset written"
    assert events[-1][1]["instances"] == 8
