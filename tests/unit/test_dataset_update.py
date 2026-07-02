import json
from pathlib import Path

from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.datasets.labels import UNKNOWN, known_solution, known_trivial, merge_labels
from ac_zero.datasets.update import (
    BreadthFirstStrategy,
    dedupe_entries,
    improve_dataset,
    label_from_entry,
)

_FAST_BFS = BreadthFirstStrategy(
    max_moves=6, total_length_cap=28, max_expansions=300, max_generated=3000
)


def _grow_fixture(path: Path, *, target: int, seed: int = 0) -> None:
    """Grow a small deterministic rank-2 dataset for the improvement tests."""
    grow_dataset(path, GrowConfig(rank=2, target=target, select="smallest", seed=seed, workers=1))


def test_merge_never_replaces_a_better_solution_with_a_worse_one() -> None:
    merged = merge_labels(known_solution(4, optimal=True), known_solution(9))
    assert merged.minimal_known_operations == 4
    assert merged.optimal is True


def test_merge_prefers_the_shorter_solution_and_keeps_triviality() -> None:
    merged = merge_labels(known_solution(9), known_solution(5))
    assert merged.minimal_known_operations == 5
    assert merged.ac_trivial is True


def test_merge_unknown_does_not_demote_known() -> None:
    assert merge_labels(known_trivial(), UNKNOWN).ac_trivial is True
    assert merge_labels(known_solution(3, optimal=True), UNKNOWN).minimal_known_operations == 3


def test_dedupe_merges_duplicates_keeping_best_label_and_min_difficulty() -> None:
    base = {"content_hash": "h", "rank": 2, "relators": [[1]]}
    worse = {**base, "difficulty": 9, **known_solution(9).to_json()}
    better = {**base, "difficulty": 4, **known_solution(4, optimal=True).to_json()}
    unique, duplicates = dedupe_entries([worse, better, worse])
    assert len(unique) == 1
    assert duplicates == 2
    assert unique[0]["minimal_known_operations"] == 4
    assert unique[0]["optimal"] is True
    assert unique[0]["difficulty"] == 4


def test_improve_dataset_only_improves_and_is_idempotent(tmp_path) -> None:
    path = tmp_path / "ds.json"
    _grow_fixture(path, target=12, seed=0)
    instances = json.loads(path.read_text())["instances"]
    before = {i["content_hash"]: i["minimal_known_operations"] for i in instances}

    first = improve_dataset(path, strategies=[_FAST_BFS], max_difficulty=6)
    after = json.loads(path.read_text())["instances"]
    by_hash = {i["content_hash"]: i for i in after}

    # protection: no entry's known solution ever got longer
    for content, old_len in before.items():
        assert by_hash[content]["minimal_known_operations"] <= old_len
    assert first.improved >= 1

    # re-running changes nothing and skips proven-optimal entries
    second = improve_dataset(path, strategies=[_FAST_BFS], max_difficulty=6)
    assert second.improved == 0
    assert second.searched < first.searched


def test_improve_does_not_overwrite_an_existing_better_label(tmp_path) -> None:
    path = tmp_path / "ds.json"
    _grow_fixture(path, target=8, seed=1)
    data = json.loads(path.read_text())
    # plant an artificially strong (short, optimal) label on every entry
    for entry in data["instances"]:
        entry.update(known_solution(1, optimal=True).to_json())
    path.write_text(json.dumps(data))

    improve_dataset(path, strategies=[_FAST_BFS], max_difficulty=8)
    for entry in json.loads(path.read_text())["instances"]:
        # a planted optimal label is never regressed by a longer search result
        assert entry["minimal_known_operations"] == 1
        assert entry["optimal"] is True


def test_improve_dataset_reports_progress(tmp_path) -> None:
    path = tmp_path / "ds.json"
    _grow_fixture(path, target=12, seed=0)

    events: list[tuple[str, dict]] = []
    report = improve_dataset(
        path,
        strategies=[_FAST_BFS],
        max_difficulty=6,
        progress=lambda message, metrics: events.append((message, metrics)),
    )

    messages = [message for message, _ in events]
    # The run opens with a full task description naming the input, output, search
    # strategies, and difficulty gate.
    assert messages[0] == "improving dataset"
    descriptor = events[0][1]
    assert descriptor["input"] == str(path)
    assert "breadth_first" in descriptor["strategies"]
    assert descriptor["max_difficulty"] == 6
    assert "deduplicated entries" in messages
    # the last improvement event accounts for every entry in the dataset
    final = next(metrics for message, metrics in reversed(events) if message == "improving dataset")
    assert final["processed"] == report.total
    assert final["total"] == report.total


def test_improve_dataset_is_identical_across_worker_counts(tmp_path) -> None:
    sequential = tmp_path / "seq.json"
    parallel = tmp_path / "par.json"
    _grow_fixture(sequential, target=16, seed=0)
    # An identical input dataset under both worker counts.
    parallel.write_text(sequential.read_text())

    seq_report = improve_dataset(sequential, strategies=[_FAST_BFS], max_difficulty=6, workers=1)
    par_report = improve_dataset(parallel, strategies=[_FAST_BFS], max_difficulty=6, workers=2)

    assert seq_report == par_report
    # Fanning the per-entry searches across processes does not change the result.
    assert json.loads(sequential.read_text()) == json.loads(parallel.read_text())


def test_label_round_trips_through_entry_dict() -> None:
    entry = {"rank": 2, "relators": [[1]], **known_solution(7, optimal=True).to_json()}
    label = label_from_entry(entry)
    assert label.minimal_known_operations == 7
    assert label.optimal is True
    assert label.ac_trivial is True
