import json
from pathlib import Path

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.datasets.validation import validate_mapping
from ac_zero.moves.primitive import move_from_json


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def test_first_run_seeds_trivial_root_and_expands(tmp_path) -> None:
    path = tmp_path / "g.json"
    report = grow_dataset(path, GrowConfig(rank=2, target=20, workers=1))
    data = _load(path)
    assert data["schema_version"] == "aczero-dataset-v3"
    # Exactly one root: the trivial standard presentation, with no construction edge.
    roots = [entry for entry in data["instances"] if not entry["predecessors"]]
    assert len(roots) == 1
    assert roots[0]["content_hash"] == BalancedPresentation.standard(2).content_hash
    assert roots[0]["difficulty"] == 0
    assert roots[0]["optimal"] is True
    assert report.added >= 20
    assert validate_mapping(data).ok


def test_predecessor_edges_reconstruct_their_child(tmp_path) -> None:
    path = tmp_path / "g.json"
    grow_dataset(path, GrowConfig(rank=2, target=40, workers=1))
    instances = _load(path)["instances"]
    by_hash = {entry["content_hash"]: entry for entry in instances}
    for entry in instances:
        assert entry["ac_trivial"] is True
        for edge in entry["predecessors"]:
            parent = BalancedPresentation.from_json(by_hash[edge["parent_hash"]])
            child = move_from_json(edge["move"]).apply(parent)
            # The recorded move genuinely maps the parent group onto this group.
            assert child.content_hash == entry["content_hash"]


def test_records_multiple_co_optimal_construction_moves(tmp_path) -> None:
    # A group reachable several equally short ways keeps every such move -- the
    # multi-modal supervised-learning target the dataset is built to capture.
    path = tmp_path / "g.json"
    grow_dataset(path, GrowConfig(rank=2, target=60, workers=1))
    instances = _load(path)["instances"]
    multi = [entry for entry in instances if len(entry["predecessors"]) > 1]
    assert multi, "expected at least one group with several co-optimal constructions"
    # Every non-root group carries at least one construction edge; the root none.
    for entry in instances:
        expected = 0 if entry["difficulty"] == 0 else 1
        assert len(entry["predecessors"]) >= expected


def test_runs_expand_the_database_without_duplicates(tmp_path) -> None:
    path = tmp_path / "g.json"
    first = grow_dataset(path, GrowConfig(rank=2, target=20, workers=1))
    before = len(_load(path)["instances"])
    second = grow_dataset(path, GrowConfig(rank=2, target=20, workers=1))
    after = _load(path)["instances"]
    hashes = [entry["content_hash"] for entry in after]
    # The second run resumes from the accumulated frontier and only ever grows.
    assert len(after) > before
    assert len(set(hashes)) == len(hashes)
    assert second.total > first.total
    # A single root survives across runs; it is never re-added.
    assert sum(1 for entry in after if not entry["predecessors"]) == 1


def test_target_zero_round_trips_the_file(tmp_path) -> None:
    path = tmp_path / "g.json"
    grow_dataset(path, GrowConfig(rank=2, target=15, workers=1))
    before = _load(path)
    grow_dataset(path, GrowConfig(rank=2, target=0, workers=1))
    after = _load(path)
    # Loading and rewriting with nothing to add preserves every group.
    assert {e["content_hash"] for e in before["instances"]} == {
        e["content_hash"] for e in after["instances"]
    }


def test_respects_total_length_cap(tmp_path) -> None:
    path = tmp_path / "g.json"
    grow_dataset(path, GrowConfig(rank=2, target=200, total_length_cap=8, workers=1))
    instances = _load(path)["instances"]
    for entry in instances:
        pres = BalancedPresentation.from_json(entry)
        assert pres.total_length <= 8


def test_smallest_selection_is_deterministic(tmp_path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    grow_dataset(a, GrowConfig(rank=2, target=30, select="smallest", workers=1))
    grow_dataset(b, GrowConfig(rank=2, target=30, select="smallest", workers=1))
    assert _load(a)["instances"] == _load(b)["instances"]


def test_weighted_random_diverges_by_seed_but_repeats_by_seed(tmp_path) -> None:
    def run(name: str, seed: int) -> set[str]:
        path = tmp_path / name
        config = GrowConfig(rank=2, target=40, select="weighted-random", seed=seed, workers=1)
        grow_dataset(path, config)
        return {entry["content_hash"] for entry in _load(path)["instances"]}

    same_a = run("a.json", seed=1)
    same_b = run("b.json", seed=1)
    other = run("c.json", seed=99)
    assert same_a == same_b  # a seed fully determines the walk
    assert same_a != other  # different machines (seeds) explore different paths


def test_checkpoints_dump_a_valid_snapshot_without_changing_the_result(tmp_path) -> None:
    checkpointed = tmp_path / "chk.json"
    events: list[str] = []
    grow_dataset(
        checkpointed,
        GrowConfig(rank=2, target=40, checkpoint_every=10, workers=1),
        progress=lambda message, metrics: events.append(message),
    )
    # At least one mid-run checkpoint fired, and the final file still validates.
    assert "checkpoint" in events
    assert validate_mapping(_load(checkpointed)).ok

    # Checkpointing is purely a durability snapshot: it never alters the outcome.
    end_only = tmp_path / "end.json"
    grow_dataset(end_only, GrowConfig(rank=2, target=40, checkpoint_every=0, workers=1))
    assert _load(checkpointed)["instances"] == _load(end_only)["instances"]


def test_multiprocess_run_produces_a_valid_dataset(tmp_path) -> None:
    path = tmp_path / "g.json"
    report = grow_dataset(path, GrowConfig(rank=2, target=50, workers=4))
    data = _load(path)
    assert validate_mapping(data).ok
    assert report.added >= 50
    assert report.total == len(data["instances"])
