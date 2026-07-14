"""Tests for closest-first generation: exact distances, complete shells, resume."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.annotate import annotation_path
from ac_zero.datasets.ball import BallConfig, ball_groups_path, grow_ball
from ac_zero.datasets.groups import read_relator_bound
from ac_zero.datasets.validation import validate_dataset
from ac_zero.moves.universal import UniversalCatalog, move_set

MOVESET = "strict-ac"
# Tight enough that the strict-AC moves actually run into it within a few shells.
BOUND = 6


def _grow(tmp_path: Path, target: int = 400, **kwargs: object) -> Path:
    groups = tmp_path / "ball.groups.json"
    grow_ball(groups, BallConfig(rank=2, moveset=MOVESET, target=target, workers=1, **kwargs))  # type: ignore[arg-type]
    return groups


def _documents(groups: Path) -> tuple[dict, dict]:
    annotations = annotation_path(groups, MOVESET)
    return json.loads(groups.read_text()), json.loads(annotations.read_text())


def _true_distances(rank: int, depth: int, bound: int = 0) -> dict[str, int]:
    """Breadth-first distances from the origin, computed independently of the store.

    ``bound`` restricts the walk to groups whose relators all fit it -- the graph a model
    of that encoder capacity moves in, which is the graph a bounded ball's distances are
    shortest paths through.
    """
    catalog = UniversalCatalog(rank)
    inverses = [catalog.move(i) for i in move_set(MOVESET, catalog).inverse_ids(catalog)]
    origin = BalancedPresentation.standard(rank)
    distances = {origin.content_hash: 0}
    queue = deque([(origin, 0)])
    while queue:
        presentation, distance = queue.popleft()
        if distance >= depth:
            continue
        for move in inverses:
            child = move.apply(presentation)
            if bound and any(len(relator.letters) > bound for relator in child.relators):
                continue
            if child.content_hash not in distances:
                distances[child.content_hash] = distance + 1
                queue.append((child, distance + 1))
    return distances


def test_the_origin_is_the_whole_ball_of_a_run_that_expands_nothing(tmp_path: Path) -> None:
    groups = _grow(tmp_path, target=0)
    document, annotations = _documents(groups)

    assert [entry["hash"] for entry in document["groups"]] == [
        BalancedPresentation.standard(2).content_hash
    ]
    assert annotations["annotations"][0]["distance_to_origin"] == 0
    assert document["provenance"]["complete_depth"] == 0


def test_distances_are_the_true_shortest_paths(tmp_path: Path) -> None:
    """Every distance the ball writes is the optimum, checked against an independent BFS."""
    groups = _grow(tmp_path, target=2000)
    document, annotations = _documents(groups)
    depth = document["provenance"]["max_distance"]
    truth = _true_distances(2, depth)

    labels = {entry["hash"]: entry["distance_to_origin"] for entry in annotations["annotations"]}
    assert labels
    for content_hash, distance in labels.items():
        assert distance == truth[content_hash]
        assert distance <= depth


def test_every_shell_up_to_the_complete_depth_is_whole(tmp_path: Path) -> None:
    """The claim the dataset makes: no group within `complete_depth` of the origin is missing."""
    groups = _grow(tmp_path, target=2000)
    document, annotations = _documents(groups)
    complete = document["provenance"]["complete_depth"]
    assert complete >= 2  # a run this size gets past the first shells

    present = {entry["hash"] for entry in annotations["annotations"]}
    expected = {
        content_hash
        for content_hash, distance in _true_distances(2, complete).items()
        if distance <= complete
    }
    assert expected <= present


def test_the_optimal_moves_step_one_closer_to_the_origin(tmp_path: Path) -> None:
    """Each co-optimal move listed for a group really lands on a group one closer."""
    groups = _grow(tmp_path, target=600)
    document, annotations = _documents(groups)
    catalog = UniversalCatalog(2)
    labels = {entry["hash"]: entry for entry in annotations["annotations"]}
    strict = move_set(MOVESET, catalog).ids

    checked = 0
    for entry in document["groups"]:
        label = labels[entry["hash"]]
        presentation = BalancedPresentation.from_letters(2, entry["relators"])
        if label["distance_to_origin"] == 0:
            assert label["optimal_moves_to_origin"] == []
            continue
        assert label["optimal_moves_to_origin"], "a group off the origin has a way back"
        for move_id in label["optimal_moves_to_origin"]:
            assert move_id in strict  # a forward move of the set, not one of its inverses
            child = catalog.move(move_id).apply(presentation)
            assert labels[child.content_hash]["distance_to_origin"] == (
                label["distance_to_origin"] - 1
            )
            checked += 1
    assert checked > 0


def test_a_resumed_run_extends_the_same_ball(tmp_path: Path) -> None:
    """Stopping and restarting deepens the ball rather than redoing or corrupting it."""
    groups = _grow(tmp_path, target=200)
    first, _ = _documents(groups)

    grow_ball(groups, BallConfig(rank=2, moveset=MOVESET, target=800, workers=1))
    second, annotations = _documents(groups)

    assert second["provenance"]["count"] > first["provenance"]["count"]
    assert second["provenance"]["complete_depth"] >= first["provenance"]["complete_depth"]
    # The groups it already had are still there, at the same distances, in the same order.
    before = [entry["hash"] for entry in first["groups"]]
    assert [entry["hash"] for entry in second["groups"]][: len(before)] == before
    labels = {entry["hash"]: entry["distance_to_origin"] for entry in annotations["annotations"]}
    truth = _true_distances(2, second["provenance"]["max_distance"])
    assert all(labels[h] == truth[h] for h in before)


def test_a_ball_of_another_move_set_is_refused(tmp_path: Path) -> None:
    groups = _grow(tmp_path, target=50)
    with pytest.raises(ValueError, match=r"strict-ac.* ball"):
        grow_ball(groups, BallConfig(rank=2, moveset="universal", target=50, workers=1))


def test_groups_that_drift_out_of_step_with_their_distances_are_refused(tmp_path: Path) -> None:
    """A resume pairs the two documents by position; it must not pair them silently wrong."""
    groups = _grow(tmp_path, target=200)
    annotations = annotation_path(groups, MOVESET)
    document = json.loads(annotations.read_text())
    document["annotations"].reverse()
    annotations.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="out of order"):
        grow_ball(groups, BallConfig(rank=2, moveset=MOVESET, target=50, workers=1))


def test_both_documents_validate(tmp_path: Path) -> None:
    groups = _grow(tmp_path, target=300)
    assert validate_dataset(groups).ok
    assert validate_dataset(annotation_path(groups, MOVESET)).ok


def test_a_checkpoint_leaves_a_resumable_file(tmp_path: Path) -> None:
    """A run that checkpoints mid-flight records how far it expanded, not just what it found."""
    groups = _grow(tmp_path, target=500, checkpoint_hours=1e-9)  # a checkpoint every merge
    document, _ = _documents(groups)

    expanded = document["ball"]["expanded"]
    assert 0 < expanded <= len(document["groups"])
    assert document["ball"]["moveset"] == MOVESET
    # The expanded prefix is exactly the groups closer than the first unexpanded one.
    assert document["provenance"]["expanded"] == expanded


def test_checkpoints_are_taken_on_the_clock_not_on_a_group_count(
    tmp_path: Path, monkeypatch
) -> None:
    """The interval is hours of work at risk, not groups added.

    A checkpoint rewrites both documents in full, so its cost grows with the ball while a
    group count does not -- and what the interval is really buying is a bound on the work
    an interruption can destroy, which is measured in time.
    """
    # A clock that advances half an hour on every read, so the run "spans" hours without
    # taking any. Swapped only inside `ball`, leaving the real clock alone elsewhere.
    now = [0.0]

    def tick() -> float:
        now[0] += 1800.0
        return now[0]

    monkeypatch.setattr("ac_zero.datasets.ball.time", SimpleNamespace(monotonic=tick))

    taken: list[int] = []

    def progress(message: str, metrics: dict) -> None:
        if message == "checkpoint":
            taken.append(int(metrics["added"]))

    groups = tmp_path / "ball.groups.json"
    grow_ball(
        groups,
        BallConfig(rank=2, moveset=MOVESET, target=600, workers=1, checkpoint_hours=1.0),
        progress=progress,
    )
    # The clock, not the group count, decided: half-hour merges under a one-hour interval.
    assert taken, "a run spanning hours must checkpoint"
    assert read_relator_bound(groups) == 0
    # And nothing was written between them: a zero interval writes only at the end.
    quiet: list[int] = []
    grow_ball(
        tmp_path / "quiet.groups.json",
        BallConfig(rank=2, moveset=MOVESET, target=600, workers=1, checkpoint_hours=0.0),
        progress=lambda message, metrics: quiet.append(1) if message == "checkpoint" else None,
    )
    assert not quiet


# -- the relator bound -------------------------------------------------------


def test_the_bound_appears_in_the_default_name() -> None:
    """A ball grown under a different bound is a different dataset, so it is a different
    file: the name says which graph the distances inside were proven in."""
    assert ball_groups_path("data/generated", 2, 48) == Path(
        "data/generated/ball_rank2_rel48.groups.json"
    )
    # Its companions derive from that stem, so they inherit the bound for free.
    assert annotation_path(ball_groups_path("data/generated", 2, 48), MOVESET) == Path(
        f"data/generated/ball_rank2_rel48.{MOVESET}.annotations.json"
    )
    # An unbounded ball -- the whole graph, no model attached -- keeps the plain name.
    assert ball_groups_path("data/generated", 2, 0) == Path("data/generated/ball_rank2.groups.json")


def test_a_bounded_ball_holds_only_groups_within_the_bound(tmp_path: Path) -> None:
    groups = _grow(tmp_path, target=800, max_relator_length=BOUND)
    document, _ = _documents(groups)

    longest = max(len(relator) for entry in document["groups"] for relator in entry["relators"])
    assert longest <= BOUND
    # Only each relator is bounded, never their sum: a presentation may grow past it.
    assert max(entry["total_length"] for entry in document["groups"]) > BOUND


def test_a_bounded_ball_records_the_bound_it_was_grown_under(tmp_path: Path) -> None:
    """The bound is part of the dataset, not of whoever reads it."""
    groups = _grow(tmp_path, target=200, max_relator_length=BOUND)
    document, annotations = _documents(groups)

    assert read_relator_bound(groups) == BOUND
    assert document["provenance"]["max_relator_length"] == BOUND
    assert annotations["provenance"]["max_relator_length"] == BOUND
    # An unbounded ball says so rather than saying nothing.
    unbounded = _grow(tmp_path / "free", target=50)
    assert read_relator_bound(unbounded) == 0


def test_a_bounded_ball_proves_shortest_paths_through_its_own_graph(tmp_path: Path) -> None:
    """The distances are optimal *in the bounded graph* -- the one the model moves in.

    A path that detours through an over-long group is not a path the environment would
    let a model walk, so it is not one the dataset may claim a distance over.
    """
    groups = _grow(tmp_path, target=2000, max_relator_length=BOUND)
    document, annotations = _documents(groups)
    depth = document["provenance"]["max_distance"]
    truth = _true_distances(2, depth, bound=BOUND)

    labels = {entry["hash"]: entry["distance_to_origin"] for entry in annotations["annotations"]}
    assert labels
    for content_hash, distance in labels.items():
        assert distance == truth[content_hash]


def test_a_ball_cannot_be_extended_under_a_different_bound(tmp_path: Path) -> None:
    """Adding groups past the bound would reroute the shortest paths already written."""
    groups = _grow(tmp_path, target=100, max_relator_length=BOUND)
    with pytest.raises(ValueError, match=f"is a max_relator_length={BOUND} ball"):
        grow_ball(
            groups,
            BallConfig(rank=2, moveset=MOVESET, target=10, workers=1, max_relator_length=BOUND + 4),
        )
