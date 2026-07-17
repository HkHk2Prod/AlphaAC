"""Tests for mapping dataset filenames to their Hugging Face bucket folders."""

from __future__ import annotations

from pathlib import Path

from ac_zero.datasets.remote_paths import ball_remote_dir, dataset_remote_name


def test_ball_remote_dir_encodes_rank_and_bound() -> None:
    assert ball_remote_dir(2, 48) == "datasets/rank2/rel-48"
    assert ball_remote_dir(3, 12) == "datasets/rank3/rel-12"


def test_ball_remote_dir_unbounded() -> None:
    # An unbounded ball (bound 0 or negative) is filed under rel-unbounded, never rel-0.
    assert ball_remote_dir(2, 0) == "datasets/rank2/rel-unbounded"
    assert ball_remote_dir(2, -1) == "datasets/rank2/rel-unbounded"


def test_groups_file_sits_at_the_ball_root() -> None:
    # The graph is move-set-independent, so it stays at the ball root, not under a move set.
    assert (
        dataset_remote_name("ball_rank2_rel48.groups.json")
        == "datasets/rank2/rel-48/ball_rank2_rel48.groups.json"
    )


def test_annotations_go_under_the_move_set() -> None:
    assert (
        dataset_remote_name("ball_rank2_rel48.strict-ac.annotations.json")
        == "datasets/rank2/rel-48/strict-ac/ball_rank2_rel48.strict-ac.annotations.json"
    )


def test_annotation_summary_sits_beside_its_annotations() -> None:
    assert (
        dataset_remote_name("ball_rank2_rel48.strict-ac.annotations.summary.md")
        == "datasets/rank2/rel-48/strict-ac/ball_rank2_rel48.strict-ac.annotations.summary.md"
    )


def test_groups_summary_sits_at_the_ball_root() -> None:
    assert (
        dataset_remote_name("ball_rank2_rel48.groups.summary.md")
        == "datasets/rank2/rel-48/ball_rank2_rel48.groups.summary.md"
    )


def test_unbounded_ball_filename_maps_to_rel_unbounded() -> None:
    assert (
        dataset_remote_name("ball_rank2.groups.json")
        == "datasets/rank2/rel-unbounded/ball_rank2.groups.json"
    )


def test_length_first_train_dataset_is_filed_by_rank() -> None:
    # No relator bound in the name -> rel-unbounded; the raw .json stays at the ball root.
    assert (
        dataset_remote_name("train_rank2.json") == "datasets/rank2/rel-unbounded/train_rank2.json"
    )
    assert (
        dataset_remote_name("train_rank2.strict-ac.annotations.json")
        == "datasets/rank2/rel-unbounded/strict-ac/train_rank2.strict-ac.annotations.json"
    )


def test_accepts_a_full_local_path_using_only_the_basename() -> None:
    assert (
        dataset_remote_name(Path("/kaggle/working/ball_rank2_rel48.groups.json"))
        == "datasets/rank2/rel-48/ball_rank2_rel48.groups.json"
    )


def test_nonconforming_name_stays_at_the_bucket_root() -> None:
    # A hand-named upload with no rank marker is left where the caller put it.
    assert dataset_remote_name("scratch.json") == "scratch.json"
    assert dataset_remote_name("notes.md") == "notes.md"
