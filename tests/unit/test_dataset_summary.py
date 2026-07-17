import json

from ac_zero.datasets.summary import (
    render_annotation_markdown,
    render_markdown,
    summarize,
    summarize_annotations,
    summary_path_for,
    write_annotation_summary,
    write_dataset_summary,
)


def _dataset() -> dict:
    # Root (expanded, 2 transitions), one expanded child (1 transition), one
    # unexpanded frontier group. Sizes are 2, 3, 3.
    return {
        "rank": 2,
        "provenance": {"generator": "universal_graph_expansion"},
        "groups": [
            {
                "relators": [[1], [2]],
                "total_length": 2,
                "source": "trivial",
                "ac_trivial": True,
                "transitions": {"0": "h1", "3": "h2"},
            },
            {
                "relators": [[1, 2], [2]],
                "total_length": 3,
                "source": "universal_expansion",
                "ac_trivial": True,
                "transitions": {"5": "h0"},
            },
            {
                "relators": [[1], [2, 1]],
                "total_length": 3,
                "source": "universal_expansion",
                "ac_trivial": None,
            },
        ],
    }


def test_summarize_buckets_every_dimension() -> None:
    summary = summarize(_dataset())
    assert summary.rank == 2
    assert summary.generator == "universal_graph_expansion"
    assert summary.total_groups == 3
    assert summary.exhausted == 2
    assert summary.frontier == 1
    assert summary.ac_trivial == 2
    assert summary.ac_unknown == 1
    assert summary.by_source == {"universal_expansion": 2, "trivial": 1}
    assert summary.total_length.counts == {2: 1, 3: 2}
    assert summary.transition_degree.counts == {1: 1, 2: 1}


def test_distribution_reports_min_max_mean() -> None:
    length = summarize(_dataset()).total_length
    assert length.minimum == 2
    assert length.maximum == 3
    assert length.population == 3
    assert length.mean == (2 + 3 + 3) / 3


def test_empty_dataset_summarizes_without_error() -> None:
    summary = summarize({"rank": 2, "groups": [], "provenance": {}})
    assert summary.total_groups == 0
    assert summary.total_length.counts == {}
    assert summary.total_length.mean is None
    assert "_No data._" in render_markdown(summary, name="empty.json")


def test_render_markdown_has_sections_and_table_rows() -> None:
    report = render_markdown(summarize(_dataset()), name="g.json")
    assert report.startswith("# Dataset summary: g.json")
    for heading in (
        "## By source",
        "## By size (total relator length)",
        "## By transition degree",
    ):
        assert heading in report
    # Two groups have size 3; the histogram row reflects that.
    assert "| 3 | 2 |" in report


def test_write_dataset_summary_targets_summary_dir(tmp_path) -> None:
    dataset_path = tmp_path / "generated" / "g.groups.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(json.dumps(_dataset()), encoding="utf-8")
    summary_dir = tmp_path / "summaries"

    written = write_dataset_summary(dataset_path, summary_dir)

    assert written == summary_path_for(dataset_path, summary_dir)
    assert written == summary_dir / "g.groups.summary.md"
    assert written.read_text(encoding="utf-8").startswith("# Dataset summary: g.groups.json")


def _annotations() -> dict:
    # Two groups reach the origin (distances 1, 2); one descends to a shorter
    # group (distance 1); the last is unresolved (shorter not proven).
    return {
        "rank": 2,
        "moveset": "strict-ac",
        "move_catalog": "univ-v1",
        "annotations": [
            {
                "hash": "root",
                "distance_to_origin": 0,
                "distance_to_shorter": None,
                "shorter_proven": True,
                "optimal": True,
            },
            {
                "hash": "h1",
                "distance_to_origin": 1,
                "distance_to_shorter": 1,
                "shorter_proven": True,
                "optimal": True,
            },
            {
                "hash": "h2",
                "distance_to_origin": 2,
                "distance_to_shorter": None,
                "shorter_proven": True,
                "optimal": True,
            },
            {
                "hash": "h3",
                "distance_to_origin": None,
                "distance_to_shorter": None,
                "shorter_proven": False,
                "optimal": False,
            },
        ],
    }


def test_summarize_annotations_counts_and_distributions() -> None:
    summary = summarize_annotations(_annotations())
    assert summary.rank == 2
    assert summary.moveset == "strict-ac"
    assert summary.move_catalog == "univ-v1"
    assert summary.total == 4
    assert summary.reached_origin == 3  # root, h1, h2 have a distance_to_origin
    assert summary.with_shorter == 1  # only h1 has a shorter descent
    assert summary.proven == 3
    assert summary.unresolved == 1  # h3 is not proven
    assert summary.distance_to_origin.counts == {0: 1, 1: 1, 2: 1}
    assert summary.distance_to_shorter.counts == {1: 1}


def test_render_annotation_markdown_has_header_and_sections() -> None:
    report = render_annotation_markdown(summarize_annotations(_annotations()), name="a.json")
    assert report.startswith("# Annotation summary: a.json")
    assert "- Move set: `strict-ac`" in report
    assert "- Proven settled: 3 | unresolved: 1" in report
    assert "## By distance to origin (moves to the trivial group)" in report
    assert "## By descent distance (moves to a strictly shorter group)" in report


def test_empty_annotations_summarize_without_error() -> None:
    summary = summarize_annotations({"rank": 2, "moveset": "universal", "annotations": []})
    assert summary.total == 0
    assert summary.move_catalog == "unknown"
    assert summary.distance_to_origin.counts == {}
    report = render_annotation_markdown(summary, name="empty.json")
    assert "_No data._" in report


def test_write_annotation_summary_targets_summary_dir(tmp_path) -> None:
    apath = tmp_path / "train_rank2.groups.strict-ac.annotations.json"
    apath.write_text(json.dumps(_annotations()), encoding="utf-8")
    summary_dir = tmp_path / "summaries"

    written = write_annotation_summary(apath, summary_dir)

    assert written == summary_dir / "train_rank2.groups.strict-ac.annotations.summary.md"
    assert written.read_text(encoding="utf-8").startswith(
        "# Annotation summary: train_rank2.groups.strict-ac.annotations.json"
    )
