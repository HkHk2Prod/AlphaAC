import json

from ac_zero.datasets.summary import (
    render_markdown,
    summarize,
    summary_path_for,
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
