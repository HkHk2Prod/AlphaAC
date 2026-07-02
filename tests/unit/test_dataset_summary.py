import json

from ac_zero.datasets.summary import (
    render_markdown,
    summarize,
    summary_path_for,
    write_dataset_summary,
)


def _dataset() -> dict:
    # Root, then two depth-1 groups: one exhausted with a single construction move,
    # one open with two co-optimal moves. Sizes are 2, 3, 3.
    return {
        "rank": 2,
        "provenance": {"generator": "trivial_graph_expansion"},
        "instances": [
            {
                "difficulty": 0,
                "relators": [[1], [2]],
                "predecessors": [],
                "minimal_known_operations": 0,
                "optimal": True,
                "exhausted": True,
            },
            {
                "difficulty": 1,
                "relators": [[1, 2], [2]],
                "predecessors": [{"parent_hash": "root", "move": {}}],
                "minimal_known_operations": 2,
                "optimal": False,
                "exhausted": True,
            },
            {
                "difficulty": 1,
                "relators": [[1], [2, 1]],
                "predecessors": [
                    {"parent_hash": "root", "move": {}},
                    {"parent_hash": "other", "move": {}},
                ],
                "minimal_known_operations": 3,
                "optimal": False,
                "exhausted": False,
            },
        ],
    }


def test_summarize_buckets_every_dimension() -> None:
    summary = summarize(_dataset())
    assert summary.rank == 2
    assert summary.generator == "trivial_graph_expansion"
    assert summary.total_groups == 3
    assert summary.roots == 1
    assert summary.exhausted == 2
    assert summary.frontier == 1
    assert summary.optimal == 1
    assert summary.difficulty.counts == {0: 1, 1: 2}
    assert summary.total_length.counts == {2: 1, 3: 2}
    assert summary.predecessors.counts == {0: 1, 1: 1, 2: 1}
    assert summary.known_operations.counts == {0: 1, 2: 1, 3: 1}


def test_distribution_reports_min_max_mean() -> None:
    length = summarize(_dataset()).total_length
    assert length.minimum == 2
    assert length.maximum == 3
    assert length.population == 3
    assert length.mean == (2 + 3 + 3) / 3


def test_empty_dataset_summarizes_without_error() -> None:
    summary = summarize({"rank": 2, "instances": [], "provenance": {}})
    assert summary.total_groups == 0
    assert summary.difficulty.counts == {}
    assert summary.difficulty.mean is None
    assert "_No data._" in render_markdown(summary, name="empty.json")


def test_render_markdown_has_sections_and_table_rows() -> None:
    report = render_markdown(summarize(_dataset()), name="g.json")
    assert report.startswith("# Dataset summary: g.json")
    for heading in (
        "## By construction difficulty",
        "## By size (total relator length)",
        "## By co-optimal construction moves",
        "## By known trivialization length",
    ):
        assert heading in report
    # Two groups sit at difficulty 1; the histogram row reflects that.
    assert "| 1 | 2 |" in report


def test_write_dataset_summary_targets_summary_dir(tmp_path) -> None:
    dataset_path = tmp_path / "generated" / "g.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(json.dumps(_dataset()), encoding="utf-8")
    summary_dir = tmp_path / "summaries"

    written = write_dataset_summary(dataset_path, summary_dir)

    assert written == summary_path_for(dataset_path, summary_dir)
    assert written == summary_dir / "g.summary.md"
    assert written.read_text(encoding="utf-8").startswith("# Dataset summary: g.json")
