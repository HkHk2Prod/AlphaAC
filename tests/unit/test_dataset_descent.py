import json
from pathlib import Path

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.descent import DescentAnnotateConfig, annotate_descent
from ac_zero.datasets.validation import validate_mapping

# A small budget: it settles the two easy fixtures in a handful of expansions while
# leaving the near-minimal standard presentation an (unproven) unknown quickly.
_CONFIG = DescentAnnotateConfig(max_expansions=50, workers=1, checkpoint_every=0)


def _dataset(*relator_sets) -> dict:
    instances = [BalancedPresentation.from_letters(2, rels).to_json() for rels in relator_sets]
    return {
        "schema_version": "aczero-dataset-v3",
        "rank": 2,
        "instances": instances,
        "provenance": {"generator": "test", "count": len(instances)},
    }


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_annotates_every_entry_with_descent_fields(tmp_path) -> None:
    path = tmp_path / "d.json"
    _write(path, _dataset([[1, -2], [2]], [[1, 2], [2]], [[1], [2]]))

    report = annotate_descent(path, _CONFIG)

    entries = json.loads(path.read_text())["instances"]
    by_hash = {e["content_hash"]: e for e in entries}
    one = by_hash[BalancedPresentation.from_letters(2, [[1, -2], [2]]).content_hash]
    two = by_hash[BalancedPresentation.from_letters(2, [[1, 2], [2]]).content_hash]
    minimal = by_hash[BalancedPresentation.from_letters(2, [[1], [2]]).content_hash]

    assert (one["descent_distance"], one["descent_proven"]) == (1, True)
    assert (two["descent_distance"], two["descent_proven"]) == (2, True)
    # The standard presentation has no cheap descent, so it is left unknown.
    assert minimal["descent_distance"] is None
    assert minimal["descent_proven"] is False
    assert report.total == 3
    assert report.computed == 3
    assert report.with_descent == 2
    assert report.max_distance == 2


def test_output_validates_and_preserves_order(tmp_path) -> None:
    path = tmp_path / "d.json"
    data = _dataset([[1, -2], [2]], [[1, 2], [2]], [[1], [2]])
    order = [e["content_hash"] for e in data["instances"]]
    _write(path, data)

    annotate_descent(path, _CONFIG)

    written = json.loads(path.read_text())
    assert [e["content_hash"] for e in written["instances"]] == order
    assert validate_mapping(written).ok


def test_repeated_pass_skips_proven_entries(tmp_path) -> None:
    path = tmp_path / "d.json"
    _write(path, _dataset([[1, -2], [2]], [[1, 2], [2]], [[1], [2]]))

    annotate_descent(path, _CONFIG)
    second = annotate_descent(path, _CONFIG)

    # Only the still-unknown standard presentation is searched again.
    assert second.computed == 1
    assert second.proven == 2


def test_separate_output_leaves_input_untouched(tmp_path) -> None:
    src = tmp_path / "in.json"
    dst = tmp_path / "out.json"
    _write(src, _dataset([[1, -2], [2]]))

    annotate_descent(src, _CONFIG, output=dst)

    assert "descent_distance" not in json.loads(src.read_text())["instances"][0]
    assert json.loads(dst.read_text())["instances"][0]["descent_distance"] == 1
