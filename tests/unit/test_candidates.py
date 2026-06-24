import json

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.cli import main
from ac_zero.datasets.candidates import (
    akbulut_kirby,
    candidate_entries,
    miller_schupp,
    standard_candidates,
    write_candidates,
)


def test_akbulut_kirby_relators_match_known_form() -> None:
    presentation = akbulut_kirby(3)
    power, braid = presentation.relators
    assert power.letters == (1, 1, 1, -2, -2, -2, -2)  # x^3 y^-4
    assert braid.letters == (1, 2, 1, -2, -1, -2)  # xyx (yxy)^-1
    assert presentation.provenance["status"] == "potential_counterexample"


def test_miller_schupp_requires_zero_x_exponent_sum() -> None:
    ok = miller_schupp(2, [1, 2, -1, -2])  # commutator: x-exponent sum 0
    shift = ok.relators[0]
    assert shift.letters == (-1, 2, 2, 1, -2, -2, -2)  # x^-1 y^2 x y^-3
    with pytest.raises(ValueError, match="x-exponent sum"):
        miller_schupp(2, [1, 2])  # net +1 x
    with pytest.raises(ValueError, match="signed generators"):
        miller_schupp(2, [3])


def test_standard_candidates_are_distinct_and_round_trip() -> None:
    presentations = standard_candidates()
    assert len(presentations) >= 5
    hashes = {p.content_hash for p in presentations}
    assert len(hashes) == len(presentations)
    for presentation in presentations:
        restored = BalancedPresentation.from_json(presentation.to_json())
        assert restored.content_hash == presentation.content_hash
        assert "leakage_warning" in presentation.provenance


def test_candidate_labels_mark_ak2_trivial_and_others_unknown() -> None:
    by_id = {p.presentation_id: label for p, label in candidate_entries()}
    assert by_id["akbulut-kirby-2"].ac_trivial is True
    assert by_id["akbulut-kirby-3"].ac_trivial is None
    assert by_id["akbulut-kirby-3"].minimal_known_operations is None
    assert by_id["akbulut-kirby-3"].optimal is None


def test_write_candidates_emits_labeled_catalog(tmp_path) -> None:
    path = tmp_path / "candidates.json"
    count = write_candidates(path)
    data = json.loads(path.read_text())
    assert data["schema_version"] == "aczero-candidates-v1"
    assert len(data["instances"]) == count
    assert "leakage_warning" in data
    families = {instance["provenance"]["family"] for instance in data["instances"]}
    assert {"akbulut_kirby", "miller_schupp"} <= families
    for instance in data["instances"]:
        assert set(instance) >= {"ac_trivial", "minimal_known_operations", "optimal"}
    ak2 = next(i for i in data["instances"] if i["presentation_id"] == "akbulut-kirby-2")
    assert ak2["ac_trivial"] is True


def test_cli_dataset_candidates_writes_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["dataset", "candidates"]) == 0
    assert (tmp_path / "data/candidates/standard.json").exists()
