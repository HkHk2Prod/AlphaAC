import json

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.cli import main
from ac_zero.datasets.candidates import (
    akbulut_kirby,
    candidate_entries,
    miller_schupp,
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


def test_candidate_presentations_are_distinct_and_round_trip() -> None:
    presentations = [presentation for presentation, _ in candidate_entries()]
    assert len(presentations) >= 5
    hashes = {p.content_hash for p in presentations}
    assert len(hashes) == len(presentations)
    for presentation in presentations:
        restored = BalancedPresentation.from_json(presentation.to_json())
        assert restored.content_hash == presentation.content_hash
        assert "leakage_warning" in presentation.provenance


def test_candidate_labels_mark_ak2_trivial_and_others_unknown() -> None:
    by_id = {p.presentation_id: ac_trivial for p, ac_trivial in candidate_entries()}
    assert by_id["akbulut-kirby-2"] is True
    assert by_id["akbulut-kirby-3"] is None


def test_write_candidates_emits_group_catalog(tmp_path) -> None:
    from ac_zero.datasets.groups import SCHEMA_VERSION

    path = tmp_path / "candidates.groups.json"
    count = write_candidates(path)
    data = json.loads(path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert len(data["groups"]) == count
    assert "leakage_warning" in data
    sources = {group["source"] for group in data["groups"]}
    assert {"akbulut_kirby", "miller_schupp"} <= sources
    # Candidates are unexpanded, so they carry no transitions.
    assert all("transitions" not in group for group in data["groups"])
    ak2_hash = akbulut_kirby(2).content_hash
    ak2 = next(g for g in data["groups"] if g["hash"] == ak2_hash)
    assert ak2["ac_trivial"] is True


def test_cli_dataset_candidates_writes_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["dataset", "candidates"]) == 0
    assert (tmp_path / "data/candidates/standard.json").exists()
