from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.labels import known_solution
from ac_zero.datasets.validation import validate_dataset, validate_mapping


def _entry(seed_letters=(1,)):
    presentation = BalancedPresentation.from_letters(2, [list(seed_letters), [2]])
    return {**presentation.to_json(), **known_solution(3).to_json()}


def _document(instances):
    return {"schema_version": "aczero-dataset-v2", "rank": 2, "instances": instances}


def test_valid_document_passes() -> None:
    report = validate_mapping(_document([_entry()]))
    assert report.ok
    assert report.instances == 1
    assert report.errors == []


def test_committed_candidates_file_validates() -> None:
    assert validate_dataset("data/candidates/standard.json").ok


def test_unknown_schema_version_fails() -> None:
    document = _document([_entry()])
    document["schema_version"] = "aczero-dataset-v9"
    assert not validate_mapping(document).ok


def test_content_hash_mismatch_is_detected() -> None:
    entry = _entry()
    entry["content_hash"] = "deadbeef"
    report = validate_mapping(_document([entry]))
    assert not report.ok
    assert any("content_hash" in error for error in report.errors)


def test_optimal_without_operations_is_rejected() -> None:
    entry = _entry()
    entry["optimal"] = True
    entry["minimal_known_operations"] = None
    report = validate_mapping(_document([entry]))
    assert not report.ok
    assert any("optimal cannot be true" in error for error in report.errors)


def test_instances_must_be_a_list() -> None:
    assert not validate_mapping({"schema_version": "aczero-dataset-v2", "rank": 2}).ok
