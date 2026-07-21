from __future__ import annotations

from pathlib import Path

import pytest

from ac_zero.benchmarks.catalog import (
    BenchmarkCatalog,
    benchmark_entries,
    catalog_name,
    miller_schupp_words,
)

_X, _Y = 1, 2


def _x_exponent_sum(word: list[int]) -> int:
    return sum(1 for letter in word if letter == _X) - sum(1 for letter in word if letter == -_X)


def test_words_are_freely_reduced_with_zero_x_exponent_sum() -> None:
    for word in miller_schupp_words(6):
        assert _x_exponent_sum(word) == 0
        assert all(word[i] != -word[i + 1] for i in range(len(word) - 1))
        assert all(abs(letter) in (_X, _Y) for letter in word)


def test_words_include_the_empty_word_and_grow_with_length() -> None:
    assert list(miller_schupp_words(0)) == [[]]
    counts = [sum(1 for _ in miller_schupp_words(k)) for k in range(5)]
    assert counts == sorted(counts)
    assert counts[4] > counts[0]


def test_words_of_odd_x_parity_are_never_emitted() -> None:
    # A single x cannot be balanced within one letter, so length 1 admits only y^+-1.
    assert sorted(miller_schupp_words(1)) == [[], [-_Y], [_Y]]


def test_negative_max_length_is_rejected() -> None:
    with pytest.raises(ValueError):
        list(miller_schupp_words(-1))


def test_entries_respect_the_relator_bound() -> None:
    bound = 9
    for entry in benchmark_entries(max_relator_length=bound, max_w_length=5):
        assert max(len(relator) for relator in entry.relators) <= bound


def test_entries_cover_both_families_and_are_unique() -> None:
    entries = benchmark_entries(max_relator_length=12, max_w_length=4)
    families = {str(entry.provenance.get("family")) for entry in entries}
    assert families == {"akbulut_kirby", "miller_schupp"}
    hashes = [entry.content_hash for entry in entries]
    assert len(hashes) == len(set(hashes))


def test_entries_are_ordered_smallest_first() -> None:
    entries = benchmark_entries(max_relator_length=10, max_w_length=4)
    lengths = [entry.total_length for entry in entries]
    assert lengths == sorted(lengths)


def test_a_tighter_w_bound_yields_a_subset() -> None:
    wide = {e.content_hash for e in benchmark_entries(max_relator_length=12, max_w_length=5)}
    narrow = {e.content_hash for e in benchmark_entries(max_relator_length=12, max_w_length=3)}
    assert narrow < wide


def test_bounds_must_be_sane() -> None:
    with pytest.raises(ValueError):
        benchmark_entries(max_relator_length=0)
    with pytest.raises(ValueError):
        benchmark_entries(max_relator_length=10, max_w_length=-1)


def test_a_bound_too_small_for_any_family_yields_nothing() -> None:
    # AK's braid relator is 6 letters and MS's shift relator is 5 at n=1.
    assert benchmark_entries(max_relator_length=2, max_w_length=4) == []


def test_catalog_round_trips_through_json(tmp_path: Path) -> None:
    catalog = BenchmarkCatalog.build(max_relator_length=10, max_w_length=4)
    path = catalog.write(tmp_path / "sub" / "catalog.json")
    restored = BenchmarkCatalog.read(path)
    assert restored.name == catalog.name
    assert restored.max_relator_length == catalog.max_relator_length
    assert restored.max_w_length == catalog.max_w_length
    assert [e.content_hash for e in restored.entries] == [e.content_hash for e in catalog.entries]


def test_catalog_name_encodes_both_bounds() -> None:
    assert catalog_name(48, 7) == "ak-ms-rel48-w7"


def test_catalog_json_reports_its_family_counts() -> None:
    catalog = BenchmarkCatalog.build(max_relator_length=10, max_w_length=4)
    payload = catalog.to_json()
    assert payload["count"] == len(catalog.entries)
    assert sum(payload["families"].values()) == len(catalog.entries)


def test_every_entry_carries_the_leakage_warning() -> None:
    for entry in benchmark_entries(max_relator_length=10, max_w_length=3):
        assert "not training data" in str(entry.provenance.get("leakage_warning", ""))


def test_create_writes_locally_and_publishes_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ac_zero.benchmarks import results as results_module
    from ac_zero.benchmarks.commands import create_catalog

    uploaded: list[str] = []
    monkeypatch.setattr(
        results_module,
        "upload_files",
        lambda pairs, *, bucket: uploaded.extend(remote for _, remote in pairs),
    )
    payload = create_catalog(
        max_relator_length=8,
        max_w_length=2,
        output=str(tmp_path / "c.json"),
        bucket="b/c",
        log=lambda _: None,
    )
    assert payload["uploaded"] is True
    assert payload["remote"] == f"benchmark_datasets/{payload['name']}.json"
    assert uploaded == [payload["remote"]]
    assert (tmp_path / "c.json").is_file()


def test_create_can_be_kept_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ac_zero.benchmarks import results as results_module
    from ac_zero.benchmarks.commands import create_catalog

    def explode(pairs: object, *, bucket: str) -> None:
        raise AssertionError("--no-upload must not touch the bucket")

    monkeypatch.setattr(results_module, "upload_files", explode)
    payload = create_catalog(
        max_relator_length=8,
        max_w_length=2,
        output=str(tmp_path / "c.json"),
        upload=False,
        log=lambda _: None,
    )
    assert payload["uploaded"] is False
    assert "remote" not in payload
    assert (tmp_path / "c.json").is_file()


def test_a_failed_upload_keeps_the_local_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ac_zero.benchmarks import results as results_module
    from ac_zero.benchmarks.commands import create_catalog

    def boom(pairs: object, *, bucket: str) -> None:
        raise RuntimeError("no token")

    monkeypatch.setattr(results_module, "upload_files", boom)
    payload = create_catalog(
        max_relator_length=8,
        max_w_length=2,
        output=str(tmp_path / "c.json"),
        bucket="b/c",
        log=lambda _: None,
    )
    assert payload["uploaded"] is False
    assert "no token" in payload["upload_error"]
    assert (tmp_path / "c.json").is_file()
