"""Tests for the single-file container of memory-mapped numpy columns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ac_zero.datasets.columnar import MAGIC, ColumnFile, write


def _columns() -> dict[str, np.ndarray]:
    return {
        "letters": np.array([1, -2, 3], dtype=np.int8),
        "offsets": np.array([0, 1, 3], dtype=np.int32),
        "digests": np.arange(64, dtype=np.uint8).reshape(2, 32),
        "prefixes": np.array([7, 9], dtype=">u8"),
    }


def test_columns_round_trip_with_dtypes_and_shapes(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    write(path, {"schema": "v1", "n": 3}, _columns())
    mapped = ColumnFile.open(path)

    assert mapped is not None
    assert mapped.path == path
    assert mapped.header["schema"] == "v1" and mapped.header["n"] == 3
    for name, expected in _columns().items():
        assert mapped.columns[name].dtype == expected.dtype
        assert mapped.columns[name].shape == expected.shape
        assert np.array_equal(mapped.columns[name], expected)


def test_columns_are_views_on_the_mapping_not_copies(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    write(path, {}, _columns())
    mapped = ColumnFile.open(path)
    assert mapped is not None
    # A view aliases the mmap: it has a base object and is not writable.
    assert mapped.columns["letters"].base is not None
    assert not mapped.columns["letters"].flags.writeable


def test_an_empty_column_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    write(path, {}, {"empty": np.empty((0, 32), dtype=np.uint8)})
    mapped = ColumnFile.open(path)
    assert mapped is not None
    assert mapped.columns["empty"].shape == (0, 32)


def test_write_replaces_an_existing_container_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    write(path, {}, {"a": np.array([1], dtype=np.int8)})
    write(path, {}, {"a": np.array([2, 3], dtype=np.int8)})
    mapped = ColumnFile.open(path)

    assert mapped is not None
    assert np.array_equal(mapped.columns["a"], [2, 3])
    assert not list(tmp_path.glob("*.tmp"))


def test_open_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert ColumnFile.open(tmp_path / "absent.bin") is None


def test_open_returns_none_for_a_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    path.write_bytes(b"NOPE" + b"\0" * 64)
    assert ColumnFile.open(path) is None


def test_open_returns_none_for_a_truncated_header(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    write(path, {"schema": "v1"}, _columns())
    path.write_bytes(path.read_bytes()[: len(MAGIC) + 8])
    assert ColumnFile.open(path) is None


def test_open_returns_none_when_the_header_lacks_a_column_map(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    blob = b'{"schema": "v1"}'
    path.write_bytes(MAGIC + len(blob).to_bytes(4, "little") + blob + b"\0" * 64)
    assert ColumnFile.open(path) is None


def test_a_failed_write_leaves_the_previous_container_intact(tmp_path: Path) -> None:
    path = tmp_path / "c.bin"
    write(path, {}, {"a": np.array([1], dtype=np.int8)})

    class Exploding(np.ndarray):
        def tofile(self, *args: object, **kwargs: object) -> None:
            raise OSError("disk full")

    broken = np.array([9], dtype=np.int8).view(Exploding)
    with pytest.raises(OSError, match="disk full"):
        write(path, {}, {"a": broken})

    mapped = ColumnFile.open(path)
    assert mapped is not None
    assert np.array_equal(mapped.columns["a"], [1])
    assert not list(tmp_path.glob("*.tmp"))
