"""A single-file container of named numpy arrays, written atomically and mmapped.

Datasets that are too large to hold as Python objects are stored as flat columns
instead. The container is one file so it can be swapped into place with a single
``os.replace``: readers either see the whole old file or the whole new one, and a
reader whose mapping is already open keeps reading the old inode safely while a
writer replaces it.

Layout: a ``ACZI`` magic, a little-endian ``uint32`` header length, then a JSON
header, padded to the alignment. Each column's recorded ``offset`` is relative to
the end of that padding, so the header's own size never feeds back into the
offsets it records.
"""

from __future__ import annotations

import json
import mmap
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MAGIC = b"ACZI"
_ALIGNMENT = 64
_LENGTH_BYTES = 4

Columns = dict[str, NDArray[Any]]


def _pad(position: int) -> int:
    """Round a byte position up to the next alignment boundary."""
    return position + -position % _ALIGNMENT


def write(path: Path, header: dict[str, Any], columns: Columns) -> None:
    """Write ``header`` and ``columns`` to ``path``, atomically replacing any file there.

    The container is assembled in a sibling temp file and moved into place, so a
    crash mid-write leaves nothing behind, and two processes racing to write the
    same derived data simply overwrite each other.
    """
    layout: dict[str, Any] = {}
    offset = 0
    for name, column in columns.items():
        offset = _pad(offset)
        layout[name] = {"dtype": column.dtype.str, "shape": list(column.shape), "offset": offset}
        offset += int(column.nbytes)
    blob = json.dumps({**header, "columns": layout}).encode("utf-8")

    handle = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(MAGIC + len(blob).to_bytes(_LENGTH_BYTES, "little") + blob)
            start = _pad(handle.tell())
            for name, column in columns.items():
                handle.write(b"\0" * (start + layout[name]["offset"] - handle.tell()))
                column.tofile(handle)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


class ColumnFile:
    """A memory-mapped container: its JSON header, and its columns as numpy views.

    The views alias the mapping directly, so nothing is copied onto the heap and
    every process that opens the same file shares one copy through the page cache.
    """

    def __init__(self, path: Path, header: dict[str, Any], start: int) -> None:
        """Map ``path`` and expose the columns its header describes."""
        self.path = path
        self.header = header
        self._handle = path.open("rb")
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.columns: Columns = {
            name: np.frombuffer(
                self._map,
                dtype=np.dtype(entry["dtype"]),
                count=int(np.prod(entry["shape"])),
                offset=start + entry["offset"],
            ).reshape(entry["shape"])
            for name, entry in header["columns"].items()
        }

    @classmethod
    def open(cls, path: Path) -> ColumnFile | None:
        """Map the container at ``path``, or return None when it is absent or corrupt."""
        try:
            with path.open("rb") as handle:
                if handle.read(len(MAGIC)) != MAGIC:
                    return None
                size = int.from_bytes(handle.read(_LENGTH_BYTES), "little")
                header: dict[str, Any] = json.loads(handle.read(size))
            return cls(path, header, _pad(len(MAGIC) + _LENGTH_BYTES + size))
        except (OSError, ValueError, KeyError):
            return None
