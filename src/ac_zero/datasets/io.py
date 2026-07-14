from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import IO, Any


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write a JSON object to `path` atomically, streaming it out member by member.

    A member whose value is an *iterator* is encoded one element at a time and
    written straight to the file, so the document never exists as a Python string.
    That is what keeps a checkpoint from being the peak of a long run:
    :func:`json.dumps` on a gigabyte-scale dataset holds the entire text in memory,
    and the caller must first materialize every entry to hand it over -- several
    times the file size in transient bytes, at the one moment the dataset itself is
    already resident.

    Streamed elements are written one per line rather than pretty-printed, which
    also stops a dataset of short relators from spending most of its bytes on
    indentation. Members are written in sorted key order, so a small member named to
    sort before the large array stays cheap for
    :func:`ac_zero.datasets.json_stream.read_members_before` to recover.

    The document goes to a sibling temp file and is moved into place with
    :func:`os.replace`, so an interruption mid-write can never leave a partially
    written dataset on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False, encoding="utf-8"
    )
    try:
        with handle:
            _write_object(handle, data)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _write_object(handle: IO[str], data: Mapping[str, Any]) -> None:
    """Write one JSON object, streaming any member handed over as an iterator."""
    handle.write("{")
    separator = "\n  "
    for key in sorted(data):
        handle.write(f"{separator}{json.dumps(str(key))}: ")
        value = data[key]
        if isinstance(value, Iterator):
            _write_array(handle, value)
        else:  # small enough to encode whole; re-indent to sit inside the object
            handle.write(json.dumps(value, indent=2, sort_keys=True).replace("\n", "\n  "))
        separator = ",\n  "
    handle.write("\n}\n")


def _write_array(handle: IO[str], items: Iterator[Any]) -> None:
    """Write an array one element per line, encoding a single element at a time."""
    handle.write("[")
    separator = "\n    "
    for item in items:
        handle.write(f"{separator}{json.dumps(item, sort_keys=True)}")
        separator = ",\n    "
    handle.write("\n  ]")
