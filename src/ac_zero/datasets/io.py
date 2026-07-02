from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write pretty-printed JSON to `path` atomically via a temp file + os.replace.

    The document is written to a sibling temp file and moved into place with
    :func:`os.replace`, so a crash or interruption mid-write can never leave a
    partially written or corrupted dataset on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False, encoding="utf-8"
    )
    try:
        with handle:
            handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
