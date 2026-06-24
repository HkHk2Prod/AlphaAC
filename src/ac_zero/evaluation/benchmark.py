from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_benchmark_report(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write machine-readable benchmark rows as stable pretty-printed JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
