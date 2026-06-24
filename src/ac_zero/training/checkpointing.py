from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


class CheckpointManager:
    """Small atomic JSON checkpoint helper for smoke workflows."""

    def __init__(self, directory: str | Path) -> None:
        """Create or reuse a checkpoint directory."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        """Atomically save a JSON payload under `<name>.json`."""
        path = self.directory / f"{name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
        return path

    def load_json(self, name: str) -> dict[str, Any]:
        """Load a JSON checkpoint payload by name."""
        return cast(dict[str, Any], json.loads((self.directory / f"{name}.json").read_text()))
