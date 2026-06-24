from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    """Minimal reproducibility metadata written alongside smoke runs."""

    utc_timestamp: str
    experiment_id: str
    master_seed: int
    python_version: str
    platform: str
    uv_lock_checksum: str | None
    resolved_configuration: dict[str, Any]

    @classmethod
    def create(
        cls, experiment_id: str, master_seed: int, config: dict[str, Any]
    ) -> ReproducibilityManifest:
        """Create a manifest from the current process and `uv.lock` checksum."""
        lock = Path("uv.lock")
        checksum = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
        return cls(
            datetime.now(UTC).isoformat(),
            experiment_id,
            master_seed,
            platform.python_version(),
            platform.platform(),
            checksum,
            config,
        )

    def write(self, path: str | Path) -> None:
        """Write the manifest as canonical pretty-printed JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
