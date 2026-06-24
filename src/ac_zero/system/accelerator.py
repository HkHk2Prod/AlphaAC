from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AcceleratorReport:
    """Serializable summary of requested and selected accelerator backend."""

    requested: str
    selected: str
    source: str
    details: dict[str, object]

    def to_json(self) -> str:
        """Render the report as stable pretty-printed JSON."""
        return json.dumps(
            {
                "requested": self.requested,
                "selected": self.selected,
                "source": self.source,
                **self.details,
            },
            indent=2,
            sort_keys=True,
        )


def inspect_accelerator(requested: str = "auto") -> AcceleratorReport:
    """Inspect the local accelerator environment without mutating dependencies."""
    if requested not in {"auto", "cpu", "cuda12", "cuda13", "tpu"}:
        raise ValueError("invalid accelerator")
    if requested == "cpu":
        return AcceleratorReport(requested, "cpu", "explicit", {})
    if requested == "tpu" or os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE") == "TPU":
        selected = "tpu" if requested != "auto" else "cpu"
        return AcceleratorReport(requested, selected, "environment", {})
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
        if proc.returncode == 0 and requested in {"auto", "cuda12", "cuda13"}:
            selected = requested if requested != "auto" else "cuda12"
            return AcceleratorReport(requested, selected, "nvidia-smi", {})
    return AcceleratorReport(requested, "cpu", "fallback", {})
