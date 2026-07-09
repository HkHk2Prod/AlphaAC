"""Thin wrapper over the Kaggle CLI used by the scheduler.

Only two operations are needed: push a kernel (launch a run) and query a
kernel's status. The subprocess runner is injectable so tests drive the
scheduler without the real ``kaggle`` binary or credentials.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

# A runner takes an argv list and returns a completed process with text output.
Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]

# Kaggle kernel statuses that mean the run is no longer occupying a slot.
TERMINAL_STATUSES = frozenset({"complete", "error", "cancelacknowledged"})

_STATUS_RE = re.compile(r'status\s+"?([A-Za-z]+)"?', re.IGNORECASE)


class KaggleError(RuntimeError):
    """Raised on a failed Kaggle CLI invocation."""


@dataclass(slots=True)
class PushResult:
    ok: bool
    output: str


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=300)


class KaggleClient:
    """Launch and inspect Kaggle notebook runs via the CLI."""

    def __init__(self, runner: Runner | None = None) -> None:
        self._run: Runner = runner or _default_runner

    def push(self, notebook_dir: str) -> PushResult:
        """``kaggle kernels push -p <dir>`` -- launch (a new version of) a run.

        Raises :class:`KaggleError` on a non-zero exit so the caller does not
        record a launch that never happened.
        """
        proc = self._run(["kaggle", "kernels", "push", "-p", notebook_dir])
        output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise KaggleError(f"`kaggle kernels push` failed ({proc.returncode}): {output.strip()}")
        return PushResult(ok=True, output=output.strip())

    def status(self, slug: str) -> str | None:
        """Return the normalised (lowercase) status of ``slug``'s latest version.

        Returns ``None`` when the status cannot be determined (e.g. the kernel
        does not exist yet) rather than raising, so reconciliation degrades to
        "trust the heartbeat" instead of aborting the whole tick.
        """
        proc = self._run(["kaggle", "kernels", "status", slug])
        if proc.returncode != 0:
            return None
        match = _STATUS_RE.search(proc.stdout or "")
        return match.group(1).lower() if match else None

    @staticmethod
    def is_terminal(status: str | None) -> bool:
        return status is not None and status.lower() in TERMINAL_STATUSES
