"""Run a scheduled job as a subprocess under the notebook's watchdog.

The unified Kaggle notebook shells out to the ``aczero`` CLI and must decide, when
the run ends, whether what happened counts as a failure worth reporting. That
decision is subtler than a return code:

* A job the watchdog stopped is *terminated*, so its return code is ``-SIGTERM``
  no matter how well it ran. Read literally, every budget-capped run "fails".
* A job that announces it is flushing (writing its checkpoint, plots and
  certificate, then pushing the bundle) must be waited for, not terminated --
  nothing in ac_zero handles SIGTERM, so terminating it there throws the work away.

:class:`JobRunner` owns both, so the notebook cell stays a single call.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

# What a job prints on stdout when its own wall-clock budget expires. Both training
# pipelines emit it ("...stopping at iteration boundary" / "...at epoch boundary") as
# they leave the training loop, with the whole flush -- checkpoint, plots, certificate,
# Hugging Face push -- still ahead of them. Seeing it switches the runner from
# "terminate on stop" to "wait for the flush", because those minutes are the entire
# durable output of the session.
FLUSH_MARKER = "wall-clock budget spent"

# How long a flushing job may take before the runner gives up and terminates it. The
# watchdog's soft deadline sits well inside Kaggle's ~12 h hard kill, so this spends
# head-room that would otherwise go unused.
FLUSH_GRACE_S = 900

# How long a terminated job gets to die before it is killed outright.
TERMINATE_GRACE_S = 120


class _Event(Protocol):
    def is_set(self) -> bool: ...


class _Process(Protocol):
    stdout: Any

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


@dataclass(frozen=True)
class JobOutcome:
    """How a job ended, in the vocabulary the scheduler's run records use.

    ``finished`` a completed run (including one the deadline ended -- that is the
                 planned end of a scheduled run, not a failure).
    ``stopped``  a clean interruption at the operator's request.
    ``failed``   the job itself exited non-zero, or the runner raised.
    """

    status: str
    error: str | None = None


def _default_popen(command: list[str]) -> Any:
    return subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )


class JobRunner:
    """Run ``command`` to completion, honoring a stop request from the watchdog."""

    def __init__(
        self,
        command: list[str],
        *,
        stop_event: _Event,
        deadline_hit: _Event,
        flush_grace_s: float = FLUSH_GRACE_S,
        popen: Callable[[list[str]], Any] = _default_popen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.command = command
        self._stop_event = stop_event
        self._deadline_hit = deadline_hit
        self._flush_grace_s = flush_grace_s
        self._popen = popen
        self._sleep = sleep
        self._monotonic = monotonic
        self._emit = emit
        self._flushing = False

    def run(self) -> JobOutcome:
        try:
            proc = self._popen(self.command)
        except Exception as exc:
            return JobOutcome("failed", repr(exc))
        try:
            terminated, stop_requested = self._pump(proc)
            self._drain(proc)
            return self._classify(proc.wait(), terminated=terminated, stop_requested=stop_requested)
        except Exception as exc:
            return JobOutcome("failed", repr(exc))

    def _pump(self, proc: _Process) -> tuple[bool, bool]:
        """Relay the job's output until it exits or the watchdog asks it to stop.

        Returns ``(terminated, stop_requested)`` -- whether we signalled the job, and
        whether a stop was asked for at all (a flushing job exits on its own, so the
        two come apart).
        """
        while proc.poll() is None:
            if self._stop_event.is_set():
                if self._flushing and self._await_flush(proc):
                    return False, True
                return self._terminate(proc), True
            if not self._relay_line(proc):
                self._sleep(1)
        return False, self._stop_event.is_set()

    def _relay_line(self, proc: _Process) -> bool:
        """Print one line of the job's output; return whether there was one."""
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            return False
        self._emit(line.rstrip("\n"))
        if FLUSH_MARKER in line:
            self._flushing = True
        return True

    def _await_flush(self, proc: _Process) -> bool:
        """Let an already-flushing job finish writing and pushing its output.

        Returns whether it exited within the grace window. Its output is relayed
        throughout -- both because those lines say whether the push landed, and because
        a full stdout pipe would block the very flush we are waiting on.
        """
        self._emit(
            "[main] stop requested, but the job is flushing its output — "
            f"waiting up to {self._flush_grace_s / 60:.0f} min for it to finish."
        )
        started = self._monotonic()
        while proc.poll() is None:
            if self._monotonic() - started >= self._flush_grace_s:
                self._emit("[main] flush did not finish within the grace window.")
                return False
            if not self._relay_line(proc):
                self._sleep(1)
        self._emit("[main] job flushed and exited on its own.")
        return True

    def _terminate(self, proc: _Process) -> bool:
        self._emit("[main] stop requested — terminating job so it can checkpoint & exit.")
        proc.terminate()
        try:
            proc.wait(timeout=TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True

    def _drain(self, proc: _Process) -> None:
        if proc.stdout:
            for line in proc.stdout:
                self._emit(line.rstrip("\n"))

    def _classify(self, rc: int, *, terminated: bool, stop_requested: bool) -> JobOutcome:
        if terminated:
            # We sent the SIGTERM, so the return code it produced is ours, not the job's:
            # it says nothing about whether the work succeeded. Classify by *why* we
            # stopped it instead.
            return JobOutcome("finished" if self._deadline_hit.is_set() else "stopped")
        if rc != 0:
            return JobOutcome("failed", f"job exited with code {rc}")
        if stop_requested and not self._deadline_hit.is_set():
            return JobOutcome("stopped")
        return JobOutcome("finished")
