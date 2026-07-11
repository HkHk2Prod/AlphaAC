from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from ac_zero.training.logging.callbacks import CallbackManager
from ac_zero.training.logging.events import LogLevel
from ac_zero.training.logging.log_sinks import (
    JsonlEventLogger,
    TerminalProgressLogger,
    format_metrics,
)


class CliReporter:
    """Route CLI diagnostics and results through the shared callback system.

    Command results are written to stdout as the machine-readable contract,
    while diagnostics, warnings, and errors are emitted as structured, leveled
    events to a rotating JSONL log plus a human-readable log file. Warnings and
    errors are mirrored to stderr so they never corrupt stdout result payloads.
    """

    def __init__(
        self,
        command: str,
        run_directory: str | Path = "runs/cli",
        *,
        stream: TextIO | None = None,
        error_stream: TextIO | None = None,
    ) -> None:
        """Open the CLI log sinks for one command invocation."""
        log_dir = Path(run_directory) / "logs"
        self._manager = CallbackManager(
            (
                JsonlEventLogger(log_dir / "events.jsonl"),
                TerminalProgressLogger(
                    log_dir / "cli.log",
                    stream=error_stream or sys.stderr,
                    min_level=LogLevel.WARNING,
                ),
            )
        )
        self._command = command
        self._stream = stream or sys.stdout
        self._progress_stream = error_stream or sys.stderr
        self._step = 0

    def info(self, phase: str, message: str, metrics: dict[str, Any] | None = None) -> None:
        """Record an informational event (captured to logs, not stdout)."""
        self._emit(LogLevel.INFO, phase, message, metrics)

    def progress(self, phase: str, message: str, metrics: dict[str, Any] | None = None) -> None:
        """Record a progress update: logged at INFO and echoed to the progress stream.

        Long-running commands call this to report incremental status. The event is
        captured to the structured logs like any info event, and a concise line is
        mirrored to the progress stream (stderr by default) so the run stays visible
        without polluting the machine-readable stdout result.
        """
        self._emit(LogLevel.INFO, phase, message, metrics)
        suffix = f" | {format_metrics(metrics)}" if metrics else ""
        print(f"{phase}: {message}{suffix}", file=self._progress_stream, flush=True)

    def warning(self, phase: str, message: str, metrics: dict[str, Any] | None = None) -> None:
        """Record a warning event, mirrored to stderr."""
        self._emit(LogLevel.WARNING, phase, message, metrics)

    def error(
        self,
        phase: str,
        message: str,
        exc: BaseException | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Record an error event, mirrored to stderr with exception details."""
        self._step += 1
        self._manager.emit_error(phase, message, exc, metrics=metrics)

    def result_json(
        self, payload: Any, *, indent: int | None = None, sort_keys: bool = False
    ) -> None:
        """Print a JSON result to stdout and log that the command produced one."""
        print(json.dumps(payload, indent=indent, sort_keys=sort_keys), file=self._stream)
        self.info(self._command, "emitted command result", {"format": "json"})

    def result_text(self, text: object) -> None:
        """Print a plain-text result to stdout and log that it was produced."""
        print(text, file=self._stream)
        self.info(self._command, "emitted command result", {"format": "text"})

    def close(self) -> None:
        """Flush and close the CLI log sinks."""
        self._manager.close()

    def _emit(
        self, level: LogLevel, phase: str, message: str, metrics: dict[str, Any] | None
    ) -> None:
        self._step += 1
        self._manager.emit(self._step, phase, message, metrics or {}, level=level)
