from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ac_zero.training.events import LogLevel, TrainingCallback, TrainingEvent
from ac_zero.training.log_sinks import (
    AsciiGraphLogger,
    JsonlEventLogger,
    RotatingFileWriter,
    TerminalProgressLogger,
)

__all__ = [
    "AsciiGraphLogger",
    "CallbackManager",
    "JsonlEventLogger",
    "LogLevel",
    "RotatingFileWriter",
    "TerminalProgressLogger",
    "TrainingCallback",
    "TrainingEvent",
    "default_smoke_callbacks",
    "default_training_callbacks",
]


class CallbackManager:
    """Fan out leveled training events to multiple callbacks."""

    def __init__(self, callbacks: Iterable[TrainingCallback]) -> None:
        """Create a manager from callbacks in deterministic invocation order."""
        self._callbacks = tuple(callbacks)
        self._last_step = 0

    def emit(
        self,
        step: int,
        phase: str,
        message: str,
        metrics: dict[str, float | int | bool | str] | None = None,
        *,
        level: LogLevel = LogLevel.INFO,
    ) -> TrainingEvent:
        """Create and dispatch one `TrainingEvent` at the given level."""

        event = TrainingEvent(step, phase, message, metrics or {}, level)
        self._last_step = step
        for callback in self._callbacks:
            callback.on_event(event)
        return event

    def emit_error(
        self,
        phase: str,
        message: str,
        exc: BaseException | None = None,
        *,
        metrics: dict[str, float | int | bool | str] | None = None,
    ) -> TrainingEvent:
        """Emit an ERROR-level event, annotating it with exception details."""

        data: dict[str, float | int | bool | str] = dict(metrics or {})
        if exc is not None:
            data.setdefault("error_type", type(exc).__name__)
            data.setdefault("error", str(exc))
        return self.emit(self._last_step + 1, phase, message, data, level=LogLevel.ERROR)

    def close(self) -> None:
        """Close callbacks in registration order."""

        for callback in self._callbacks:
            callback.close()


def default_smoke_callbacks(run_directory: str | Path) -> CallbackManager:
    """Create the default file, terminal, and graph callbacks for smoke runs."""

    run = Path(run_directory)
    log_dir = run / "logs"
    graph_dir = run / "artifacts"
    return CallbackManager(
        (
            JsonlEventLogger(log_dir / "training_events.jsonl"),
            TerminalProgressLogger(log_dir / "progress.log"),
            AsciiGraphLogger(
                graph_dir / "live_graphs.txt",
                graph_dir / "final_graphs.txt",
                (
                    "loss",
                    "value",
                    "target",
                    "replay_size",
                    "root_expanded_nodes",
                    "certificate_verified",
                ),
            ),
        )
    )


def default_training_callbacks(
    run_directory: str | Path, *, extra: Iterable[TrainingCallback] = ()
) -> CallbackManager:
    """Create default callbacks for config-driven training pipeline runs.

    ``extra`` callbacks (e.g. a checkpoint uploader) are appended after the
    default file loggers so callers can add behaviour without rebuilding them.
    """

    run = Path(run_directory)
    log_dir = run / "logs"
    graph_dir = run / "artifacts"
    return CallbackManager(
        (
            JsonlEventLogger(log_dir / "training_events.jsonl"),
            TerminalProgressLogger(log_dir / "progress.log"),
            AsciiGraphLogger(
                graph_dir / "live_graphs.txt",
                graph_dir / "final_graphs.txt",
                (
                    "total_loss",
                    "policy_loss",
                    "value_loss",
                    "replay_size",
                    "episodes",
                    "mean_return",
                    "success_rate",
                ),
            ),
            *extra,
        )
    )
