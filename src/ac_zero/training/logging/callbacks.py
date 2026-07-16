from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ac_zero.training.logging.console_summary import ConsoleSummaryLogger
from ac_zero.training.logging.events import LogLevel, TrainingCallback, TrainingEvent, Verbosity
from ac_zero.training.logging.log_sinks import (
    AsciiGraphLogger,
    JsonlEventLogger,
    RotatingFileWriter,
    TerminalProgressLogger,
)

__all__ = [
    "AsciiGraphLogger",
    "CallbackManager",
    "ConsoleSummaryLogger",
    "JsonlEventLogger",
    "LogLevel",
    "RotatingFileWriter",
    "TerminalProgressLogger",
    "TrainingCallback",
    "TrainingEvent",
    "Verbosity",
    "default_smoke_callbacks",
    "default_training_callbacks",
]

# Metrics the training graph sink tracks; small enough to render as ASCII bars.
# The list spans both kinds of run -- the RL backends emit the self-play series and
# the supervised run the ``val_*`` ones. A metric a run never emits records no values
# and is simply left out of its graphs, so neither kind shows the other's rows.
_TRAINING_GRAPH_METRICS = (
    "total_loss",
    "policy_loss",
    "value_loss",
    "replay_size",
    "episodes",
    "mean_return",
    "success_rate",
    # Supervised: the per-epoch validation scores, led by the descent accuracy the
    # run selects its best checkpoint on.
    "val_descent_accuracy",
    "val_mean_delta",
    "val_policy_loss",
)


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
    run_directory: str | Path,
    *,
    verbosity: Verbosity | str = Verbosity.SUMMARY,
    extra: Iterable[TrainingCallback] = (),
) -> CallbackManager:
    """Create default callbacks for config-driven training pipeline runs.

    ``verbosity`` picks what reaches the terminal (see
    :class:`ac_zero.training.logging.events.Verbosity`); every level still writes the full
    JSONL event log and the live/final graph files. At ``verbose`` the console
    gets the historical per-event lines and live graphs; at ``summary`` (default)
    a compact per-iteration summary plus the final graph; at ``quiet`` only the
    start/stop milestones and diagnostics. ``extra`` callbacks (e.g. a checkpoint
    uploader) are appended after the default loggers.
    """

    verbosity = Verbosity.parse(verbosity)
    verbose = verbosity >= Verbosity.VERBOSE
    run = Path(run_directory)
    log_dir = run / "logs"
    graph_dir = run / "artifacts"
    callbacks: list[TrainingCallback] = [
        JsonlEventLogger(log_dir / "training_events.jsonl"),
        TerminalProgressLogger(log_dir / "progress.log", console=verbose),
        AsciiGraphLogger(
            graph_dir / "live_graphs.txt",
            graph_dir / "final_graphs.txt",
            _TRAINING_GRAPH_METRICS,
            every_n_events=1 if verbose else 50,
            console_live=verbose,
            console_final=verbosity >= Verbosity.SUMMARY,
        ),
    ]
    if not verbose:
        callbacks.append(ConsoleSummaryLogger(iterations=verbosity >= Verbosity.SUMMARY))
    callbacks.extend(extra)
    return CallbackManager(callbacks)
