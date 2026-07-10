from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol


class LogLevel(IntEnum):
    """Severity levels for training and CLI events, ordered like stdlib logging."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class Verbosity(IntEnum):
    """How much of a run's event stream a training console shows.

    ``VERBOSE`` is the historical behaviour: a terminal line per event plus a
    live ASCII graph re-rendered on every event. ``SUMMARY`` (the default) folds
    each iteration's episodes into one compact line and prints the final graph
    once at the end. ``QUIET`` keeps only the start/stop milestones plus warnings
    and errors. Every level still writes the full JSONL and graph files on disk.
    """

    QUIET = 0
    SUMMARY = 10
    VERBOSE = 20

    @classmethod
    def parse(cls, value: Verbosity | str) -> Verbosity:
        """Coerce a config/CLI string (or an enum) to a ``Verbosity`` level."""
        if isinstance(value, Verbosity):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError:
            names = ", ".join(level.name.lower() for level in cls)
            raise ValueError(f"verbosity must be one of {names}") from None


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """Structured progress event emitted by training, smoke, and CLI workflows."""

    step: int
    phase: str
    message: str
    metrics: dict[str, float | int | bool | str]
    level: LogLevel = LogLevel.INFO


class TrainingCallback(Protocol):
    """Callback interface for logging, terminal progress, and visual summaries."""

    def on_event(self, event: TrainingEvent) -> None:
        """Handle one progress event."""

    def close(self) -> None:
        """Flush final summaries and release resources."""
