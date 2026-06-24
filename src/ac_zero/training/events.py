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

    @classmethod
    def from_name(cls, name: str) -> LogLevel:
        """Resolve a level from a case-insensitive name, defaulting to INFO."""
        try:
            return cls[name.strip().upper()]
        except KeyError:
            return cls.INFO


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
