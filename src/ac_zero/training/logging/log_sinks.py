from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from ac_zero.training.logging.events import LogLevel, TrainingEvent

DEFAULT_MAX_BYTES = 5_000_000
DEFAULT_BACKUP_COUNT = 3


class RotatingFileWriter:
    """Append text lines to a file, rotating it once it exceeds a size budget."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        encoding: str = "utf-8",
    ) -> None:
        """Open the target file and prepare size-based rotation."""
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if backup_count < 0:
            raise ValueError("backup_count must be non-negative")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._encoding = encoding
        self._handle = self.path.open("w", encoding=encoding)
        self._written = 0

    def write_line(self, line: str) -> None:
        """Append one line, rotating first when the size budget would overflow."""
        size = len((line + "\n").encode(self._encoding))
        if self._max_bytes and self._written and self._written + size > self._max_bytes:
            self._rotate()
        self._handle.write(line + "\n")
        self._handle.flush()
        self._written += size

    def _rotate(self) -> None:
        self._handle.close()
        if self._backup_count == 0:
            self._handle = self.path.open("w", encoding=self._encoding)
            self._written = 0
            return
        for index in range(self._backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.exists():
                source.replace(self._backup_path(index + 1))
        self.path.replace(self._backup_path(1))
        self._handle = self.path.open("w", encoding=self._encoding)
        self._written = 0

    def _backup_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()


def event_to_dict(event: TrainingEvent) -> dict[str, object]:
    """Serialize an event to a JSON-friendly dict with a readable level name."""
    data = asdict(event)
    data["level"] = event.level.name
    return data


class JsonlEventLogger:
    """Append every training event at or above ``min_level`` to a JSONL log."""

    def __init__(
        self,
        path: str | Path,
        *,
        min_level: LogLevel = LogLevel.DEBUG,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
    ) -> None:
        """Open a rotating JSONL log with an optional minimum level filter."""
        self._writer = RotatingFileWriter(path, max_bytes=max_bytes, backup_count=backup_count)
        self._min_level = min_level
        self.path = self._writer.path

    def on_event(self, event: TrainingEvent) -> None:
        """Write one event as stable JSON when it meets the level threshold."""
        if event.level < self._min_level:
            return
        self._writer.write_line(json.dumps(event_to_dict(event), sort_keys=True))

    def close(self) -> None:
        """Close the JSONL file handle."""
        self._writer.close()


class TerminalProgressLogger:
    """Print concise leveled progress lines and mirror them to a rotating log."""

    def __init__(
        self,
        path: str | Path,
        stream: TextIO | None = None,
        *,
        min_level: LogLevel = LogLevel.INFO,
        console: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
    ) -> None:
        """Create a terminal logger with an optional output stream and level.

        ``console=False`` keeps the rotating text mirror on disk but never prints
        to the stream, so a low-verbosity run can retain the full progress.log
        without flooding the terminal.
        """
        self._writer = RotatingFileWriter(path, max_bytes=max_bytes, backup_count=backup_count)
        self._stream = stream or sys.stdout
        self._min_level = min_level
        self._console = console
        self.path = self._writer.path

    def on_event(self, event: TrainingEvent) -> None:
        """Print and persist one progress line when it meets the level threshold."""
        if event.level < self._min_level:
            return
        metric_text = format_metrics(event.metrics)
        suffix = f" | {metric_text}" if metric_text else ""
        line = (
            f"[step {event.step:03d}] {event.level.name:<7} {event.phase}: {event.message}{suffix}"
        )
        if self._console:
            print(line, file=self._stream)
        self._writer.write_line(line)

    def close(self) -> None:
        """Close the mirrored text log."""
        self._writer.close()


class AsciiGraphLogger:
    """Render small terminal-friendly graphs during and after training."""

    def __init__(
        self,
        live_path: str | Path,
        final_path: str | Path,
        metric_names: Sequence[str],
        *,
        stream: TextIO | None = None,
        every_n_events: int = 1,
        width: int = 24,
        console_live: bool = True,
        console_final: bool = True,
    ) -> None:
        """Track selected numeric metrics and write live/final graph summaries.

        ``console_live``/``console_final`` gate whether the live and final graphs
        are printed to the stream; the ``live_graphs.txt``/``final_graphs.txt``
        files are always written, so a low-verbosity run keeps the graph record on
        disk while showing only what its level allows on the terminal.
        """
        if every_n_events <= 0:
            raise ValueError("every_n_events must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        self.live_path = Path(live_path)
        self.final_path = Path(final_path)
        for path in (self.live_path, self.final_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._metric_names = tuple(metric_names)
        self._series: dict[str, list[float]] = {name: [] for name in self._metric_names}
        self._stream = stream or sys.stdout
        self._every_n_events = every_n_events
        self._width = width
        self._console_live = console_live
        self._console_final = console_final
        self._seen_events = 0

    def on_event(self, event: TrainingEvent) -> None:
        """Collect metric values and periodically render live ASCII graphs."""
        self._seen_events += 1
        updated = False
        for name in self._metric_names:
            value = event.metrics.get(name)
            if isinstance(value, bool):
                self._series[name].append(1.0 if value else 0.0)
                updated = True
                continue
            if isinstance(value, str) or value is None:
                continue
            numeric = float(value)
            # A diverged metric (NaN/inf) must not reach the sparkline renderer,
            # where ``int(NaN)`` would abort the whole run at render/close time.
            if not math.isfinite(numeric):
                continue
            self._series[name].append(numeric)
            updated = True
        if updated and self._seen_events % self._every_n_events == 0:
            rendered = self._render(title=f"live graphs after {event.phase}")
            if self._console_live:
                print(rendered, file=self._stream)
            self.live_path.write_text(rendered + "\n", encoding="utf-8")

    def close(self) -> None:
        """Write and optionally print the final graph summaries."""
        rendered = self._render(title="final training graphs")
        if self._console_final:
            print(rendered, file=self._stream)
        self.final_path.write_text(rendered + "\n", encoding="utf-8")

    def _render(self, title: str) -> str:
        lines = [title]
        for name in self._metric_names:
            values = self._series[name]
            if values:
                lines.append(f"{name:>22}: {_sparkline(values, self._width)} {values[-1]:.4g}")
        if len(lines) == 1:
            lines.append("        no numeric metrics recorded")
        return "\n".join(lines)


def format_metrics(metrics: dict[str, float | int | bool | str]) -> str:
    """Render metric key/values as a compact ``key=value`` comma-separated string."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _sparkline(values: Sequence[float], width: int) -> str:
    if not values:
        return ""
    clipped = list(values[-width:])
    if len(clipped) == 1:
        return "."
    low = min(clipped)
    high = max(clipped)
    ticks = "._:-=+*#"
    if high == low:
        return ticks[0] * len(clipped)
    scale = (len(ticks) - 1) / (high - low)
    return "".join(ticks[int((value - low) * scale)] for value in clipped)
