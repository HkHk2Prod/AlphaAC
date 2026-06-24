from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TextIO


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """Structured progress event emitted by training and smoke workflows."""

    step: int
    phase: str
    message: str
    metrics: dict[str, float | int | bool | str]


class TrainingCallback(Protocol):
    """Callback interface for logging, terminal progress, and visual summaries."""

    def on_event(self, event: TrainingEvent) -> None:
        """Handle one progress event."""

    def close(self) -> None:
        """Flush final summaries and release resources."""


class CallbackManager:
    """Fan out training events to multiple callbacks."""

    def __init__(self, callbacks: Iterable[TrainingCallback]) -> None:
        """Create a manager from callbacks in deterministic invocation order."""
        self._callbacks = tuple(callbacks)

    def emit(
        self,
        step: int,
        phase: str,
        message: str,
        metrics: dict[str, float | int | bool | str] | None = None,
    ) -> TrainingEvent:
        """Create and dispatch one `TrainingEvent`."""

        event = TrainingEvent(step, phase, message, metrics or {})
        for callback in self._callbacks:
            callback.on_event(event)
        return event

    def close(self) -> None:
        """Close callbacks in registration order."""

        for callback in self._callbacks:
            callback.close()


class JsonlEventLogger:
    """Append every training event to a JSONL log file."""

    def __init__(self, path: str | Path) -> None:
        """Open the target file and create parent directories."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def on_event(self, event: TrainingEvent) -> None:
        """Write one event as stable JSON."""
        self._handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the JSONL file handle."""
        self._handle.close()


class TerminalProgressLogger:
    """Print concise progress lines and mirror them to a text log file."""

    def __init__(self, path: str | Path, stream: TextIO | None = None) -> None:
        """Create a terminal logger with an optional output stream override."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._stream = stream or sys.stdout

    def on_event(self, event: TrainingEvent) -> None:
        """Print and persist one progress line."""
        metric_text = _format_metrics(event.metrics)
        suffix = f" | {metric_text}" if metric_text else ""
        line = f"[step {event.step:03d}] {event.phase}: {event.message}{suffix}"
        print(line, file=self._stream)
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the mirrored text log."""
        self._handle.close()


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
    ) -> None:
        """Track selected numeric metrics and write live/final graph summaries."""
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
        self._seen_events = 0

    def on_event(self, event: TrainingEvent) -> None:
        """Collect metric values and periodically print live ASCII graphs."""
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
            self._series[name].append(float(value))
            updated = True
        if updated and self._seen_events % self._every_n_events == 0:
            rendered = self._render(title=f"live graphs after {event.phase}")
            print(rendered, file=self._stream)
            self.live_path.write_text(rendered + "\n", encoding="utf-8")

    def close(self) -> None:
        """Write and print final graph summaries."""
        rendered = self._render(title="final training graphs")
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


def default_training_callbacks(run_directory: str | Path) -> CallbackManager:
    """Create default callbacks for config-driven training pipeline runs."""

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
        )
    )


def _format_metrics(metrics: dict[str, float | int | bool | str]) -> str:
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
