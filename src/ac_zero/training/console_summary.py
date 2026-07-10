"""Compact console sink that folds a run's episodes into per-iteration summaries.

Split out of :mod:`ac_zero.training.log_sinks` because it is a different kind of
sink: rather than mirroring every event, it suppresses the per-episode and
per-optimizer-step stream (still recorded to the JSONL log and the graph files)
and prints one bundled line per logged iteration plus the run's milestones. It is
what the ``summary``/``quiet`` verbosity levels put on the terminal in place of
the historical per-event flood (see :class:`ac_zero.training.events.Verbosity`).
"""

from __future__ import annotations

import sys
from typing import TextIO

from ac_zero.training.events import LogLevel, TrainingEvent

# Loss keys carried on optimizer events; the newest value is folded into the next
# iteration line so a bundled summary reports return, success, and loss together.
_LOSS_KEYS = ("total_loss", "policy_loss", "value_loss")

# The subset of each milestone phase's metrics worth showing on its console line;
# the rest (histograms, per-component breakdowns) stay in the JSONL record.
_MILESTONE_FIELDS: dict[str, tuple[str, ...]] = {
    "self_play": (),  # the worker-pool description event: message only
    # Show where episodes start from: `source` (dataset vs scramble) plus, for a
    # grown dataset, how many groups are in play -- so the terminal confirms a run
    # is seeded from the HF dataset rather than random scrambles.
    "dataset": ("source", "groups_used", "annotated"),
    "navigation": ("success_rate", "progress_rate", "alpha"),
    "curriculum": ("L_max", "frontier_episode_fraction"),
    "length_cap": ("L_max", "direction", "max_moves"),
    "budget": ("iteration",),
    "certificate": ("certificate_verified",),
    "completed": ("optimizer_updates", "replay_size", "total_loss"),
}

# Milestones important enough to surface even at the ``quiet`` level: the run's
# terminal state plus every distance-curriculum length-cap change.
_QUIET_MILESTONES = ("completed", "budget", "length_cap")


class ConsoleSummaryLogger:
    """Print one compact line per logged iteration, folding episodes into a summary.

    Warnings and errors always print. ``iterations=False`` keeps only the
    start/stop milestones and diagnostics -- the ``quiet`` verbosity level.
    """

    def __init__(self, stream: TextIO | None = None, *, iterations: bool = True) -> None:
        """Create the summary logger, optionally muting the per-iteration lines."""
        self._stream = stream or sys.stdout
        self._iterations = iterations
        self._loss: dict[str, float] = {}

    def on_event(self, event: TrainingEvent) -> None:
        """Fold loss state and print a line for iterations, milestones, and diagnostics."""
        self._track_loss(event)
        line = self._format(event)
        if line is not None:
            print(line, file=self._stream, flush=True)

    def close(self) -> None:
        """No resources to release; the graph/JSONL sinks own the files."""

    def _track_loss(self, event: TrainingEvent) -> None:
        for key in _LOSS_KEYS:
            value = event.metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._loss[key] = float(value)

    def _format(self, event: TrainingEvent) -> str | None:
        if event.level >= LogLevel.WARNING:
            return self._diagnostic(event)
        if event.level < LogLevel.INFO:
            return None
        if event.phase == "start":
            return self._start_report(event)
        if not self._iterations:
            return self._milestone(event) if event.phase in _QUIET_MILESTONES else None
        if event.phase == "self_play" and "iteration" in event.metrics:
            return self._iteration_line(event)
        if event.phase in _MILESTONE_FIELDS:
            return self._milestone(event)
        return None

    def _iteration_line(self, event: TrainingEvent) -> str:
        """Bundle one iteration's episodes into a single return/success/loss line."""
        metrics = event.metrics
        parts = [f"iter {int(metrics['iteration']):>5}"]
        if "episodes" in metrics:
            parts.append(f"eps={int(metrics['episodes'])}")
        if "mean_return" in metrics:
            parts.append(f"return={float(metrics['mean_return']):+.3f}")
        if "success_rate" in metrics:
            parts.append(f"success={float(metrics['success_rate']):.2f}")
        if "total_loss" in self._loss:
            parts.append(f"loss={self._loss['total_loss']:.4f}")
        if "replay_size" in metrics:
            parts.append(f"replay={int(metrics['replay_size'])}")
        elif "examples" in metrics:
            parts.append(f"examples={int(metrics['examples'])}")
        return "  ".join(parts)

    def _start_report(self, event: TrainingEvent) -> str:
        """Print the opening banner followed by every run parameter, one per line.

        The ``start`` event carries the full ``run_description`` config dump; the
        summary console renders it as a readable block so a run is fully described
        on the terminal from its first line, not just in the JSONL log.
        """
        lines = [f"start: {event.message}"]
        for key in sorted(event.metrics):
            value = event.metrics[key]
            rendered = f"{value:.4g}" if isinstance(value, float) else value
            lines.append(f"  {key} = {rendered}")
        return "\n".join(lines)

    def _milestone(self, event: TrainingEvent) -> str:
        fields = _MILESTONE_FIELDS.get(event.phase, ())
        suffix = _select(event.metrics, fields)
        return f"{event.phase}: {event.message}{suffix}"

    def _diagnostic(self, event: TrainingEvent) -> str:
        suffix = _select(event.metrics, tuple(event.metrics))
        return f"{event.phase} {event.level.name}: {event.message}{suffix}"


def _select(metrics: dict[str, float | int | bool | str], keys: tuple[str, ...]) -> str:
    """Render the chosen metric keys (in order, those present) as ``| k=v`` text."""
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return f" | {', '.join(parts)}" if parts else ""
