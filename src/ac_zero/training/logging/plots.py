from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PlotSpec:
    """One figure: a set of metric series drawn against a shared x-axis."""

    filename: str
    title: str
    x_field: str
    y_fields: tuple[str, ...]
    ylabel: str = "value"


# The figures rendered after a run. The fields match the per-update metric rows
# the training pipeline records, so plotting needs no extra bookkeeping. The specs
# span both kinds of run: a figure whose series are all absent from the rows is
# skipped, so an RL run writes no validation figure and a supervised run no
# self-play one.
TRAINING_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec(
        "loss_curves.png",
        "Training loss",
        "optimizer_step",
        # The supervised run's validation losses share the axis with the training
        # losses they are meant to be read against -- the gap between them is the
        # point.
        ("total_loss", "policy_loss", "value_loss", "val_policy_loss", "val_value_loss"),
        ylabel="loss",
    ),
    PlotSpec(
        "selfplay_progress.png",
        "Self-play progress",
        "optimizer_step",
        ("mean_return", "success_rate"),
    ),
    PlotSpec(
        "shaping_alpha.png",
        "Navigation shaping weight",
        "optimizer_step",
        # The navigation reward's one adaptive knob: it rises while the policy makes
        # little progress, falls once it progresses without solving, and anneals as
        # success sets in -- so its trace reads as the run's difficulty schedule.
        ("alpha",),
        ylabel="alpha",
    ),
    PlotSpec(
        "validation.png",
        "Validation (supervised)",
        "optimizer_step",
        ("val_descent_accuracy", "val_mean_delta", "val_unknown_rate"),
    ),
)


class PlotsUnavailable(RuntimeError):
    """Raised when plotting is requested but matplotlib is not installed."""


def render_training_plots(
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    specs: Sequence[PlotSpec] = TRAINING_PLOTS,
) -> list[Path]:
    """Render PNG plots of a run's metric history and return the files written.

    Reads the per-update metric rows collected during training and draws one
    figure per :class:`PlotSpec` (loss curves, self-play progress). A figure is
    skipped when none of its series carry numeric data. Returns an empty list
    when there are no rows. Raises :class:`PlotsUnavailable` if matplotlib is not
    installed, so callers can fall back to the ASCII graphs.
    """
    if not rows:
        return []
    try:
        import matplotlib

        # Force the non-interactive Agg backend so rendering never needs a display
        # and behaves identically on headless servers and CI.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotsUnavailable("matplotlib is required to render training plots") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in specs:
        series = _numeric_series(rows, spec)
        if not series:
            continue
        xs = [_as_float(row.get(spec.x_field, index)) for index, row in enumerate(rows)]
        figure, axes = plt.subplots(figsize=(8, 4.5))
        for name, values in series.items():
            axes.plot(xs[: len(values)], values, marker="o", markersize=3, label=name)
        axes.set_title(spec.title)
        axes.set_xlabel(spec.x_field)
        axes.set_ylabel(spec.ylabel)
        axes.grid(True, alpha=0.3)
        axes.legend()
        figure.tight_layout()
        path = out / spec.filename
        figure.savefig(path, dpi=120)
        plt.close(figure)
        written.append(path)
    return written


def _numeric_series(rows: Sequence[dict[str, Any]], spec: PlotSpec) -> dict[str, list[float]]:
    """Extract each plotted field's numeric values, dropping empty series."""
    series: dict[str, list[float]] = {}
    for name in spec.y_fields:
        values = [_as_float(row[name]) for row in rows if _is_number(row.get(name))]
        if values:
            series[name] = values
    return series


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0
