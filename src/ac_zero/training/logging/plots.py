from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PlotSpec:
    """One figure: a set of metric series drawn against a shared x-axis.

    ``right_fields`` names series that belong on a secondary y-axis drawn on the
    right. Each axis autoscales independently, so series whose magnitudes differ
    by an order of magnitude -- self-play's fractional accuracy against its much
    larger mean return -- both fill the plot area and stay legible together.
    """

    filename: str
    title: str
    x_field: str
    y_fields: tuple[str, ...]
    ylabel: str = "value"
    right_fields: tuple[str, ...] = ()
    right_ylabel: str = "value"


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
        # Accuracy is fractional; mean return runs an order of magnitude larger, so
        # they get separate axes -- accuracy on the left, mean return on the right.
        ("success_rate",),
        ylabel="success_rate",
        right_fields=("mean_return",),
        right_ylabel="mean_return",
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
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for spec in specs:
        series = _numeric_series(rows, spec)
        if not series:
            continue
        xs = [_as_float(row.get(spec.x_field, index)) for index, row in enumerate(rows)]
        figure, axes = plt.subplots(figsize=(8, 4.5))
        left_names = [name for name in spec.y_fields if name in series]
        right_names = [name for name in spec.right_fields if name in series]

        handles = []
        # Right-axis series take the first colours of the cycle, left-axis the rest,
        # so on the self-play figure mean return is blue and success rate orange.
        for offset, name in enumerate(left_names, start=len(right_names)):
            values = series[name]
            (line,) = axes.plot(
                xs[: len(values)],
                values,
                marker="o",
                markersize=3,
                color=color_cycle[offset % len(color_cycle)],
                label=name,
            )
            handles.append(line)
        axes.set_ylabel(spec.ylabel)
        _tint_axis(axes, "left", handles[:1] if len(left_names) == 1 else [])

        if right_names:
            right_ax = axes.twinx()
            for offset, name in enumerate(right_names):
                values = series[name]
                (line,) = right_ax.plot(
                    xs[: len(values)],
                    values,
                    marker="o",
                    markersize=3,
                    color=color_cycle[offset % len(color_cycle)],
                    label=name,
                )
                handles.append(line)
            right_ax.set_ylabel(spec.right_ylabel)
            # A faint reference line at mean return zero: the sign flip that separates
            # runs that lose reward on average from those that gain it.
            right_ax.axhline(0.0, color="#404040", linewidth=0.8, alpha=0.4, zorder=0)
            _tint_axis(right_ax, "right", handles[-1:] if len(right_names) == 1 else [])

        axes.set_title(spec.title)
        axes.set_xlabel(spec.x_field)
        axes.grid(True, alpha=0.3)
        axes.legend(handles=handles, labels=[line.get_label() for line in handles])
        figure.tight_layout()
        path = out / spec.filename
        figure.savefig(path, dpi=120)
        plt.close(figure)
        written.append(path)
    return written


def _tint_axis(axis: Any, side: str, lines: Sequence[Any]) -> None:
    """Colour an axis's label and ticks to match its curve.

    Only applied when the axis carries a single series: on a dual-scale figure that
    is what tells the reader which curve each scale belongs to. With two or more
    series sharing the axis there is no one colour to use, so it is left default.
    """
    if not lines:
        return
    color = lines[0].get_color()
    axis.yaxis.label.set_color(color)
    axis.tick_params(axis="y", colors=color)
    axis.spines[side].set_color(color)


def _numeric_series(rows: Sequence[dict[str, Any]], spec: PlotSpec) -> dict[str, list[float]]:
    """Extract each plotted field's numeric values, dropping empty series."""
    series: dict[str, list[float]] = {}
    for name in (*spec.y_fields, *spec.right_fields):
        values = [_as_float(row[name]) for row in rows if _is_number(row.get(name))]
        if values:
            series[name] = values
    return series


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0
