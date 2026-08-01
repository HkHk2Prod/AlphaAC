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
        # Self-play is measured once per iteration, so that -- not the optimizer
        # step, which advances many times inside one measurement -- is its axis.
        "iteration",
        # Both fractional: the share of episodes that reached the destination, and
        # the share of the start distance the average episode closed. Progress is
        # the leading indicator -- it moves long before the first solve does.
        ("success_rate", "progress_rate"),
        ylabel="rate",
        right_fields=("mean_return",),
        right_ylabel="mean_return",
    ),
    PlotSpec(
        "shaping_alpha.png",
        "Navigation shaping weight",
        "iteration",
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
        figure, axes = plt.subplots(figsize=(10, 4.8))
        left_names = [name for name in spec.y_fields if name in series]
        right_names = [name for name in spec.right_fields if name in series]

        handles = []
        # Right-axis series take the first colours of the cycle, left-axis the rest,
        # so on the self-play figure mean return is blue and success rate orange.
        for offset, name in enumerate(left_names, start=len(right_names)):
            handles.append(
                _draw_series(axes, series[name], color_cycle[offset % len(color_cycle)], name)
            )
        axes.set_ylabel(spec.ylabel)
        _tint_axis(axes, "left", handles[:1] if len(left_names) == 1 else [])

        if right_names:
            right_ax = axes.twinx()
            for offset, name in enumerate(right_names):
                handles.append(
                    _draw_series(
                        right_ax, series[name], color_cycle[offset % len(color_cycle)], name
                    )
                )
            right_ax.set_ylabel(spec.right_ylabel)
            # A faint reference line at mean return zero: the sign flip that separates
            # runs that lose reward on average from those that gain it.
            right_ax.axhline(0.0, color=_MUTED, linewidth=0.8, alpha=0.35, zorder=0)
            right_ax.spines["top"].set_visible(False)
            right_ax.tick_params(labelsize=9)
            _tint_axis(right_ax, "right", handles[-1:] if len(right_names) == 1 else [])

        for boundary in _session_boundaries(rows, spec.x_field):
            axes.axvline(boundary, color=_MUTED, linestyle=":", linewidth=1.1, alpha=0.65, zorder=0)

        axes.set_title(spec.title, loc="left")
        axes.set_xlabel(spec.x_field)
        _style_axis(axes)
        axes.legend(
            handles=handles,
            labels=[line.get_label() for line in handles],
            loc="upper left",
            framealpha=0.92,
            fontsize=9,
        )
        figure.tight_layout()
        path = out / spec.filename
        figure.savefig(path, dpi=120)
        plt.close(figure)
        written.append(path)
    return written


# A long run's per-iteration series is far too noisy to read point by point, so past
# this many points the raw trace is drawn faintly behind a rolling mean that carries
# the trend. Below it the raw markers are the clearest thing to show.
_SMOOTH_ABOVE = 60
# Rolling window as a share of the series, bounded so it neither vanishes on a short
# run nor flattens away a real turn on a long one.
_WINDOW_FRACTION = 0.02
_WINDOW_BOUNDS = (5, 250)
_MUTED = "#5A6474"


def _rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Trailing mean over the last ``window`` values, seeded from the first point."""
    out: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        out.append(total / min(index + 1, window))
    return out


def _draw_series(axis: Any, points: Points, color: str, name: str) -> Any:
    """Draw one series and return the line the legend should carry.

    A short series is drawn as plain markers. A long one gets its raw trace at low
    opacity behind a rolling mean, because a per-iteration self-play measurement
    swings by most of its own range between neighbouring points -- the batch is 32
    episodes -- and the trend is the part worth reading.
    """
    xs, ys = _xy(points)
    if len(points) <= _SMOOTH_ABOVE:
        (line,) = axis.plot(xs, ys, marker="o", markersize=3, color=color, label=name)
        return line
    low, high = _WINDOW_BOUNDS
    window = min(high, max(low, int(len(points) * _WINDOW_FRACTION)))
    axis.plot(xs, ys, color=color, linewidth=0.4, alpha=0.22, zorder=1)
    (line,) = axis.plot(
        xs,
        _rolling_mean(ys, window),
        color=color,
        linewidth=1.9,
        zorder=2,
        label=f"{name} ({window}-pt mean)",
    )
    return line


def _session_boundaries(rows: Sequence[dict[str, Any]], x_field: str) -> list[float]:
    """The x positions where one run's rows give way to the next.

    The all-runs figure concatenates every session of a lineage, and a resume is
    exactly where a checkpoint round trip could go wrong, so it is worth marking.
    Empty for a single-run figure, whose rows all carry the same ``run``.
    """
    boundaries: list[float] = []
    previous: object = None
    for row in rows:
        run = row.get("run")
        if run is None or not _is_number(row.get(x_field)):
            continue
        if previous is not None and run != previous:
            boundaries.append(_as_float(row[x_field]))
        previous = run
    return boundaries


def _style_axis(axis: Any) -> None:
    """Quiet the frame so the data carries the figure."""
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=9)


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


Points = list[tuple[float, float]]


def _numeric_series(rows: Sequence[dict[str, Any]], spec: PlotSpec) -> dict[str, Points]:
    """Extract each plotted field's ``(x, y)`` points, dropping empty series.

    A row contributes to a series only when it carries both that field and the
    figure's x field, and it carries its own x with it. Series recorded at
    different cadences therefore keep their own axis values: the per-iteration
    self-play summary and the per-minibatch losses live in the same row list, and
    pairing every series against a shared row index would have drawn one of them
    against the other's x values.
    """
    series: dict[str, Points] = {}
    for name in (*spec.y_fields, *spec.right_fields):
        points = [
            (_as_float(row[spec.x_field]), _as_float(row[name]))
            for row in rows
            if _is_number(row.get(name)) and _is_number(row.get(spec.x_field))
        ]
        if points:
            series[name] = points
    return series


def _xy(points: Points) -> tuple[list[float], list[float]]:
    """Split ``(x, y)`` pairs into the two sequences matplotlib plots."""
    return [x for x, _ in points], [y for _, y in points]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0
