from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ac_zero.training.logging.plots import (
    PlotSpec,
    _draw_series,
    _numeric_series,
    _session_boundaries,
    render_training_plots,
)


@dataclass
class _FakeLine:
    """The bit of a matplotlib Line2D the drawing helper actually uses."""

    ys: list[float]
    label: str = ""

    def get_label(self) -> str:
        return self.label

    def get_color(self) -> str:
        return "#000000"


@dataclass
class _FakeAxis:
    """Records what would be drawn, so the helper is testable without a canvas."""

    plots: list[dict] = field(default_factory=list)

    def plot(self, xs: list[float], ys: list[float], **kwargs: object) -> tuple[_FakeLine]:
        self.plots.append({"xs": xs, "ys": ys, **kwargs})
        return (_FakeLine(ys, str(kwargs.get("label", ""))),)


# What an RL run records: one self-play row per iteration, then one row per
# optimizer step inside it. The two cadences share the list and carry different
# x fields, which is what the figures have to keep straight.
_ROWS = [
    {"iteration": 1, "mean_return": 0.2, "success_rate": 0.0, "progress_rate": 0.1},
    {"iteration": 1, "optimizer_step": 1, "total_loss": 2.5, "policy_loss": 2.4, "value_loss": 0.1},
    {"iteration": 1, "optimizer_step": 2, "total_loss": 2.0, "policy_loss": 1.9, "value_loss": 0.1},
    {"iteration": 2, "mean_return": 0.6, "success_rate": 0.5, "progress_rate": 0.4},
    {"iteration": 2, "optimizer_step": 3, "total_loss": 1.5, "policy_loss": 1.4, "value_loss": 0.1},
]


def test_render_training_plots_writes_expected_pngs(tmp_path: Path) -> None:
    paths = render_training_plots(_ROWS, tmp_path)
    names = {path.name for path in paths}
    # An RL run carries no `val_*` series, so it gets no supervised validation figure.
    assert names == {"loss_curves.png", "selfplay_progress.png"}
    for path in paths:
        # A real PNG file with content, written under the requested directory.
        assert path.parent == tmp_path
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


_SUPERVISED_ROWS = [
    {
        "optimizer_step": 100 * step,
        "total_loss": 2.5 - step,
        "policy_loss": 2.4 - step,
        "value_loss": 0.1,
        "val_policy_loss": 2.6 - step,
        "val_descent_accuracy": 0.2 * step,
        "val_mean_delta": -0.2 * step,
        "val_unknown_rate": 0.05,
    }
    for step in (1, 2, 3)
]


def test_render_training_plots_draws_the_supervised_validation_curves(tmp_path: Path) -> None:
    # A supervised run has no self-play, so that figure is skipped -- but its
    # validation scores (what the run picks its best checkpoint on) get a figure,
    # and its validation loss shares the axis with the training loss.
    paths = render_training_plots(_SUPERVISED_ROWS, tmp_path)
    names = {path.name for path in paths}
    assert names == {"loss_curves.png", "validation.png"}
    for path in paths:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_training_plots_draws_dual_axis_series(tmp_path: Path) -> None:
    # A spec with right_fields draws those series on a secondary right-hand axis
    # that autoscales independently of the left, and the legend carries both.
    spec = PlotSpec(
        "dual.png",
        "Dual",
        "iteration",
        ("success_rate",),
        ylabel="accuracy",
        right_fields=("mean_return",),
        right_ylabel="mean return",
    )
    paths = render_training_plots(_ROWS, tmp_path, specs=(spec,))
    assert [path.name for path in paths] == ["dual.png"]
    assert paths[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_numeric_series_pairs_each_point_with_its_own_x() -> None:
    # The self-play series is recorded once per iteration and the loss series once
    # per optimizer step, interleaved in one row list. Each must carry its own x:
    # pairing them against a shared row index drew self-play at the losses' x.
    spec = PlotSpec("s.png", "S", "iteration", ("success_rate",), right_fields=("mean_return",))
    assert _numeric_series(_ROWS, spec) == {
        "success_rate": [(1.0, 0.0), (2.0, 0.5)],
        "mean_return": [(1.0, 0.2), (2.0, 0.6)],
    }


def test_long_series_are_drawn_as_a_rolling_mean_over_a_faint_trace() -> None:
    # Past the smoothing threshold a series gets two lines — the raw trace and the
    # mean that carries the trend — and the legend names the one worth reading.
    rows = [{"iteration": i, "success_rate": (i % 2) * 0.5} for i in range(500)]
    spec = PlotSpec("s.png", "S", "iteration", ("success_rate",))
    axis = _FakeAxis()
    line = _draw_series(axis, _numeric_series(rows, spec)["success_rate"], "#000", "success_rate")
    assert len(axis.plots) == 2
    assert axis.plots[0]["alpha"] < 0.5 and "label" not in axis.plots[0]
    # 2% of 500 points, inside the (5, 250) bounds.
    assert line.label == "success_rate (10-pt mean)"
    # The mean of an alternating 0 / 0.5 series settles at 0.25, not at either value.
    assert line.ys[-1] == pytest.approx(0.25)


def test_short_series_keep_their_raw_markers() -> None:
    rows = [{"iteration": i, "success_rate": 0.5} for i in range(10)]
    spec = PlotSpec("s.png", "S", "iteration", ("success_rate",))
    axis = _FakeAxis()
    line = _draw_series(axis, _numeric_series(rows, spec)["success_rate"], "#000", "success_rate")
    assert len(axis.plots) == 1
    assert line.label == "success_rate"


def test_session_boundaries_mark_where_a_run_hands_over() -> None:
    rows = [
        {"iteration": 1, "run": "a"},
        {"iteration": 2, "run": "a"},
        {"iteration": 3, "run": "b"},
        {"iteration": 4, "run": "b"},
        {"iteration": 5, "run": "c"},
    ]
    assert _session_boundaries(rows, "iteration") == [3.0, 5.0]


def test_a_single_run_figure_has_no_session_boundaries() -> None:
    # A run plotting its own rows has no `run` key at all, so nothing is marked.
    assert _session_boundaries([{"iteration": 1}, {"iteration": 2}], "iteration") == []


def test_numeric_series_skips_rows_missing_the_x_field() -> None:
    # The loss rows carry `optimizer_step`; the self-play rows do not, and must not
    # be silently drawn at some neighbouring row's step.
    spec = PlotSpec("l.png", "L", "optimizer_step", ("total_loss",))
    assert _numeric_series(_ROWS, spec) == {"total_loss": [(1.0, 2.5), (2.0, 2.0), (3.0, 1.5)]}


def test_render_training_plots_handles_no_rows(tmp_path: Path) -> None:
    assert render_training_plots([], tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_render_training_plots_skips_figures_without_numeric_data(tmp_path: Path) -> None:
    # Only the x-axis is present, so every y-series is empty and nothing is drawn.
    rows = [{"optimizer_step": 1, "iteration": 1}, {"optimizer_step": 2, "iteration": 1}]
    assert render_training_plots(rows, tmp_path) == []


def test_render_training_plots_ignores_missing_and_boolean_values(tmp_path: Path) -> None:
    spec = PlotSpec("loss.png", "Loss", "optimizer_step", ("total_loss", "flag"))
    rows = [
        {"optimizer_step": 1, "total_loss": 1.0, "flag": True},
        {"optimizer_step": 2, "total_loss": 0.5, "flag": False},
    ]
    paths = render_training_plots(rows, tmp_path, specs=(spec,))
    # total_loss drives the figure; the boolean `flag` series is dropped, but the
    # numeric series alone is enough to render the chart.
    assert [path.name for path in paths] == ["loss.png"]


def test_render_training_plots_raises_when_matplotlib_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    from ac_zero.training.logging.plots import PlotsUnavailable

    real_import = builtins.__import__

    def _no_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)
    with pytest.raises(PlotsUnavailable):
        render_training_plots(_ROWS, tmp_path)
