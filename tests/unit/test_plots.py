from pathlib import Path

import pytest

from ac_zero.training.plots import PlotSpec, render_training_plots

_ROWS = [
    {
        "optimizer_step": 1,
        "total_loss": 2.5,
        "policy_loss": 2.4,
        "value_loss": 0.1,
        "mean_return": 0.2,
        "success_rate": 0.0,
    },
    {
        "optimizer_step": 2,
        "total_loss": 2.0,
        "policy_loss": 1.9,
        "value_loss": 0.1,
        "mean_return": 0.4,
        "success_rate": 0.5,
    },
    {
        "optimizer_step": 3,
        "total_loss": 1.5,
        "policy_loss": 1.4,
        "value_loss": 0.1,
        "mean_return": 0.6,
        "success_rate": 1.0,
    },
]


def test_render_training_plots_writes_expected_pngs(tmp_path: Path) -> None:
    paths = render_training_plots(_ROWS, tmp_path)
    names = {path.name for path in paths}
    assert names == {"loss_curves.png", "selfplay_progress.png"}
    for path in paths:
        # A real PNG file with content, written under the requested directory.
        assert path.parent == tmp_path
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_training_plots_handles_no_rows(tmp_path: Path) -> None:
    assert render_training_plots([], tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_render_training_plots_skips_figures_without_numeric_data(tmp_path: Path) -> None:
    # Only the x-axis is present, so every y-series is empty and nothing is drawn.
    rows = [{"optimizer_step": 1}, {"optimizer_step": 2}]
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

    from ac_zero.training.plots import PlotsUnavailable

    real_import = builtins.__import__

    def _no_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)
    with pytest.raises(PlotsUnavailable):
        render_training_plots(_ROWS, tmp_path)
