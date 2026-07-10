import io
from pathlib import Path

import pytest

from ac_zero.training.callbacks import (
    AsciiGraphLogger,
    ConsoleSummaryLogger,
    TerminalProgressLogger,
    default_training_callbacks,
)
from ac_zero.training.events import LogLevel, TrainingEvent, Verbosity


def _event(
    phase: str, metrics: dict, level: LogLevel = LogLevel.INFO, step: int = 1
) -> TrainingEvent:
    return TrainingEvent(step, phase, f"{phase} message", metrics, level)


def test_verbosity_parse_accepts_names_and_enums_and_rejects_others() -> None:
    assert Verbosity.parse("quiet") is Verbosity.QUIET
    assert Verbosity.parse("SUMMARY") is Verbosity.SUMMARY
    assert Verbosity.parse(Verbosity.VERBOSE) is Verbosity.VERBOSE
    assert Verbosity.QUIET < Verbosity.SUMMARY < Verbosity.VERBOSE
    with pytest.raises(ValueError):
        Verbosity.parse("loud")


def test_terminal_logger_console_false_keeps_file_but_mutes_stream(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = TerminalProgressLogger(tmp_path / "progress.log", stream=stream, console=False)
    logger.on_event(_event("optimizer", {"total_loss": 0.5}))
    logger.close()

    assert stream.getvalue() == ""
    # The rotating text mirror still records the line.
    assert "optimizer message" in (tmp_path / "progress.log").read_text()


def test_ascii_graph_console_flags_gate_live_and_final_separately(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = AsciiGraphLogger(
        tmp_path / "live.txt",
        tmp_path / "final.txt",
        ("total_loss",),
        stream=stream,
        console_live=False,
        console_final=True,
    )
    logger.on_event(_event("optimizer", {"total_loss": 0.5}))
    # Live graph is written to disk but not printed.
    assert stream.getvalue() == ""
    assert (tmp_path / "live.txt").read_text().strip()

    logger.close()
    # The final graph prints once and is written to disk.
    assert "final training graphs" in stream.getvalue()
    assert (tmp_path / "final.txt").read_text().strip()


def test_console_summary_bundles_iteration_and_drops_per_episode_events() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    # An optimizer step primes the loss that the next iteration line folds in.
    logger.on_event(_event("optimizer", {"total_loss": 0.1234, "optimizer_step": 3}))
    # Per-episode navigation spam is suppressed on the console.
    logger.on_event(_event("navigation_episode", {"iteration": 5, "alpha": 0.3}))
    logger.on_event(
        _event(
            "self_play",
            {
                "iteration": 5,
                "episodes": 120,
                "mean_return": 0.42,
                "success_rate": 0.5,
                "replay_size": 4096,
            },
        )
    )
    printed = stream.getvalue()
    assert "navigation_episode" not in printed
    assert "iter     5" in printed
    assert "eps=120" in printed
    assert "return=+0.420" in printed
    assert "success=0.50" in printed
    assert "loss=0.1234" in printed
    assert "replay=4096" in printed


def test_console_summary_quiet_keeps_only_milestones_and_diagnostics() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream, iterations=False)
    logger.on_event(_event("start", {"seed": 0}))
    logger.on_event(_event("self_play", {"iteration": 2, "mean_return": 0.1}))
    logger.on_event(_event("completed", {"optimizer_updates": 9, "replay_size": 8}))
    logger.on_event(_event("dataset", {"errors": 1}, level=LogLevel.WARNING))
    printed = stream.getvalue()

    assert "start:" in printed
    assert "completed:" in printed
    assert "dataset WARNING:" in printed
    # No per-iteration line in quiet mode.
    assert "iter " not in printed


def test_console_summary_always_prints_warnings_even_with_iterations() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    logger.on_event(_event("optimizer", {"total_loss": 0.2}, level=LogLevel.DEBUG))
    logger.on_event(_event("budget", {"iteration": 4}, level=LogLevel.WARNING))
    printed = stream.getvalue()
    # DEBUG optimizer noise is dropped; the WARNING budget line surfaces.
    assert "optimizer" not in printed
    assert "budget WARNING:" in printed


def _callback_types(manager) -> list[str]:
    return [type(cb).__name__ for cb in manager._callbacks]


def test_default_callbacks_summary_mutes_console_sinks_and_adds_summary(tmp_path: Path) -> None:
    manager = default_training_callbacks(tmp_path / "run", verbosity="summary")
    types = _callback_types(manager)
    assert "ConsoleSummaryLogger" in types

    terminal = next(cb for cb in manager._callbacks if isinstance(cb, TerminalProgressLogger))
    graph = next(cb for cb in manager._callbacks if isinstance(cb, AsciiGraphLogger))
    assert terminal._console is False
    assert graph._console_live is False
    # Summary prints the final graph once at the end.
    assert graph._console_final is True


def test_default_callbacks_verbose_keeps_console_and_omits_summary(tmp_path: Path) -> None:
    manager = default_training_callbacks(tmp_path / "run", verbosity=Verbosity.VERBOSE)
    types = _callback_types(manager)
    assert "ConsoleSummaryLogger" not in types

    terminal = next(cb for cb in manager._callbacks if isinstance(cb, TerminalProgressLogger))
    graph = next(cb for cb in manager._callbacks if isinstance(cb, AsciiGraphLogger))
    assert terminal._console is True
    assert graph._console_live is True


def test_default_callbacks_quiet_suppresses_final_graph_and_iteration_lines(tmp_path: Path) -> None:
    manager = default_training_callbacks(tmp_path / "run", verbosity="quiet")
    graph = next(cb for cb in manager._callbacks if isinstance(cb, AsciiGraphLogger))
    summary = next(cb for cb in manager._callbacks if isinstance(cb, ConsoleSummaryLogger))
    assert graph._console_final is False
    assert summary._iterations is False


def test_default_callbacks_appends_extra_callbacks(tmp_path: Path) -> None:
    sentinel = ConsoleSummaryLogger(stream=io.StringIO())
    manager = default_training_callbacks(tmp_path / "run", verbosity="quiet", extra=(sentinel,))
    assert manager._callbacks[-1] is sentinel
