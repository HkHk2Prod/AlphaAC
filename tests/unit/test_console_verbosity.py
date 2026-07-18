import io
from pathlib import Path

import pytest

from ac_zero.training.logging.callbacks import (
    AsciiGraphLogger,
    ConsoleSummaryLogger,
    TerminalProgressLogger,
    default_training_callbacks,
)
from ac_zero.training.logging.events import LogLevel, TrainingEvent, Verbosity


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
                # The run's dynamic learning parameter, folded onto the iteration line.
                "alpha": 0.3,
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
    # Each iteration line shows the current value of the run's dynamic parameter.
    assert "alpha=0.300" in printed


def test_console_iteration_line_omits_inactive_dynamic_parameters() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    # A non-navigation run carries no alpha, so the line adds no params.
    logger.on_event(_event("self_play", {"iteration": 1, "episodes": 4, "mean_return": 0.1}))
    printed = stream.getvalue()
    assert "iter     1" in printed
    assert "alpha=" not in printed


def test_console_summary_prints_a_line_for_each_supervised_epoch() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    logger.on_event(
        _event(
            "epoch",
            {
                "iteration": 7,
                "optimizer_step": 700,
                "batch_size": 64,
                "policy_loss": 0.25,
                "value_loss": 0.5,
                "total_loss": 0.3456,
                "val_policy_loss": 0.2718,
                "val_descent_accuracy": 0.812,
                "val_mean_delta": -0.6,
            },
        )
    )
    printed = stream.getvalue()
    # The epoch line names its unit and carries the training loss plus the
    # validation metrics the run actually selects its best checkpoint on.
    assert "epoch     7" in printed
    assert "loss=0.3456" in printed
    assert "val_acc=0.812" in printed
    assert "val_loss=0.2718" in printed
    assert "steps=700" in printed
    # A supervised epoch has no self-play fields to report.
    assert "return=" not in printed
    assert "replay=" not in printed


def test_console_summary_prints_supervised_milestones() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    # The sidecar build precedes the first epoch by minutes on a large ball, so its
    # progress has to reach the terminal or the run reads as a hang.
    logger.on_event(_event("sidecar", {"groups": 2089615}))
    logger.on_event(_event("model", {"parameters": 99000000}))
    logger.on_event(_event("test", {"test_descent_accuracy": 0.79, "test_groups": 4096}))
    printed = stream.getvalue()
    assert "sidecar:" in printed
    assert "groups=2089615" in printed
    assert "model:" in printed
    assert "parameters=99000000" in printed
    assert "test:" in printed
    assert "test_descent_accuracy=0.79" in printed


def test_console_summary_quiet_drops_supervised_epoch_lines() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream, iterations=False)
    logger.on_event(_event("epoch", {"iteration": 3, "total_loss": 0.2}))
    assert stream.getvalue() == ""


def test_console_summary_start_prints_every_parameter() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    logger.on_event(_event("start", {"seed": 7, "learning_rate": 0.05, "agent": "alphazero"}))
    printed = stream.getvalue()
    # The banner plus one indented line per run parameter (sorted).
    assert "start: start message" in printed
    assert "  agent = alphazero" in printed
    assert "  learning_rate = 0.05" in printed
    assert "  seed = 7" in printed


def test_console_summary_dataset_line_names_the_instance_source() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    # A grown-dataset run must visibly report `source=dataset` (plus how many
    # groups seed self-play) so it is not mistaken for random-scramble self-play.
    logger.on_event(
        _event(
            "dataset",
            {"source": "dataset", "groups_used": 2089615, "annotated": 2089615, "rank": 2},
        )
    )
    printed = stream.getvalue()
    assert "dataset: dataset message" in printed
    assert "source=dataset" in printed
    assert "groups_used=2089615" in printed
    # A scramble run reports its source too, without the dataset-only fields.
    scramble = io.StringIO()
    ConsoleSummaryLogger(stream=scramble).on_event(
        _event("dataset", {"source": "scramble", "rank": 2, "depth": 3})
    )
    assert "source=scramble" in scramble.getvalue()
    assert "groups_used" not in scramble.getvalue()
    # The supervised run describes its dataset with its own keys: labelled groups
    # and the sizes of the splits it trains, scores, and tests on.
    supervised = io.StringIO()
    ConsoleSummaryLogger(stream=supervised).on_event(
        _event("dataset", {"groups": 200, "train": 160, "val": 20, "test": 20, "moveset": "ac"})
    )
    assert "groups=200" in supervised.getvalue()
    assert "train=160" in supervised.getvalue()
    assert "val=20" in supervised.getvalue()


def test_console_summary_drops_checkpoint_events() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream)
    # Checkpoint saves are recorded to JSONL but no longer clutter the console.
    logger.on_event(_event("checkpoint", {"iteration": 3, "optimizer_step": 12}))
    assert "checkpoint" not in stream.getvalue()


def test_console_summary_quiet_keeps_only_milestones_and_diagnostics() -> None:
    stream = io.StringIO()
    logger = ConsoleSummaryLogger(stream=stream, iterations=False)
    logger.on_event(_event("start", {"seed": 0}))
    logger.on_event(_event("self_play", {"iteration": 2, "mean_return": 0.1}))
    logger.on_event(_event("checkpoint", {"iteration": 2}))
    logger.on_event(_event("completed", {"optimizer_updates": 9, "replay_size": 8}))
    logger.on_event(_event("dataset", {"errors": 1}, level=LogLevel.WARNING))
    printed = stream.getvalue()

    assert "start:" in printed
    assert "completed:" in printed
    assert "dataset WARNING:" in printed
    # Checkpoints and iteration lines do not surface in quiet mode.
    assert "checkpoint" not in printed
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


def test_default_graphs_track_supervised_validation_curves(tmp_path: Path) -> None:
    manager = default_training_callbacks(tmp_path / "run", verbosity="summary")
    graph = next(cb for cb in manager._callbacks if isinstance(cb, AsciiGraphLogger))
    for epoch, accuracy in ((1, 0.25), (2, 0.5)):
        manager.emit(
            epoch, "epoch", "trained", {"total_loss": 1.0, "val_descent_accuracy": accuracy}
        )
    manager.close()

    final = (tmp_path / "run" / "artifacts" / "final_graphs.txt").read_text(encoding="utf-8")
    # The supervised run's graphs show the score it selects its best checkpoint on,
    # and none of the self-play rows it never emits.
    assert "val_descent_accuracy" in final
    assert "total_loss" in final
    assert "mean_return" not in final
    assert "replay_size" not in final
    assert "val_descent_accuracy" in graph._metric_names


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
