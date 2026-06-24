import io
import json
from pathlib import Path

import pytest

from ac_zero.system.reporting import CliReporter
from ac_zero.training.callbacks import CallbackManager
from ac_zero.training.events import LogLevel, TrainingEvent
from ac_zero.training.log_sinks import (
    JsonlEventLogger,
    RotatingFileWriter,
    TerminalProgressLogger,
    event_to_dict,
)


def test_log_level_from_name_is_case_insensitive_with_default() -> None:
    assert LogLevel.from_name("warning") is LogLevel.WARNING
    assert LogLevel.from_name(" ERROR ") is LogLevel.ERROR
    assert LogLevel.from_name("nonsense") is LogLevel.INFO


def test_rotating_writer_rotates_and_keeps_backups(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    writer = RotatingFileWriter(path, max_bytes=10, backup_count=2)
    for index in range(5):
        writer.write_line(f"line-{index}")
    writer.close()

    assert path.exists()
    assert (tmp_path / "log.txt.1").exists()
    assert (tmp_path / "log.txt.2").exists()
    # backup_count=2 means only two rotated files are retained.
    assert not (tmp_path / "log.txt.3").exists()
    assert path.read_text(encoding="utf-8").strip() == "line-4"


def test_rotating_writer_truncates_when_no_backups_kept(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    writer = RotatingFileWriter(path, max_bytes=10, backup_count=0)
    writer.write_line("first-line")
    writer.write_line("second-line")
    writer.close()

    assert not (tmp_path / "log.txt.1").exists()
    assert path.read_text(encoding="utf-8").strip() == "second-line"


def test_rotating_writer_writes_oversized_line_without_infinite_rotation(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    writer = RotatingFileWriter(path, max_bytes=4, backup_count=1)
    writer.write_line("a-very-long-line")
    writer.close()
    assert path.read_text(encoding="utf-8").strip() == "a-very-long-line"


def test_rotating_writer_rejects_negative_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RotatingFileWriter(tmp_path / "x", max_bytes=-1)
    with pytest.raises(ValueError):
        RotatingFileWriter(tmp_path / "x", backup_count=-1)


def test_jsonl_logger_filters_below_min_level_and_records_level_name(tmp_path: Path) -> None:
    logger = JsonlEventLogger(tmp_path / "events.jsonl", min_level=LogLevel.INFO)
    logger.on_event(TrainingEvent(0, "debug", "skipped", {}, LogLevel.DEBUG))
    logger.on_event(TrainingEvent(1, "info", "kept", {"x": 2}, LogLevel.INFO))
    logger.close()

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["level"] == "INFO"
    assert rows[0]["phase"] == "info"


def test_event_to_dict_uses_readable_level() -> None:
    event = TrainingEvent(3, "p", "m", {"a": 1}, LogLevel.ERROR)
    assert event_to_dict(event)["level"] == "ERROR"


def test_terminal_logger_includes_level_and_respects_threshold(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = TerminalProgressLogger(
        tmp_path / "progress.log", stream=stream, min_level=LogLevel.WARNING
    )
    logger.on_event(TrainingEvent(1, "info", "quiet", {}, LogLevel.INFO))
    logger.on_event(TrainingEvent(2, "boom", "loud", {}, LogLevel.ERROR))
    logger.close()

    printed = stream.getvalue()
    assert "quiet" not in printed
    assert "ERROR" in printed
    assert "boom: loud" in printed
    assert "quiet" not in (tmp_path / "progress.log").read_text()


def test_callback_manager_emit_error_follows_last_step_and_annotates_exception() -> None:
    captured: list[TrainingEvent] = []

    class _Spy:
        def on_event(self, event: TrainingEvent) -> None:
            captured.append(event)

        def close(self) -> None:
            return None

    manager = CallbackManager((_Spy(),))
    manager.emit(7, "work", "did a thing")
    manager.emit_error("error", "it broke", ValueError("bad input"))

    error_event = captured[-1]
    assert error_event.level is LogLevel.ERROR
    assert error_event.step == 8
    assert error_event.metrics["error_type"] == "ValueError"
    assert error_event.metrics["error"] == "bad input"


def test_cli_reporter_keeps_stdout_clean_and_logs_warnings(tmp_path: Path) -> None:
    out = io.StringIO()
    err = io.StringIO()
    reporter = CliReporter("demo", run_directory=tmp_path / "cli", stream=out, error_stream=err)
    reporter.info("demo", "starting")
    reporter.warning("demo", "heads up")
    reporter.result_json({"ok": True}, sort_keys=True)
    reporter.close()

    assert out.getvalue().strip() == json.dumps({"ok": True}, sort_keys=True)
    assert "WARNING" in err.getvalue()
    assert "heads up" in err.getvalue()

    events = (tmp_path / "cli/logs/events.jsonl").read_text().splitlines()
    phases = [json.loads(line)["phase"] for line in events]
    assert "demo" in phases
    # The INFO "starting" event is captured to the log but never printed to stderr.
    assert "starting" not in err.getvalue()


def test_smoke_training_emits_error_event_on_failure(tmp_path: Path, monkeypatch) -> None:
    import ac_zero.training.smoke as smoke

    captured: list[TrainingEvent] = []

    class _Spy:
        def on_event(self, event: TrainingEvent) -> None:
            captured.append(event)

        def close(self) -> None:
            return None

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("generator exploded")

    monkeypatch.setattr(smoke, "generate_solvable", _boom)
    manager = CallbackManager((_Spy(),))

    with pytest.raises(RuntimeError):
        smoke.run_smoke_training(0, run_directory=tmp_path / "smoke", callbacks=manager)

    error_events = [event for event in captured if event.level is LogLevel.ERROR]
    assert error_events
    assert error_events[-1].metrics["error_type"] == "RuntimeError"


def test_training_pipeline_emits_error_event_on_failure(tmp_path: Path, monkeypatch) -> None:
    import ac_zero.training.pipeline as pipeline

    captured: list[TrainingEvent] = []

    class _Spy:
        def on_event(self, event: TrainingEvent) -> None:
            captured.append(event)

        def close(self) -> None:
            return None

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("episode collection failed")

    monkeypatch.setattr(pipeline, "_collect_episode", _boom)
    config = pipeline.TrainingPipelineConfig(
        iterations=1,
        episodes_per_iteration=1,
        run_directory=str(tmp_path / "train"),
    )
    manager = CallbackManager((_Spy(),))

    with pytest.raises(RuntimeError):
        pipeline.run_training_pipeline(config, seed=0, callbacks=manager)

    assert any(event.level is LogLevel.ERROR for event in captured)


def test_cli_reporter_logs_errors_with_exception(tmp_path: Path) -> None:
    err = io.StringIO()
    reporter = CliReporter(
        "demo", run_directory=tmp_path / "cli", stream=io.StringIO(), error_stream=err
    )
    reporter.error("demo", "command failed", RuntimeError("nope"))
    reporter.close()

    rows = [
        json.loads(line) for line in (tmp_path / "cli/logs/events.jsonl").read_text().splitlines()
    ]
    error_row = next(row for row in rows if row["level"] == "ERROR")
    assert error_row["metrics"]["error_type"] == "RuntimeError"
    assert "RuntimeError" in err.getvalue()
