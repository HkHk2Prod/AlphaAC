"""The first-run-vs-follow-up precedence of `_warm_start_from_hf`.

The RL checkpoint of a task always wins once it exists; the supervised-pretrained
checkpoint only ever seeds the task's very first run; with neither on the bucket the run
trains from scratch. The decision hinges entirely on which `download_best_checkpoint`
returns a path, so the network call is stubbed to script each case.
"""

from __future__ import annotations

from pathlib import Path

import ac_zero.cli as cli
from ac_zero.system.reporting import CliReporter
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig


def _config(tmp_path: Path, **overrides: object) -> TrainingPipelineConfig:
    settings: dict[str, object] = {
        "checkpoint_name": "rank2-ppo-transformer",
        "run_directory": str(tmp_path / "run"),
    }
    settings.update(overrides)
    return TrainingPipelineConfig(**settings)  # type: ignore[arg-type]


def _stub_downloads(monkeypatch, present: set[str]) -> None:
    """Make `download_best_checkpoint` return a written file for names in `present`."""

    def fake(name, local_path, *, bucket, missing_ok):  # type: ignore[no-untyped-def]
        if name not in present:
            return None
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}", encoding="utf-8")
        return dest

    monkeypatch.setattr(cli, "download_best_checkpoint", fake)


def test_the_run_checkpoint_wins_when_it_exists(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, pretrained_checkpoint="pretrained-transformer")
    _stub_downloads(monkeypatch, {"rank2-ppo-transformer", "pretrained-transformer"})
    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))
    assert result.warm_start == str(Path(config.run_directory) / "warm_start.json")


def test_the_pretrained_checkpoint_seeds_the_first_run(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, pretrained_checkpoint="pretrained-transformer")
    _stub_downloads(monkeypatch, {"pretrained-transformer"})  # no run checkpoint yet
    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))
    assert result.warm_start == str(Path(config.run_directory) / "pretrained_warm_start.json")


def test_no_checkpoint_anywhere_trains_from_scratch(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, pretrained_checkpoint="pretrained-transformer")
    _stub_downloads(monkeypatch, set())
    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))
    assert result.warm_start is None


def test_without_a_pretrained_name_the_first_run_is_from_scratch(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)  # no pretrained_checkpoint
    _stub_downloads(monkeypatch, set())
    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))
    assert result.warm_start is None
