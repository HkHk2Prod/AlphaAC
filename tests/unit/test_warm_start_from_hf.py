"""The first-run-vs-follow-up precedence of `_warm_start_from_hf`.

The RL checkpoint of a task always wins once it exists -- its `latest.json`, falling
back to `best.json` for a lineage written before `latest` was published; the
supervised-pretrained checkpoint only ever seeds the task's very first run; with
neither on the bucket the run trains from scratch. The decision hinges entirely on
which download returns a path, so both network calls are stubbed to script each case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ac_zero.cli as cli
from ac_zero.models.trainable import MODEL_FORMAT_VERSION
from ac_zero.system.reporting import CliReporter
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig


def _config(tmp_path: Path, **overrides: object) -> TrainingPipelineConfig:
    settings: dict[str, object] = {
        "checkpoint_name": "rank2-ppo-transformer",
        "run_directory": str(tmp_path / "run"),
    }
    settings.update(overrides)
    return TrainingPipelineConfig(**settings)  # type: ignore[arg-type]


def _stub_downloads(
    monkeypatch,
    present: set[str],
    format_version: int = MODEL_FORMAT_VERSION,
    latest: set[str] | None = None,
) -> None:
    """Make both checkpoint downloads return a written file for the named lineages.

    `present` holds the names with a `best.json`; `latest` those with a
    `latest.json`, defaulting to the same set -- a lineage that has checkpointed at
    all has both. The file records which of the two it came from and carries a model
    state of `format_version`, which the warm start checks before accepting it.
    """
    has_latest = present if latest is None else latest

    def _fake(names: set[str], source: str):  # type: ignore[no-untyped-def]
        def fake(name, local_path, *, bucket, missing_ok):  # type: ignore[no-untyped-def]
            if name not in names:
                return None
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = {"source": source, "model_state": {"format_version": format_version}}
            dest.write_text(json.dumps(payload), encoding="utf-8")
            return dest

        return fake

    monkeypatch.setattr(cli, "download_best_checkpoint", _fake(present, "best"))
    monkeypatch.setattr(cli, "download_latest_checkpoint", _fake(has_latest, "latest"))


def _stub_archive(monkeypatch, present: set[str]) -> list[tuple[str, str]]:
    """Record archive calls and empty the archived name, as the real move does."""
    calls: list[tuple[str, str]] = []

    def fake(name, stamp, *, bucket):  # type: ignore[no-untyped-def]
        calls.append((name, stamp))
        present.discard(name)
        return 1

    monkeypatch.setattr(cli, "archive_checkpoint_lineage", fake)
    return calls


def test_start_fresh_archives_the_lineage_and_reseeds_from_pretrained(
    tmp_path: Path, monkeypatch
) -> None:
    """Even with a run checkpoint present, a fresh start falls back to pretrained."""
    config = _config(tmp_path, pretrained_checkpoint="pretrained-transformer")
    present = {"rank2-ppo-transformer", "pretrained-transformer"}
    _stub_downloads(monkeypatch, present)
    calls = _stub_archive(monkeypatch, present)

    result = cli._warm_start_from_hf(
        config, "bucket", CliReporter("train", tmp_path), start_fresh=True
    )

    assert [name for name, _ in calls] == ["rank2-ppo-transformer"]
    assert result.warm_start == str(Path(config.run_directory) / "pretrained_warm_start.json")


def test_start_fresh_on_a_scratch_task_trains_from_zero(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)  # no pretrained_checkpoint: the scratch arm of the ablation
    present = {"rank2-ppo-transformer"}
    _stub_downloads(monkeypatch, present)
    _stub_archive(monkeypatch, present)

    result = cli._warm_start_from_hf(
        config, "bucket", CliReporter("train", tmp_path), start_fresh=True
    )

    assert result.warm_start is None


def test_without_start_fresh_the_lineage_is_never_archived(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    present = {"rank2-ppo-transformer"}
    _stub_downloads(monkeypatch, present)
    calls = _stub_archive(monkeypatch, present)

    cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))

    assert calls == []


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


def test_a_stale_format_run_checkpoint_fails_the_run_at_the_pull(
    tmp_path: Path, monkeypatch
) -> None:
    """A pre-bump checkpoint cannot be loaded, so say so now, not an hour into the run."""
    config = _config(tmp_path)
    _stub_downloads(monkeypatch, {"rank2-ppo-transformer"}, format_version=1)

    with pytest.raises(ValueError, match=r"rank2-ppo-transformer.*format v1"):
        cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))


def test_a_stale_format_pretrained_checkpoint_fails_the_run_at_the_pull(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path, pretrained_checkpoint="pretrained-transformer")
    _stub_downloads(monkeypatch, {"pretrained-transformer"}, format_version=1)

    with pytest.raises(ValueError, match="supervised pretraining"):
        cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))


def test_without_a_pretrained_name_the_first_run_is_from_scratch(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)  # no pretrained_checkpoint
    _stub_downloads(monkeypatch, set())
    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))
    assert result.warm_start is None


def test_a_resumed_run_continues_from_latest_not_best(tmp_path: Path, monkeypatch) -> None:
    """`best.json` is a fixed point: a run that never beats it would reload it forever."""
    config = _config(tmp_path)
    _stub_downloads(monkeypatch, {"rank2-ppo-transformer"})

    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))

    assert result.warm_start is not None
    assert json.loads(Path(result.warm_start).read_text())["source"] == "latest"


def test_a_lineage_without_latest_falls_back_to_its_best(tmp_path: Path, monkeypatch) -> None:
    """Lineages checkpointed before `latest.json` was published still resume."""
    config = _config(tmp_path)
    _stub_downloads(monkeypatch, {"rank2-ppo-transformer"}, latest=set())

    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))

    assert result.warm_start is not None
    assert json.loads(Path(result.warm_start).read_text())["source"] == "best"


def test_the_pretrained_seed_still_comes_from_its_best(tmp_path: Path, monkeypatch) -> None:
    """The pretrained lineage publishes no RL `latest`; its best is the seed."""
    config = _config(tmp_path, pretrained_checkpoint="pretrained-transformer")
    _stub_downloads(monkeypatch, {"pretrained-transformer"})

    result = cli._warm_start_from_hf(config, "bucket", CliReporter("train", tmp_path))

    assert result.warm_start == str(Path(config.run_directory) / "pretrained_warm_start.json")
    assert json.loads(Path(result.warm_start).read_text())["source"] == "best"
