import json
from pathlib import Path

import numpy as np
import pytest

from ac_zero.cli import main
from ac_zero.training.callbacks import CallbackManager
from ac_zero.training.checkpointing import CheckpointManager
from ac_zero.training.events import TrainingEvent
from ac_zero.training.losses import masked_softmax, policy_value_loss, visit_count_policy
from ac_zero.training.pipeline import TrainingPipelineConfig, run_training_pipeline


class _CapturingSink:
    """Collect every emitted event so a test can inspect the run log."""

    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


def test_training_pipeline_opens_with_full_task_description(tmp_path: Path) -> None:
    config = TrainingPipelineConfig(
        scramble_depth=1,
        max_moves=4,
        model="residual_mlp",
        mcts_simulations=4,
        iterations=1,
        episodes_per_iteration=2,
        optimizer_updates=2,
        batch_size=2,
        workers=2,
        run_directory=str(tmp_path / "train"),
    )
    sink = _CapturingSink()
    run_training_pipeline(config, seed=7, callbacks=CallbackManager((sink,)))

    start = sink.events[0]
    assert start.message == "starting training pipeline"
    # Every parameter that shapes the trained model is named in the opening event.
    assert start.metrics["seed"] == 7
    assert start.metrics["rank"] == config.rank
    assert start.metrics["iterations"] == config.iterations
    assert start.metrics["episodes_per_iteration"] == config.episodes_per_iteration
    assert start.metrics["learning_rate"] == config.learning_rate
    assert start.metrics["run_directory"] == config.run_directory
    # The run then reports whether it fans self-play out across worker processes.
    worker_event = sink.events[1]
    assert worker_event.metrics["parallel"] is True
    assert worker_event.metrics["workers"] == 2


def test_config_exposes_c_puct_for_harder_runs() -> None:
    config = TrainingPipelineConfig.from_mapping(
        {"training": {"c_puct": 2.5, "mcts_simulations": 64}}
    )
    assert config.c_puct == 2.5
    assert config.mcts_simulations == 64
    assert TrainingPipelineConfig().c_puct == 1.5  # default
    with pytest.raises(ValueError, match="c_puct must be positive"):
        TrainingPipelineConfig(c_puct=0.0).validate()


def test_visit_policy_and_masked_loss_ignore_illegal_actions() -> None:
    mask = (True, False, True)
    target = visit_count_policy((3, 99, 1), mask)
    assert np.allclose(target, np.asarray([0.75, 0.0, 0.25]))

    probs = masked_softmax(np.asarray([2.0, 100.0, 0.0]), mask)
    assert probs[1] == 0.0
    assert np.isclose(float(np.sum(probs)), 1.0)

    loss = policy_value_loss(np.asarray([2.0, 100.0, 0.0]), 0.25, target, 0.5, mask)
    assert loss.policy_loss > 0.0
    assert loss.value_loss == 0.0625
    assert loss.total_loss > loss.value_loss


def test_training_pipeline_writes_checkpoint_and_summary(tmp_path: Path) -> None:
    config = TrainingPipelineConfig(
        scramble_depth=1,
        max_moves=4,
        model="residual_mlp",
        mcts_simulations=4,
        iterations=1,
        episodes_per_iteration=2,
        optimizer_updates=2,
        batch_size=2,
        run_directory=str(tmp_path / "train"),
    )
    summary = run_training_pipeline(config, seed=7)

    assert summary.checkpoint_restored
    assert summary.certificate_verified
    assert summary.model_name == "residual_mlp"
    assert summary.optimizer_updates == 2
    assert summary.replay_size > 0
    assert Path(summary.event_log_path).exists()
    assert Path(summary.final_graph_path).exists()
    checkpoint = CheckpointManager(tmp_path / "train/checkpoints").load_json("latest")
    assert checkpoint["schema_version"] == "aczero-training-checkpoint-v1"
    assert checkpoint["optimizer_state"]["step"] == 2
    assert checkpoint["model_state"]["architecture"] == "residual_mlp"
    summary_json = json.loads((tmp_path / "train/artifacts/training_summary.json").read_text())
    assert summary_json["optimizer_updates"] == 2
    # Training-progress plots are rendered and recorded on the summary.
    assert summary.plot_paths
    for plot in summary.plot_paths:
        assert Path(plot).exists()
        assert Path(plot).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_training_pipeline_model_is_invariant_to_worker_count(tmp_path: Path) -> None:
    def _run(workers: int, name: str) -> dict:
        config = TrainingPipelineConfig(
            scramble_depth=1,
            max_moves=4,
            model="residual_mlp",
            mcts_simulations=4,
            iterations=2,
            episodes_per_iteration=3,
            optimizer_updates=2,
            batch_size=2,
            workers=workers,
            run_directory=str(tmp_path / name),
        )
        run_training_pipeline(config, seed=11)
        return CheckpointManager(tmp_path / name / "checkpoints").load_json("latest")["model_state"]

    # Self-play episodes run in order, so multi-process self-play trains the same
    # model as the single-process run. Under PyTorch the weights match within
    # floating-point tolerance rather than bit-for-bit.
    sequential = _run(1, "seq")
    parallel = _run(2, "par")
    assert sequential.keys() == parallel.keys()
    for key in sequential:
        if key == "parameters":
            continue
        assert sequential[key] == parallel[key]
    seq_params, par_params = sequential["parameters"], parallel["parameters"]
    assert seq_params.keys() == par_params.keys()
    for name, seq_weight in seq_params.items():
        assert np.allclose(seq_weight, par_params[name], rtol=1e-5, atol=1e-6)


def test_config_reads_worker_count_from_mapping() -> None:
    assert TrainingPipelineConfig.from_mapping({"training": {"workers": 4}}).workers == 4
    # The default autodetects: 0 means "use every CPU core".
    assert TrainingPipelineConfig().workers == 0
    assert TrainingPipelineConfig.from_mapping({"training": {"workers": 1}}).workers == 1


def test_config_reads_dataset_seeding_from_mapping() -> None:
    config = TrainingPipelineConfig.from_mapping(
        {"dataset": {"path": "data/train_rank2.json", "max_difficulty": 5, "bucket": "ns/bucket"}}
    )
    assert config.dataset_path == "data/train_rank2.json"
    assert config.dataset_max_difficulty == 5
    assert config.dataset_bucket == "ns/bucket"
    # Absent by default: self-play falls back to random scrambles.
    assert TrainingPipelineConfig().dataset_path is None
    assert TrainingPipelineConfig().dataset_max_difficulty is None


def test_config_rejects_negative_max_difficulty() -> None:
    with pytest.raises(ValueError):
        TrainingPipelineConfig(dataset_max_difficulty=-1).validate()


def test_config_reads_moveset_from_mapping() -> None:
    assert TrainingPipelineConfig().moveset == "strict-ac"
    assert TrainingPipelineConfig.from_mapping({"moveset": "universal"}).moveset == "universal"


def test_config_rejects_unknown_moveset() -> None:
    with pytest.raises(ValueError, match="moveset"):
        TrainingPipelineConfig(moveset="nope").validate()


def test_config_requires_dataset_for_descent_reward() -> None:
    with pytest.raises(ValueError, match="descent"):
        TrainingPipelineConfig(reward_mode="descent").validate()
    # A descent run needs both the group dataset and its annotations.
    with pytest.raises(ValueError, match="descent"):
        TrainingPipelineConfig(
            reward_mode="descent", dataset_path="data/train.groups.json"
        ).validate()
    TrainingPipelineConfig(
        reward_mode="descent",
        dataset_path="data/train.groups.json",
        dataset_annotations_path="data/train.strict-ac.annotations.json",
    ).validate()


def test_ensure_training_dataset_pulls_only_when_missing(monkeypatch, tmp_path: Path) -> None:
    from ac_zero.cli import _ensure_training_dataset
    from ac_zero.system.reporting import CliReporter

    calls: list[tuple[str, str]] = []

    def _fake_download(local, *, remote_name=None, bucket, missing_ok=False):  # type: ignore[no-untyped-def]
        calls.append((str(local), bucket))
        Path(local).write_text("{}", encoding="utf-8")
        return Path(local)

    monkeypatch.setattr("ac_zero.cli.download_dataset", _fake_download)
    reporter = CliReporter("train", run_directory=str(tmp_path / "logs"))
    dataset = tmp_path / "train_rank2.json"

    # No dataset_path configured: nothing is pulled.
    _ensure_training_dataset(TrainingPipelineConfig(), reporter)
    assert calls == []

    # Configured but missing: pulled from the configured bucket.
    config = TrainingPipelineConfig(dataset_path=str(dataset), dataset_bucket="ns/bucket")
    _ensure_training_dataset(config, reporter)
    assert calls == [(str(dataset), "ns/bucket")]

    # Already on disk: not pulled again.
    _ensure_training_dataset(config, reporter)
    assert len(calls) == 1
    reporter.close()


def test_cli_train_uses_configured_pipeline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "rank: 2",
                "max_moves: 4",
                "model: linear_policy_value",
                "dataset:",
                "  count: 1",
                "  depth: 1",
                "training:",
                "  iterations: 1",
                "  episodes_per_iteration: 1",
                "  optimizer_updates: 1",
                "  batch_size: 1",
                "  mcts_simulations: 2",
                "  run_directory: runs/train/test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config_path), "--seed", "3"]) == 0
    checkpoint = CheckpointManager(tmp_path / "runs/train/test/checkpoints").load_json("latest")
    assert checkpoint["schema_version"] == "aczero-training-checkpoint-v1"
    assert checkpoint["optimizer_state"]["step"] == 1
    assert (tmp_path / "runs/train/test/artifacts/training_summary.json").exists()
    # The command presents the rendered plots: they exist on disk.
    artifacts = tmp_path / "runs/train/test/artifacts"
    assert (artifacts / "loss_curves.png").exists()
    assert (artifacts / "selfplay_progress.png").exists()


def test_present_plots_reports_paths_without_opening_when_headless(monkeypatch) -> None:
    from ac_zero import cli

    opened: list[str] = []
    reported: list[tuple[str, dict]] = []
    monkeypatch.setattr(cli, "_open_in_viewer", lambda path: opened.append(path))
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)

    class _Reporter:
        def progress(self, phase: str, message: str, metrics: dict) -> None:
            reported.append((message, metrics))

    cli._present_plots(["a.png", "b.png"], _Reporter())  # type: ignore[arg-type]

    # Both plots are reported, but no viewer is launched without an interactive tty.
    assert [metrics["path"] for _, metrics in reported] == ["a.png", "b.png"]
    assert opened == []
