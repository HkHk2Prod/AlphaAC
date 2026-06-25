import json
from pathlib import Path

import numpy as np
import pytest

from ac_zero.cli import main
from ac_zero.training.checkpointing import CheckpointManager
from ac_zero.training.losses import masked_softmax, policy_value_loss, visit_count_policy
from ac_zero.training.pipeline import TrainingPipelineConfig, run_training_pipeline


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
