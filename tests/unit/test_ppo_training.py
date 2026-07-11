from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from ac_zero.cli import main
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.registry import create_trainable_model
from ac_zero.training.checkpointing.checkpointing import CheckpointManager
from ac_zero.training.navigation.navigation_curriculum import DistanceCurriculumConfig
from ac_zero.training.pipeline.instance_source import build_instance_source
from ac_zero.training.pipeline.pipeline import run_training_pipeline
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
from ac_zero.training.pipeline.pipeline_episodes import EpisodeMetrics
from ac_zero.training.ppo.losses import masked_softmax
from ac_zero.training.ppo.ppo import (
    PPOExample,
    PPOTrainer,
    _generalized_advantages,
    _Rollout,
    _Transition,
    collect_rollouts,
)


def _transition(reward: float, value: float) -> _Transition:
    return _Transition(
        encoding=None,  # type: ignore[arg-type]
        legal_mask=(),
        action=0,
        log_prob=0.0,
        reward=reward,
        value=value,
    )


def _rollout(transitions: list[_Transition], bootstrap: float) -> _Rollout:
    metrics = EpisodeMetrics(0.0, 0.0, False, len(transitions))
    return _Rollout(transitions, bootstrap, metrics)


def _flatten(pairs: list[tuple[float, float]]) -> list[float]:
    return [value for pair in pairs for value in pair]


def _ppo_config(**overrides: object) -> TrainingPipelineConfig:
    base = dict(
        agent="ppo",
        scramble_depth=2,
        # Scrambles carry no distance, so self-play uses the unknown-distance
        # fallback horizon; cap it small here to keep the test's episodes short.
        curriculum_config=DistanceCurriculumConfig(unknown_distance_max_moves=6),
        model="residual_mlp",
        iterations=1,
        episodes_per_iteration=4,
        batch_size=4,
        ppo_epochs=2,
        learning_rate=0.02,
        workers=1,
    )
    base.update(overrides)
    return TrainingPipelineConfig(**base)  # type: ignore[arg-type]


def _model_and_state() -> tuple[object, ACEnvironment, np.ndarray, tuple[bool, ...], int, float]:
    """A built model plus one state's encoding, mask, greedy action, and log-prob."""
    model = create_trainable_model("residual_mlp", seed=0)
    encoder = StateEncoder()
    env = ACEnvironment(generate_solvable(2, 2, 0).presentation, ACEnvironmentConfig(max_moves=6))
    encoding = encoder.encode(env.state)
    mask = env.legal_action_mask()
    output = model.apply(encoding, len(mask))
    probs = masked_softmax(output.logits, mask)
    action = int(np.argmax(probs))
    return model, encoding, mask, action, float(np.log(probs[action])), float(output.value)


def _log_prob(model: object, encoding: object, mask: tuple[bool, ...], action: int) -> float:
    probs = masked_softmax(model.apply(encoding, len(mask)).logits, mask)  # type: ignore[attr-defined]
    return float(np.log(probs[action]))


def test_generalized_advantages_matches_manual_gae() -> None:
    rollout = _rollout([_transition(1.0, 0.5), _transition(0.0, 0.2)], bootstrap=0.0)

    undiscounted = _generalized_advantages(rollout, gamma=1.0, gae_lambda=1.0)
    assert _flatten(undiscounted) == pytest.approx([0.5, 1.0, -0.2, 0.0])

    discounted = _generalized_advantages(rollout, gamma=0.9, gae_lambda=0.5)
    assert _flatten(discounted) == pytest.approx([0.59, 1.09, -0.2, 0.0])


def test_collect_rollouts_normalizes_advantages_and_records_legal_actions() -> None:
    config = _ppo_config()
    examples, episodes = collect_rollouts(
        config,
        StateEncoder(config.max_relator_tokens),
        create_trainable_model("residual_mlp"),
        1,
        0,
        build_instance_source(config),
    )
    assert len(episodes) == config.episodes_per_iteration
    assert examples
    advantages = np.asarray([example.advantage for example in examples])
    assert abs(float(advantages.mean())) < 1e-6  # standardized to zero mean
    for example in examples:
        assert example.old_log_prob <= 1e-9  # a log-probability is non-positive
        assert example.legal_mask[example.action]  # only legal actions are sampled


def test_ppo_update_moves_action_probability_with_advantage_sign() -> None:
    model, encoding, mask, action, old_log_prob, _ = _model_and_state()
    example = PPOExample(encoding, mask, action, old_log_prob, advantage=1.0, return_target=0.0)  # type: ignore[arg-type]
    model.ppo_update(  # type: ignore[attr-defined]
        [example], learning_rate=0.5, clip_ratio=0.2, value_weight=0.0, entropy_weight=0.0
    )
    assert _log_prob(model, encoding, mask, action) > old_log_prob

    model, encoding, mask, action, old_log_prob, _ = _model_and_state()
    example = PPOExample(encoding, mask, action, old_log_prob, advantage=-1.0, return_target=0.0)  # type: ignore[arg-type]
    model.ppo_update(  # type: ignore[attr-defined]
        [example], learning_rate=0.5, clip_ratio=0.2, value_weight=0.0, entropy_weight=0.0
    )
    assert _log_prob(model, encoding, mask, action) < old_log_prob


def test_ppo_update_clipping_blocks_gradient_outside_the_trust_region() -> None:
    model, encoding, mask, action, old_log_prob, value = _model_and_state()
    # An old log-prob far below the current one makes the ratio exp(5) >> 1 + clip.
    # With a positive advantage the surrogate is clipped, so no policy gradient
    # flows; a matched return target zeroes the value gradient too, leaving the
    # action's probability unchanged.
    example = PPOExample(encoding, mask, action, old_log_prob - 5.0, 1.0, value)  # type: ignore[arg-type]
    model.ppo_update(  # type: ignore[attr-defined]
        [example], learning_rate=0.5, clip_ratio=0.2, value_weight=1.0, entropy_weight=0.0
    )
    assert _log_prob(model, encoding, mask, action) == pytest.approx(old_log_prob, abs=1e-6)


def test_ppo_update_rejects_an_empty_batch() -> None:
    model = create_trainable_model("residual_mlp", seed=0)
    with pytest.raises(ValueError, match="batch must not be empty"):
        model.ppo_update(
            [], learning_rate=0.1, clip_ratio=0.2, value_weight=0.5, entropy_weight=0.0
        )


def test_trainer_iteration_updates_the_model_and_reports_finite_stats() -> None:
    config = _ppo_config()
    model = create_trainable_model("residual_mlp", seed=0)
    # Force the lazy build so the "before" snapshot has real weights to compare.
    env = ACEnvironment(generate_solvable(2, 2, 0).presentation, ACEnvironmentConfig(max_moves=6))
    model.apply(StateEncoder().encode(env.state), len(env.legal_action_mask()))
    before = model.to_json()["parameters"]
    trainer = PPOTrainer(
        config, StateEncoder(config.max_relator_tokens), build_instance_source(config)
    )
    result = trainer.run_iteration(model, seed=5, iteration=1, rng=random.Random(5))

    assert len(result.episodes) == config.episodes_per_iteration
    assert result.updates
    for stats in result.updates:
        components = [stats.policy_loss, stats.value_loss, stats.entropy, stats.total_loss]
        assert np.isfinite(components).all()
        assert 0.0 <= stats.clip_fraction <= 1.0
    after = model.to_json()["parameters"]
    assert any(not np.allclose(before[name], after[name]) for name in before)


def test_pipeline_ppo_backend_trains_certifies_and_stays_on_policy(tmp_path: Path) -> None:
    config = _ppo_config(iterations=2, run_directory=str(tmp_path / "ppo"))
    summary = run_training_pipeline(config, seed=7)

    assert summary.optimizer_updates > 0
    assert summary.replay_size == 0  # PPO is on-policy: nothing is buffered
    assert summary.checkpoint_restored
    assert summary.certificate_verified
    checkpoint = CheckpointManager(tmp_path / "ppo" / "checkpoints").load_json("latest")
    assert checkpoint["optimizer_state"]["step"] == summary.optimizer_updates
    assert checkpoint["config"]["agent"] == "ppo"
    assert summary.plot_paths
    for plot in summary.plot_paths:
        assert Path(plot).exists()


def test_pipeline_ppo_backend_is_invariant_to_worker_count(tmp_path: Path) -> None:
    def _run(workers: int, name: str) -> dict:
        config = _ppo_config(
            iterations=2,
            episodes_per_iteration=3,
            workers=workers,
            run_directory=str(tmp_path / name),
        )
        run_training_pipeline(config, seed=11)
        return CheckpointManager(tmp_path / name / "checkpoints").load_json("latest")["model_state"]

    sequential = _run(1, "seq")["parameters"]
    parallel = _run(2, "par")["parameters"]
    assert sequential.keys() == parallel.keys()
    for name, weight in sequential.items():
        assert np.allclose(weight, parallel[name], rtol=1e-5, atol=1e-6)


def test_cli_train_selects_the_ppo_backend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "ppo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "rank: 2",
                "agent: ppo",
                "model: residual_mlp",
                "dataset:",
                "  count: 3",
                "  depth: 2",
                "training:",
                "  curriculum:",
                "    unknown_distance_max_moves: 6",
                "  iterations: 1",
                "  episodes_per_iteration: 3",
                "  batch_size: 3",
                "  ppo_epochs: 2",
                "  run_directory: runs/ppo",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    argv = ["train", "--config", str(config_path), "--seed", "3"]
    argv += ["--workers", "1", "--self-generated"]
    assert main(argv) == 0
    checkpoint = CheckpointManager(tmp_path / "runs/ppo/checkpoints").load_json("latest")
    assert checkpoint["config"]["agent"] == "ppo"
    assert checkpoint["optimizer_state"]["step"] > 0


def test_config_reads_and_validates_ppo_settings() -> None:
    config = TrainingPipelineConfig.from_mapping(
        {"agent": "ppo", "training": {"ppo_clip": 0.3, "gamma": 0.95, "entropy_coef": 0.02}}
    )
    assert config.agent == "ppo"
    assert config.ppo_clip == 0.3
    assert config.gamma == 0.95
    assert config.entropy_coef == 0.02

    with pytest.raises(ValueError, match="agent must be"):
        TrainingPipelineConfig(agent="dqn").validate()
    with pytest.raises(ValueError, match="ppo_clip must be positive"):
        TrainingPipelineConfig(ppo_clip=0.0).validate()
    with pytest.raises(ValueError, match="gamma must be in"):
        TrainingPipelineConfig(gamma=1.5).validate()
