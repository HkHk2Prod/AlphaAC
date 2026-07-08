"""Tests for deterministic checkpoint-name derivation."""

from __future__ import annotations

from dataclasses import replace

from ac_zero.training.checkpoint_name import derive_checkpoint_name
from ac_zero.training.pipeline_config import TrainingPipelineConfig


def test_readable_slug_reflects_task_and_model() -> None:
    config = TrainingPipelineConfig(
        rank=2,
        agent="ppo",
        model="residual_mlp",
        moveset="strict-ac",
        reward_mode="length_reduction_and_goal",
    )
    name = derive_checkpoint_name(config)
    assert name.startswith("rank2-ppo-residual_mlp-strict_ac-length_reduction_and_goal-")
    # The trailing hash is a short hex digest.
    assert len(name.rsplit("-", 1)[-1]) == 6


def test_same_task_config_is_stable() -> None:
    config = TrainingPipelineConfig(model="residual_mlp")
    # Operational-only differences do not change the name: same warm-start lineage.
    other = replace(
        config,
        iterations=999,
        workers=4,
        learning_rate=0.001,
        run_directory="elsewhere",
        batch_size=64,
        checkpoint_every=7,
    )
    assert derive_checkpoint_name(config) == derive_checkpoint_name(other)


def test_task_defining_change_yields_new_name() -> None:
    config = TrainingPipelineConfig(model="residual_mlp")
    base = derive_checkpoint_name(config)
    assert base != derive_checkpoint_name(replace(config, gamma=0.5))
    assert base != derive_checkpoint_name(replace(config, moveset="universal"))
    assert base != derive_checkpoint_name(replace(config, reward_mode="sparse_goal"))
