"""Tests for deterministic checkpoint-name derivation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ac_zero.training.checkpointing.checkpoint_name import derive_checkpoint_name
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig


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


def test_suffix_splits_a_task_into_parallel_lineages() -> None:
    # The ablation twins differ only in `pretrained_checkpoint`, which the name does
    # not read -- without a suffix they would collide into one lineage.
    config = TrainingPipelineConfig(model="residual_mlp")
    collide = replace(config, pretrained_checkpoint="pretrained-rank2-residual_mlp-rel48")
    assert derive_checkpoint_name(config) == derive_checkpoint_name(collide)

    pretrained = replace(collide, checkpoint_name_suffix="pretrained")
    scratch = replace(config, checkpoint_name_suffix="scratch")
    assert derive_checkpoint_name(pretrained) != derive_checkpoint_name(scratch)
    assert derive_checkpoint_name(pretrained).startswith("rank2-alphazero-residual_mlp-")


def test_suffix_lands_before_the_hash_so_it_stays_trailing() -> None:
    name = derive_checkpoint_name(
        TrainingPipelineConfig(model="residual_mlp", checkpoint_name_suffix="scratch")
    )
    readable, digest = name.rsplit("-", 1)
    assert len(digest) == 6
    assert readable.endswith("-scratch")


def test_both_suffixed_lineages_fork_together_on_a_task_change() -> None:
    # The point of a suffix over a pinned `checkpoint_name`: a task-defining change
    # must fork *both* arms, or the ablation silently compares incompatible runs.
    base = TrainingPipelineConfig(model="residual_mlp")
    arms = [replace(base, checkpoint_name_suffix=s) for s in ("pretrained", "scratch")]
    before = [derive_checkpoint_name(c) for c in arms]
    after = [derive_checkpoint_name(replace(c, gamma=0.5)) for c in arms]
    assert all(b != a for b, a in zip(before, after, strict=True))
    assert len(set(before)) == len(set(after)) == 2


def test_suffix_is_slugified() -> None:
    name = derive_checkpoint_name(
        TrainingPipelineConfig(model="residual_mlp", checkpoint_name_suffix="From Zero/v2")
    )
    assert "from_zero_v2" in name
    assert " " not in name and "/" not in name


def test_suffix_alongside_a_pinned_name_is_rejected() -> None:
    # `checkpoint_name` bypasses derivation entirely, so a suffix would be ignored.
    with pytest.raises(ValueError, match="checkpoint_name_suffix"):
        TrainingPipelineConfig(
            checkpoint_name="pinned", checkpoint_name_suffix="scratch"
        ).validate()
