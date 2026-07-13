from __future__ import annotations

import torch

from ac_zero.models.registry import create_trainable_model
from ac_zero.models.torch_utils import use_single_torch_thread
from ac_zero.training.pipeline import pipeline_episodes
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
from ac_zero.training.ppo import ppo


def test_use_single_torch_thread_pins_the_intra_op_pool() -> None:
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, original))
        use_single_torch_thread()
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def _config() -> TrainingPipelineConfig:
    return TrainingPipelineConfig(rank=2, model="residual_mlp", moveset="strict-ac")


def test_episode_worker_initializer_pins_torch_to_one_thread() -> None:
    """Each self-play worker is its own process, so it must pin its own pool.

    Without this every worker spawns a full intra-op pool and the run oversubscribes
    the machine by a factor of the core count.
    """
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, original))
        model_state = create_trainable_model("residual_mlp", seed=0).to_json()
        pipeline_episodes._init_episode_worker(_config(), model_state, None, None)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def test_rollout_worker_initializer_pins_torch_to_one_thread() -> None:
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, original))
        model_state = create_trainable_model("residual_mlp", seed=0).to_json()
        ppo._init_rollout_worker(_config(), model_state, None, None)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)
