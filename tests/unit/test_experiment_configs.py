from pathlib import Path

import pytest
import yaml

from ac_zero.training.pipeline.pipeline import TrainingPipelineConfig

CONFIG_DIR = Path("configs/experiments")


def _load(name: str) -> TrainingPipelineConfig:
    data = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    return TrainingPipelineConfig.from_mapping(data)


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_experiment_config_parses(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert TrainingPipelineConfig.from_mapping(data).rank > 0


def test_alphazero_14m_matches_the_checkpoint_it_warm_starts_from() -> None:
    """The fine-tune must share the pretrained model's shape or the weights cannot load."""
    pretrain = _load("supervised_rel48_14m.yaml")
    finetune = _load("alphazero_rank2_14m.yaml")
    assert finetune.model == pretrain.model
    assert finetune.model_config == pretrain.model_config
    assert finetune.max_relator_tokens == pretrain.max_relator_tokens
    assert finetune.rank == pretrain.rank
    assert finetune.moveset == pretrain.moveset
    assert finetune.warm_start == f"{pretrain.run_directory}/model_checkpoint/best.json"
