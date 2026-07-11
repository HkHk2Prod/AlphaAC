import numpy as np

from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.deepsets import DeepSetsPolicyValueModel
from ac_zero.models.gru import GRUPolicyValueModel
from ac_zero.models.registry import create_model
from ac_zero.models.residual_mlp import ResidualMLPPolicyValueModel
from ac_zero.models.transformer import TransformerPolicyValueModel
from ac_zero.training.checkpointing.checkpointing import CheckpointManager
from ac_zero.training.smoke import run_smoke_training


def test_registered_smoke_models_have_finite_outputs() -> None:
    instance = generate_solvable(rank=2, depth=1, seed=0)
    env = ACEnvironment(instance.presentation, ACEnvironmentConfig(max_moves=4))
    encoding = StateEncoder(max_relator_tokens=8).encode(env.state)
    action_count = len(env.catalog)
    expected_types = {
        "deepsets": DeepSetsPolicyValueModel,
        "gru": GRUPolicyValueModel,
        "residual_mlp": ResidualMLPPolicyValueModel,
        "transformer": TransformerPolicyValueModel,
    }
    for name, expected_type in expected_types.items():
        model = create_model(name)
        assert isinstance(model, expected_type)
        output = model.apply(encoding, action_count)
        assert output.logits.shape == (action_count,)
        assert np.isfinite(output.logits).all()
        assert np.isfinite(output.value)


def test_run_smoke_training_exercises_full_smoke_stack(tmp_path) -> None:
    summary = run_smoke_training(seed=0, run_directory=tmp_path / "smoke")
    assert summary.certificate_verified
    assert summary.checkpoint_restored
    assert summary.optimizer_updates == 1
    assert summary.replay_size == 1
    assert summary.mcts_simulations == 8
    assert summary.event_log_path.endswith("training_events.jsonl")
    assert summary.final_graph_path.endswith("final_graphs.txt")
    checkpoint = CheckpointManager(tmp_path / "smoke/checkpoints").load_json("latest")
    assert checkpoint["optimizer_state"]["step"] == 1
    events = (tmp_path / "smoke/logs/training_events.jsonl").read_text().splitlines()
    assert any('"phase": "optimizer"' in line for line in events)
    assert any('"phase": "completed"' in line for line in events)
    assert "live graphs" in (tmp_path / "smoke/artifacts/live_graphs.txt").read_text()
    assert "final training graphs" in (tmp_path / "smoke/artifacts/final_graphs.txt").read_text()
