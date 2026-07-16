from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.registry import (
    create_model,
    create_trainable_model,
    model_from_json,
)
from ac_zero.training.ppo.losses import visit_count_policy

ARCHITECTURES = ["linear_policy_value", "residual_mlp", "deepsets", "gru", "transformer"]


@dataclass
class _Example:
    encoding: PaddedEncoding
    legal_mask: tuple[bool, ...]
    policy_target: Any
    value_target: float


def _fixture(seed: int = 1, depth: int = 2) -> tuple[PaddedEncoding, tuple[bool, ...], int, Any]:
    instance = generate_solvable(rank=2, depth=depth, seed=seed)
    env = ACEnvironment(instance.presentation, ACEnvironmentConfig(max_moves=4))
    encoding = StateEncoder(max_relator_tokens=16).encode(env.state)
    mask = env.legal_action_mask()
    counts = tuple(3 if ok else 0 for ok in mask)
    target = visit_count_policy(counts, mask)
    return encoding, mask, len(env.catalog), target


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_apply_returns_finite_outputs(name: str) -> None:
    encoding, _, action_count, _ = _fixture()
    output = create_trainable_model(name, seed=0).apply(encoding, action_count)
    assert output.logits.shape == (action_count,)
    assert np.isfinite(output.logits).all()
    assert np.isfinite(output.value)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_training_reduces_loss(name: str) -> None:
    encoding, mask, _, target = _fixture()
    batch = [_Example(encoding, mask, target, 0.5)] * 4
    model = create_trainable_model(name, seed=0)
    first = model.train_batch(batch, learning_rate=0.1, value_loss_weight=1.0)
    for _ in range(25):
        last = model.train_batch(batch, learning_rate=0.1, value_loss_weight=1.0)
    assert last.total_loss < first.total_loss


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_checkpoint_round_trip_is_exact(name: str) -> None:
    encoding, mask, action_count, target = _fixture()
    model = create_trainable_model(name, seed=0)
    example = _Example(encoding, mask, target, 0.5)
    model.train_batch([example], learning_rate=0.1, value_loss_weight=1.0)
    restored = model_from_json(model.to_json())
    before = model.apply(encoding, action_count)
    after = restored.apply(encoding, action_count)
    assert np.allclose(before.logits, after.logits)
    assert before.value == pytest.approx(after.value)


def test_transformer_grad_checkpointing_matches_plain_path() -> None:
    """Checkpointing recomputes activations; the logits, values, and grads are unchanged."""
    encoding, _, action_count, _ = _fixture()
    encodings = [encoding] * 4
    outputs = []
    first_grads = []
    for grad_checkpoint in (0, 1):
        model = create_trainable_model("transformer", seed=0, grad_checkpoint=grad_checkpoint)
        logits, values = model.forward(model.encode(encodings), action_count)
        (logits.pow(2).mean() + values.pow(2).mean()).backward()
        params = model.parameters()
        outputs.append((logits.detach().numpy(), values.detach().numpy()))
        first_grads.append(params[0].grad.numpy().copy())

    assert model._net.trunk._grad_checkpoint is True  # the flag reached the trunk
    assert np.allclose(outputs[0][0], outputs[1][0], atol=1e-6)
    assert np.allclose(outputs[0][1], outputs[1][1], atol=1e-6)
    assert np.allclose(first_grads[0], first_grads[1], atol=1e-6)


def test_transformer_grad_checkpoint_flag_survives_serialization() -> None:
    encoding, mask, _, target = _fixture()
    model = create_trainable_model("transformer", seed=0, grad_checkpoint=1)
    example = _Example(encoding, mask, target, 0.5)
    model.train_batch([example], learning_rate=0.1, value_loss_weight=1.0)
    restored = model_from_json(model.to_json())
    assert restored.to_json()["hyperparameters"]["grad_checkpoint"] == 1


def test_deepsets_is_permutation_invariant() -> None:
    instance = generate_solvable(rank=2, depth=2, seed=4)
    presentation = instance.presentation
    swapped = BalancedPresentation(
        relators=tuple(reversed(presentation.relators)),
        rank=presentation.rank,
        provenance=presentation.provenance,
    )
    encoder = StateEncoder(max_relator_tokens=16)
    model = create_trainable_model("deepsets", seed=0)
    config = ACEnvironmentConfig(max_moves=4)
    base = ACEnvironment(presentation, config)
    permuted = ACEnvironment(swapped, config)
    action_count = len(base.catalog)
    original = model.apply(encoder.encode(base.state), action_count)
    reordered = model.apply(encoder.encode(permuted.state), action_count)
    assert original.value == pytest.approx(reordered.value)
    assert np.allclose(original.logits, reordered.logits)


def test_build_rejects_changing_action_count() -> None:
    encoding, _, action_count, _ = _fixture()
    model = create_trainable_model("residual_mlp", seed=0)
    model.apply(encoding, action_count)
    with pytest.raises(ValueError, match="action_count"):
        model.apply(encoding, action_count + 1)


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(KeyError):
        create_trainable_model("nonexistent")
    assert create_model("uniform").apply(_fixture()[0], _fixture()[2]).value == 0.0
