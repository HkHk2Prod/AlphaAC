"""Tests for the batched model path: batching must not change what a state predicts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.batch import encode_batch
from ac_zero.models.features import global_features, relator_features, vocabulary_size
from ac_zero.models.registry import create_trainable_model

ARCHITECTURES = ("linear_policy_value", "residual_mlp", "deepsets", "gru", "transformer")


def _encodings(count: int) -> list[PaddedEncoding]:
    encoder = StateEncoder(16)
    states = []
    for seed in range(count):
        pres = generate_solvable(2, 3, seed).presentation
        env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=8), encoder)
        states.append(encoder.encode(env.state))
    return states


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_a_batched_forward_matches_the_single_state_apply(name: str) -> None:
    """Row `i` of a batch predicts exactly what state `i` predicts on its own.

    This is the contract the whole refactor rests on: search evaluates states one at a
    time and supervised training evaluates them a thousand at a time, and both must be
    reading the same model.
    """
    encodings = _encodings(6)
    model = create_trainable_model(name, seed=0)
    singles = [model.apply(encoding, 12) for encoding in encodings]

    with torch.no_grad():
        logits, values = model.forward(model.encode(encodings), 12)

    for index, single in enumerate(singles):
        assert np.allclose(logits[index].numpy(), single.logits, atol=1e-5)
        assert float(values[index]) == pytest.approx(single.value, abs=1e-5)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_a_batch_trains_and_stays_finite(name: str) -> None:
    encodings = _encodings(4)
    model = create_trainable_model(name, seed=0)
    batch = model.encode(encodings)
    logits, values = model.forward(batch, 12)

    assert logits.shape == (4, 12)
    assert values.shape == (4,)
    assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(values).all())
    assert bool((values.abs() <= 1.0).all())  # the tanh-bounded value head
    assert model.parameter_count > 0


def test_deepsets_stays_permutation_invariant_when_batched() -> None:
    encoder = StateEncoder(16)
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2, -1, -1]])
    swapped = BalancedPresentation.from_letters(2, [[2, -1, -1], [1, 2]])
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=8), encoder)
    other = ACEnvironment(swapped, ACEnvironmentConfig(max_moves=8), encoder)

    model = create_trainable_model("deepsets", seed=0)
    with torch.no_grad():
        logits, _ = model.forward(
            model.encode([encoder.encode(env.state), encoder.encode(other.state)]), 12
        )
    assert np.allclose(logits[0].numpy(), logits[1].numpy(), atol=1e-6)


def test_the_transformer_multi_head_split_requires_a_divisible_width() -> None:
    model = create_trainable_model("transformer", seed=0, embed_dim=12, num_heads=5)
    with pytest.raises(ValueError, match="not divisible by num_heads"):
        model.apply(_encodings(1)[0], 12)


def test_batched_features_reproduce_the_per_state_definitions() -> None:
    """The vectorized feature blocks agree with the aggregate statistics they replaced."""
    encodings = _encodings(3)
    tokens = np.stack([e.tokens for e in encodings])
    mask = np.stack([e.mask for e in encodings])
    scalars = np.stack([e.scalar_features for e in encodings])

    globals_ = global_features(tokens, mask, scalars)
    relators = relator_features(tokens, mask)
    assert globals_.shape == (3, 8)
    assert relators.shape == (3, 2, 6)

    for index, encoding in enumerate(encodings):
        real = encoding.tokens[encoding.mask].astype(np.float64)
        slots = encoding.tokens.shape[1]
        assert globals_[index][0] == 1.0  # the bias term
        assert globals_[index][6] == pytest.approx(float(np.mean(real)) / 10.0)
        assert globals_[index][7] == pytest.approx(float(np.std(real)) / 10.0)
        rows = zip(encoding.tokens, encoding.mask, strict=True)
        for row, (relator, relator_mask) in enumerate(rows):
            letters = relator[relator_mask].astype(np.float64)
            assert relators[index][row][1] == pytest.approx(letters.size / slots)
            assert relators[index][row][2] == pytest.approx(float(np.mean(letters)) / 10.0)


def test_an_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        encode_batch([], torch.device("cpu"))


def test_vocabulary_covers_every_signed_generator_plus_padding() -> None:
    assert vocabulary_size(2) == 6  # padding, +-1, +-2, and the exclusive upper bound
    assert vocabulary_size(3) == 8


def test_the_encoder_refuses_to_truncate_a_relator() -> None:
    """A truncated relator is a different presentation, so it must fail, not silently pass."""
    encoder = StateEncoder(4)
    long_relator = BalancedPresentation.from_letters(2, [[1, 2, 1, 2, 1], [2]])
    env = ACEnvironment(long_relator, ACEnvironmentConfig(max_moves=8), encoder)
    with pytest.raises(ValueError, match="exceeds the encoder capacity"):
        encoder.encode(env.state)


def test_the_environment_masks_out_moves_that_would_not_fit_the_encoder() -> None:
    """The episode never reaches a state the encoder would have to refuse."""
    encoder = StateEncoder(6)
    pres = BalancedPresentation.from_letters(2, [[1, 2, 1, 2, 1, 2], [2, 1]])
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=8), encoder)

    for action, legal in enumerate(env.legal_action_mask()):
        result = env.catalog.move(action).apply(env.state.presentation)
        fits = all(len(relator.letters) <= 6 for relator in result.relators)
        if not fits:
            assert not legal
