from __future__ import annotations

from typing import Any

from ac_zero.models.base import PolicyValueModel, UniformPolicyValueModel
from ac_zero.models.deepsets import DeepSetsPolicyValueModel
from ac_zero.models.gru import GRUPolicyValueModel
from ac_zero.models.linear import LinearPolicyValueModel
from ac_zero.models.residual_mlp import ResidualMLPPolicyValueModel
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.models.transformer import TransformerPolicyValueModel

_TRAINABLE: dict[str, type[TrainablePolicyValueModel]] = {
    "linear_policy_value": LinearPolicyValueModel,
    "residual_mlp": ResidualMLPPolicyValueModel,
    "deepsets": DeepSetsPolicyValueModel,
    "gru": GRUPolicyValueModel,
    "transformer": TransformerPolicyValueModel,
}


def create_trainable_model(
    name: str, *, seed: int = 0, device: str = "cpu", **hyperparameters: Any
) -> TrainablePolicyValueModel:
    """Construct a trainable policy-value model from a stable configuration name.

    Extra keyword arguments override the architecture's size hyperparameters (e.g.
    ``embed_dim``, ``hidden_dim``, ``num_layers``); unknown keys for the chosen
    architecture raise ``TypeError`` from its constructor. ``device`` is where the
    network's parameters live -- ``"auto"`` takes a GPU when one is present.
    """
    normalized = name.lower().replace("-", "_")
    try:
        factory = _TRAINABLE[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown model {name!r}") from exc
    return factory(seed=seed, device=device, **hyperparameters)


def create_model(
    name: str, *, seed: int = 0, device: str = "cpu", **hyperparameters: Any
) -> PolicyValueModel:
    """Construct any registered policy-value model, including the uniform baseline."""
    normalized = name.lower().replace("-", "_")
    if normalized == "uniform":
        return UniformPolicyValueModel()
    return create_trainable_model(normalized, seed=seed, device=device, **hyperparameters)


def model_from_json(data: dict[str, Any], *, device: str = "cpu") -> TrainablePolicyValueModel:
    """Reconstruct a trainable model and its weights from a checkpoint payload.

    ``device`` is not part of the payload: where a checkpoint was trained does not
    constrain where it is replayed, so a GPU-trained model loads onto a CPU box.
    """
    hyperparameters = dict(data.get("hyperparameters", {}))
    seed = int(hyperparameters.pop("seed", 0))
    model = _TRAINABLE[data["architecture"]](seed=seed, device=device, **hyperparameters)
    model.load_state(data)
    return model
