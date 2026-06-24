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


def create_trainable_model(name: str, *, seed: int = 0) -> TrainablePolicyValueModel:
    """Construct a trainable policy-value model from a stable configuration name."""
    normalized = name.lower().replace("-", "_")
    try:
        return _TRAINABLE[normalized](seed=seed)
    except KeyError as exc:
        raise KeyError(f"unknown model {name!r}") from exc


def create_model(name: str, *, seed: int = 0) -> PolicyValueModel:
    """Construct any registered policy-value model, including the uniform baseline."""
    normalized = name.lower().replace("-", "_")
    if normalized == "uniform":
        return UniformPolicyValueModel()
    return create_trainable_model(normalized, seed=seed)


def model_from_json(data: dict[str, Any]) -> TrainablePolicyValueModel:
    """Reconstruct a trainable model and its weights from a checkpoint payload."""
    hyperparameters = dict(data.get("hyperparameters", {}))
    seed = int(hyperparameters.pop("seed", 0))
    model = _TRAINABLE[data["architecture"]](seed=seed, **hyperparameters)
    model.load_state(data)
    return model
