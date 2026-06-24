from __future__ import annotations

import numpy as np

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.autograd import Node
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, global_features
from ac_zero.models.trainable import TrainablePolicyValueModel


class ResidualMLPPolicyValueModel(TrainablePolicyValueModel):
    """Residual multilayer perceptron over the global Markov feature vector.

    The trunk projects the fixed feature vector into a hidden space and adds one
    residual ReLU block, giving the policy/value heads a nonlinear, trainable
    representation while staying small enough for deterministic CPU runs.
    """

    architecture = "residual_mlp"

    def __init__(self, *, seed: int = 0, hidden_dim: int = 16) -> None:
        super().__init__(seed=seed, hidden_dim=hidden_dim)

    def _build_trunk(self, rng: np.random.Generator, encoding: PaddedEncoding) -> int:
        del encoding
        hidden = self._hp["hidden_dim"]
        scale_in = 1.0 / np.sqrt(GLOBAL_FEATURE_COUNT)
        scale_hidden = 1.0 / np.sqrt(hidden)
        self._param("w1", rng.normal(0.0, scale_in, (GLOBAL_FEATURE_COUNT, hidden)))
        self._param("b1", np.zeros((1, hidden)))
        self._param("w2", rng.normal(0.0, scale_hidden, (hidden, hidden)))
        self._param("b2", np.zeros((1, hidden)))
        return hidden

    def _forward_trunk(self, encoding: PaddedEncoding) -> Node:
        x = Node(global_features(encoding)[np.newaxis, :])
        hidden = (x @ self._params["w1"] + self._params["b1"]).relu()
        residual = (hidden @ self._params["w2"] + self._params["b2"]).relu()
        return hidden + residual
