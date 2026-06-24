from __future__ import annotations

import numpy as np

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.autograd import Node, concat_cols
from ac_zero.models.features import (
    GLOBAL_FEATURE_COUNT,
    RELATOR_FEATURE_COUNT,
    global_features,
    relator_features,
)
from ac_zero.models.trainable import TrainablePolicyValueModel


class DeepSetsPolicyValueModel(TrainablePolicyValueModel):
    """Permutation-invariant DeepSets model over per-relator descriptors.

    Each relator is embedded independently by a shared element network ``phi``,
    the embeddings are sum-pooled (the source of permutation invariance), and a
    set network ``rho`` combines the pooled vector with global features. Relator
    order therefore cannot change the prediction.
    """

    architecture = "deepsets"

    def __init__(self, *, seed: int = 0, hidden_dim: int = 16) -> None:
        super().__init__(seed=seed, hidden_dim=hidden_dim)

    def _build_trunk(self, rng: np.random.Generator, encoding: PaddedEncoding) -> int:
        del encoding
        hidden = self._hp["hidden_dim"]
        scale_phi = 1.0 / np.sqrt(RELATOR_FEATURE_COUNT)
        rho_in = hidden + GLOBAL_FEATURE_COUNT
        scale_rho = 1.0 / np.sqrt(rho_in)
        self._param("phi_w", rng.normal(0.0, scale_phi, (RELATOR_FEATURE_COUNT, hidden)))
        self._param("phi_b", np.zeros((1, hidden)))
        self._param("rho_w", rng.normal(0.0, scale_rho, (rho_in, hidden)))
        self._param("rho_b", np.zeros((1, hidden)))
        return hidden

    def _forward_trunk(self, encoding: PaddedEncoding) -> Node:
        relators = Node(relator_features(encoding))
        embedded = (relators @ self._params["phi_w"] + self._params["phi_b"]).relu()
        pooled = embedded.sum(axis=0, keepdims=True)
        combined = concat_cols([pooled, Node(global_features(encoding)[np.newaxis, :])])
        return (combined @ self._params["rho_w"] + self._params["rho_b"]).relu()
