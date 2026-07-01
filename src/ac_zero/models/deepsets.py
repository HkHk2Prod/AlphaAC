from __future__ import annotations

import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import (
    GLOBAL_FEATURE_COUNT,
    RELATOR_FEATURE_COUNT,
    global_features,
    relator_features,
)
from ac_zero.models.torch_utils import float_tensor
from ac_zero.models.trainable import TrainablePolicyValueModel


class _DeepSetsTrunk(nn.Module):
    """Embed each relator with a shared ``phi``, sum-pool, then combine via ``rho``."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.phi = nn.Linear(RELATOR_FEATURE_COUNT, hidden_dim)
        self.rho = nn.Linear(hidden_dim + GLOBAL_FEATURE_COUNT, hidden_dim)

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        relators = float_tensor(relator_features(encoding))
        embedded = torch.relu(self.phi(relators))
        pooled = embedded.sum(dim=0, keepdim=True)
        globals_ = float_tensor(global_features(encoding)).unsqueeze(0)
        combined = torch.cat([pooled, globals_], dim=1)
        return torch.relu(self.rho(combined))


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

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        del encoding
        hidden = self._hp["hidden_dim"]
        return _DeepSetsTrunk(hidden), hidden
