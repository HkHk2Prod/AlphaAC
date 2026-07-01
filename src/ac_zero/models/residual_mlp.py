from __future__ import annotations

import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, global_features
from ac_zero.models.torch_utils import float_tensor
from ac_zero.models.trainable import TrainablePolicyValueModel


class _ResidualMLPTrunk(nn.Module):
    """Project the global features into a hidden space and add one residual block."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(GLOBAL_FEATURE_COUNT, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        x = float_tensor(global_features(encoding)).unsqueeze(0)
        hidden = torch.relu(self.fc1(x))
        residual = torch.relu(self.fc2(hidden))
        return hidden + residual


class ResidualMLPPolicyValueModel(TrainablePolicyValueModel):
    """Residual multilayer perceptron over the global Markov feature vector.

    The trunk projects the fixed feature vector into a hidden space and adds one
    residual ReLU block, giving the policy/value heads a nonlinear, trainable
    representation while staying small enough for deterministic CPU runs.
    """

    architecture = "residual_mlp"

    def __init__(self, *, seed: int = 0, hidden_dim: int = 16) -> None:
        super().__init__(seed=seed, hidden_dim=hidden_dim)

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        del encoding
        hidden = self._hp["hidden_dim"]
        return _ResidualMLPTrunk(hidden), hidden
