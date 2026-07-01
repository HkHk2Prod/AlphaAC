from __future__ import annotations

import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, global_features
from ac_zero.models.torch_utils import float_tensor
from ac_zero.models.trainable import TrainablePolicyValueModel


class _LinearTrunk(nn.Module):
    """Identity trunk that hands the global feature vector to the heads."""

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        return float_tensor(global_features(encoding)).unsqueeze(0)


class LinearPolicyValueModel(TrainablePolicyValueModel):
    """Linear policy/value model over fixed whole-presentation features.

    This is the deterministic CPU baseline: the trunk is the identity on the
    global feature vector, so only the policy and value heads carry parameters.
    """

    architecture = "linear_policy_value"

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        del encoding
        return _LinearTrunk(), GLOBAL_FEATURE_COUNT
