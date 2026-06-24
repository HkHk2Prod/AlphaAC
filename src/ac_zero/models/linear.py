from __future__ import annotations

import numpy as np

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.autograd import Node
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, global_features
from ac_zero.models.trainable import TrainablePolicyValueModel


class LinearPolicyValueModel(TrainablePolicyValueModel):
    """Linear policy/value model over fixed whole-presentation features.

    This is the deterministic CPU baseline: the trunk is the identity on the
    global feature vector, so only the policy and value heads carry parameters.
    """

    architecture = "linear_policy_value"

    def _build_trunk(self, rng: np.random.Generator, encoding: PaddedEncoding) -> int:
        del rng, encoding
        return GLOBAL_FEATURE_COUNT

    def _forward_trunk(self, encoding: PaddedEncoding) -> Node:
        return Node(global_features(encoding)[np.newaxis, :])
