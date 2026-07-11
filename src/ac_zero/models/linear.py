from __future__ import annotations

import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, global_features, vocabulary_size
from ac_zero.models.torch_utils import float_tensor, long_tensor
from ac_zero.models.trainable import TrainablePolicyValueModel


class _LinearTrunk(nn.Module):
    """Embed every relator letter and hand the flattened grid to the heads.

    A shared embedding table maps each signed-generator token to a learned vector,
    so the same letter embeds identically everywhere. The ``(rank, max_relator_tokens)``
    grid is embedded, flattened, and concatenated with the global Markov features.
    The trunk carries no further parameters, so the policy/value heads remain a
    linear map over the embedded token features.
    """

    def __init__(self, vocab: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        tokens = long_tensor(encoding.tokens)
        embeds = self.embedding(tokens).reshape(1, -1)
        globals_ = float_tensor(global_features(encoding)).unsqueeze(0)
        return torch.cat([embeds, globals_], dim=1)


class LinearPolicyValueModel(TrainablePolicyValueModel):
    """Linear policy/value model over the embedded relator sequence matrix.

    The trunk only embeds the relator letters and flattens them alongside the
    global features, so the trainable heads form a linear model over the same
    word-aware features the residual MLP consumes. This is the deterministic CPU
    baseline: a single learned embedding plus linear heads, no hidden trunk layers.
    """

    architecture = "linear_policy_value"

    def __init__(self, *, seed: int = 0, embed_dim: int = 8) -> None:
        super().__init__(seed=seed, embed_dim=embed_dim)

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        feature_dim = encoding.tokens.size * embed + GLOBAL_FEATURE_COUNT
        return _LinearTrunk(vocabulary_size(encoding), embed), feature_dim
