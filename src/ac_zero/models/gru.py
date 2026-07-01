from __future__ import annotations

import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import (
    GLOBAL_FEATURE_COUNT,
    global_features,
    token_sequence,
    vocabulary_size,
)
from ac_zero.models.torch_utils import float_tensor, long_tensor
from ac_zero.models.trainable import TrainablePolicyValueModel


class _GRUTrunk(nn.Module):
    """Embed the token sequence, run a GRU, and concatenate global features."""

    def __init__(self, vocab: int, embed_dim: int, hidden_dim: int, max_steps: int) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        tokens = long_tensor(token_sequence(encoding, self.max_steps))
        embeds = self.embedding(tokens).unsqueeze(0)
        _, hidden = self.gru(embeds)
        globals_ = float_tensor(global_features(encoding)).unsqueeze(0)
        return torch.cat([hidden[-1], globals_], dim=1)


class GRUPolicyValueModel(TrainablePolicyValueModel):
    """Gated recurrent unit over the flattened relator token sequence.

    Tokens are embedded and consumed by a GRU; the final hidden state is
    concatenated with global features for the heads. Training propagates gradients
    through the full recurrence (backpropagation through time).
    """

    architecture = "gru"

    def __init__(
        self,
        *,
        seed: int = 0,
        embed_dim: int = 8,
        hidden_dim: int = 16,
        max_steps: int = 24,
    ) -> None:
        super().__init__(seed=seed, embed_dim=embed_dim, hidden_dim=hidden_dim, max_steps=max_steps)

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        hidden = self._hp["hidden_dim"]
        trunk = _GRUTrunk(vocabulary_size(encoding), embed, hidden, self._hp["max_steps"])
        return trunk, hidden + GLOBAL_FEATURE_COUNT
