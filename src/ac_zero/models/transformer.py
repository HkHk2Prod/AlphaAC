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


class _TransformerTrunk(nn.Module):
    """Token + positional embeddings, one self-attention encoder block, mean pool."""

    def __init__(self, vocab: int, embed_dim: int, ff_dim: int, max_steps: int) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.position = nn.Embedding(max_steps, embed_dim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=1,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        tokens = long_tensor(token_sequence(encoding, self.max_steps))
        positions = torch.arange(tokens.shape[0])
        sequence = (self.embedding(tokens) + self.position(positions)).unsqueeze(0)
        pooled = self.encoder(sequence).mean(dim=1)
        globals_ = float_tensor(global_features(encoding)).unsqueeze(0)
        return torch.cat([pooled, globals_], dim=1)


class TransformerPolicyValueModel(TrainablePolicyValueModel):
    """Single-block self-attention encoder over the relator token sequence.

    Token plus learned positional embeddings feed one scaled dot-product
    self-attention block with a residual feed-forward network. The attended tokens
    are mean-pooled and concatenated with global features for the heads. Gradients
    flow through the attention softmax during training.
    """

    architecture = "transformer"

    def __init__(
        self,
        *,
        seed: int = 0,
        embed_dim: int = 8,
        ff_dim: int = 16,
        max_steps: int = 24,
    ) -> None:
        super().__init__(seed=seed, embed_dim=embed_dim, ff_dim=ff_dim, max_steps=max_steps)

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        trunk = _TransformerTrunk(
            vocabulary_size(encoding), embed, self._hp["ff_dim"], self._hp["max_steps"]
        )
        return trunk, embed + GLOBAL_FEATURE_COUNT
