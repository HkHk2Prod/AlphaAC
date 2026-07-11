from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import vocabulary_size
from ac_zero.models.torch_utils import long_tensor
from ac_zero.models.trainable import TrainablePolicyValueModel


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


class _RotaryEncoding(nn.Module):
    """Rotary positional encoding (RoPE) over a ``(seq, dim)`` tensor.

    Position is injected by rotating query/key feature pairs by an angle that grows
    with sequence index, so attention depends only on relative offsets between
    tokens rather than absolute positions. ``dim`` must be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("rotary embedding dimension must be even")
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = torch.outer(torch.arange(x.shape[0]).float(), self.inv_freq)
        emb = torch.cat([angles, angles], dim=-1)
        return x * emb.cos() + _rotate_half(x) * emb.sin()


class _RotaryAttention(nn.Module):
    """Single-head self-attention with RoPE on queries/keys and a padding mask."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.rotary = _RotaryEncoding(dim)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        q = self.rotary(self.query(x))
        k = self.rotary(self.key(x))
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~key_mask.unsqueeze(0), float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        return self.out(attention @ self.value(x))


class _EncoderBlock(nn.Module):
    """Pre-residual self-attention and feed-forward sublayers with layer norm."""

    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.attention = _RotaryAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x, key_mask))
        return self.norm2(x + self.feed_forward(x))


class _TransformerTrunk(nn.Module):
    """Embed the fixed ``(rank, max_relator_tokens)`` grid and run RoPE blocks.

    Each relator occupies a reserved ``max_relator_tokens`` slot, so the flattened
    sequence places relator ``i`` at a deterministic position block and RoPE encodes
    the boundary as a fixed offset. Padding slots (token 0) embed to zero and are
    masked out of attention and the mean pool so only real letters contribute.
    """

    def __init__(self, vocab: int, embed_dim: int, ff_dim: int, layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.blocks = nn.ModuleList(_EncoderBlock(embed_dim, ff_dim) for _ in range(layers))

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        tokens = long_tensor(encoding.tokens.reshape(-1))
        mask = torch.from_numpy(np.ascontiguousarray(encoding.mask.reshape(-1)))
        if not bool(mask.any()):  # fully reduced presentation: attend uniformly
            mask = torch.ones_like(mask)
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return x[mask].mean(dim=0, keepdim=True)


class TransformerPolicyValueModel(TrainablePolicyValueModel):
    """Multi-block self-attention encoder over the relator token grid.

    Letters are embedded with a shared table and processed by stacked encoder
    blocks whose attention uses rotary positional encoding. The input is the fixed
    ``(rank, max_relator_tokens)`` padded grid, so every relator keeps a reserved
    slot instead of being concatenated into one variable-length stream. Real tokens
    are mean-pooled into the feature vector; padding and global Markov features are
    excluded, so the trunk sees only the relators.
    """

    architecture = "transformer"

    def __init__(
        self,
        *,
        seed: int = 0,
        embed_dim: int = 8,
        ff_dim: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__(seed=seed, embed_dim=embed_dim, ff_dim=ff_dim, num_layers=num_layers)

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        trunk = _TransformerTrunk(
            vocabulary_size(encoding), embed, self._hp["ff_dim"], self._hp["num_layers"]
        )
        return trunk, embed
