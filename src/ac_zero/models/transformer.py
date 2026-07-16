from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ac_zero.models.batch import EncodedBatch
from ac_zero.models.features import vocabulary_size
from ac_zero.models.trainable import TrainablePolicyValueModel


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


class _RotaryEncoding(nn.Module):
    """Rotary positional encoding (RoPE) over the last two dims of a ``(..., seq, dim)`` tensor.

    Position is injected by rotating query/key feature pairs by an angle that grows
    with sequence index, so attention depends only on relative offsets between
    tokens rather than absolute positions. ``dim`` must be even.
    """

    inv_freq: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("rotary embedding dimension must be even")
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        emb = torch.cat([angles, angles], dim=-1)
        return x * emb.cos() + _rotate_half(x) * emb.sin()


class _RotaryAttention(nn.Module):
    """Multi-head self-attention with RoPE on queries/keys and a padding mask.

    Heads split the model dimension, so ``num_heads = 1`` is exactly the original
    single-head layer -- same parameter shapes, same arithmetic -- and a wide model
    gets the head count it needs without a new checkpoint format.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.rotary = _RotaryEncoding(self.head_dim)
        self.scale = self.head_dim**-0.5

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(batch, seq, dim)`` into ``(batch, heads, seq, head_dim)``."""
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q = self.rotary(self._heads(self.query(x)))
        k = self.rotary(self._heads(self.key(x)))
        v = self._heads(self.value(x))
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ v
        merged = attended.transpose(1, 2).reshape(batch, seq, dim)
        out: torch.Tensor = self.out(merged)
        return out


class _EncoderBlock(nn.Module):
    """Pre-residual self-attention and feed-forward sublayers with layer norm."""

    def __init__(self, dim: int, ff_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = _RotaryAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x, key_mask))
        out: torch.Tensor = self.norm2(x + self.feed_forward(x))
        return out


class _TransformerTrunk(nn.Module):
    """Embed the fixed ``(rank, max_relator_tokens)`` grid and run RoPE blocks.

    Each relator occupies a reserved ``max_relator_tokens`` slot, so the flattened
    sequence places relator ``i`` at a deterministic position block and RoPE encodes
    the boundary as a fixed offset. Padding slots (token 0) embed to zero and are
    masked out of attention and the mean pool so only real letters contribute; a
    fully reduced presentation has no real letters at all, so it attends uniformly
    rather than softmaxing a row of ``-inf``.
    """

    def __init__(
        self,
        vocab: int,
        embed_dim: int,
        ff_dim: int,
        layers: int,
        heads: int,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.blocks = nn.ModuleList(_EncoderBlock(embed_dim, ff_dim, heads) for _ in range(layers))
        # Recompute each block's activations during backward instead of storing them.
        # A deep, wide trunk is activation-bound long before it is parameter-bound, so
        # this is what lets the ~100M-parameter config train at a real batch size on a
        # 16 GB GPU; it costs a second forward per block, so it stays off by default and
        # the large supervised config opts in.
        self._grad_checkpoint = grad_checkpoint

    def forward(self, batch: EncodedBatch) -> torch.Tensor:
        size = batch.size
        mask = batch.mask.reshape(size, -1)
        mask = mask | ~mask.any(dim=1, keepdim=True)
        x = self.embedding(batch.tokens.reshape(size, -1))
        # Checkpointing only earns its recompute when a graph is being built for backward;
        # under `no_grad` inference it would recompute for nothing, so gate on grad state.
        checkpointing = self._grad_checkpoint and torch.is_grad_enabled()
        for block in self.blocks:
            if checkpointing:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        weights = mask.unsqueeze(-1).to(x.dtype)
        pooled: torch.Tensor = (x * weights).sum(dim=1) / weights.sum(dim=1)
        return pooled


class TransformerPolicyValueModel(TrainablePolicyValueModel):
    """Multi-block self-attention encoder over the relator token grid.

    Letters are embedded with a shared table and processed by stacked encoder
    blocks whose attention uses rotary positional encoding. The input is the fixed
    ``(rank, max_relator_tokens)`` padded grid, so every relator keeps a reserved
    slot instead of being concatenated into one variable-length stream. Real tokens
    are mean-pooled into the feature vector; padding and global Markov features are
    excluded, so the trunk sees only the relators.

    This is the architecture the supervised stage scales up: ``embed_dim``,
    ``ff_dim``, ``num_layers`` and ``num_heads`` are the knobs that take it from the
    few-thousand-parameter CPU baseline to the ~100M-parameter model trained
    directly on the dataset (see ``configs/experiments/supervised_large.yaml``).
    """

    architecture = "transformer"

    def __init__(
        self,
        *,
        seed: int = 0,
        device: str = "cpu",
        embed_dim: int = 8,
        ff_dim: int = 16,
        num_layers: int = 2,
        num_heads: int = 1,
        grad_checkpoint: int = 0,
    ) -> None:
        super().__init__(
            seed=seed,
            device=device,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            # A 0/1 flag rather than a size, but it rides the same `model_config` channel
            # and so is stored and serialized alongside the other hyperparameters.
            grad_checkpoint=grad_checkpoint,
        )

    def _build_trunk(self, batch: EncodedBatch) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        trunk = _TransformerTrunk(
            vocabulary_size(batch.rank),
            embed,
            self._hp["ff_dim"],
            self._hp["num_layers"],
            self._hp["num_heads"],
            grad_checkpoint=bool(self._hp.get("grad_checkpoint", 0)),
        )
        return trunk, embed
