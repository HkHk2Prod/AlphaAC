from __future__ import annotations

import torch
from torch import nn

from ac_zero.models.batch import EncodedBatch
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, vocabulary_size
from ac_zero.models.trainable import TrainablePolicyValueModel


class _ResidualMLPTrunk(nn.Module):
    """Embed every relator letter, flatten the sequence matrix, add one residual block.

    A single shared embedding table maps each signed-generator token to a learned
    vector, so the same letter is embedded identically wherever it appears. The
    ``(rank, max_relator_tokens)`` padded token matrix is embedded and flattened, then
    concatenated with the global Markov features (horizon, lengths) that the token
    grid does not carry. This preserves the full word structure the old aggregate
    features discarded.
    """

    def __init__(self, vocab: int, embed_dim: int, token_slots: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        input_dim = token_slots * embed_dim + GLOBAL_FEATURE_COUNT
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch: EncodedBatch) -> torch.Tensor:
        embeds = self.embedding(batch.tokens).reshape(batch.size, -1)
        x = torch.cat([embeds, batch.globals], dim=1)
        hidden = torch.relu(self.fc1(x))
        residual = torch.relu(self.fc2(hidden))
        return hidden + residual


class ResidualMLPPolicyValueModel(TrainablePolicyValueModel):
    """Residual MLP over the embedded relator sequence matrix.

    Each letter of every relator is passed through a shared learned embedding; the
    embedded ``(rank, max_relator_tokens)`` grid is flattened, concatenated with the
    global Markov features, and projected into a hidden space with one residual
    ReLU block. Unlike the linear baseline, the trunk sees the actual word content
    rather than a handful of aggregate token statistics.
    """

    architecture = "residual_mlp"

    def __init__(
        self, *, seed: int = 0, device: str = "cpu", embed_dim: int = 8, hidden_dim: int = 64
    ) -> None:
        super().__init__(seed=seed, device=device, embed_dim=embed_dim, hidden_dim=hidden_dim)

    def _build_trunk(self, batch: EncodedBatch) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        hidden = self._hp["hidden_dim"]
        trunk = _ResidualMLPTrunk(vocabulary_size(batch.rank), embed, batch.token_slots, hidden)
        return trunk, hidden
