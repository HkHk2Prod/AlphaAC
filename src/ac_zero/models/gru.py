from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, global_features, vocabulary_size
from ac_zero.models.torch_utils import float_tensor, long_tensor
from ac_zero.models.trainable import TrainablePolicyValueModel


class _GRUTrunk(nn.Module):
    """Run a GRU over each relator's reserved slot, pool, and add global features.

    The input is the fixed ``(rank, max_relator_tokens)`` padded grid. Each relator
    row is embedded and fed to the GRU as its own sequence, packed to its real
    length so trailing padding never enters the recurrence; the per-relator final
    hidden states are mean-pooled (empty relators contribute nothing) and
    concatenated with the global Markov features for the heads.
    """

    def __init__(self, vocab: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, encoding: PaddedEncoding) -> torch.Tensor:
        tokens = long_tensor(encoding.tokens)
        lengths = torch.from_numpy(np.ascontiguousarray(encoding.mask.sum(axis=1)))
        embeds = self.embedding(tokens)
        packed = pack_padded_sequence(
            embeds, lengths.clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        # Zero the final state of any empty relator so it drops out of the mean.
        per_relator = hidden[-1] * (lengths > 0).unsqueeze(1)
        pooled = per_relator.mean(dim=0, keepdim=True)
        globals_ = float_tensor(global_features(encoding)).unsqueeze(0)
        return torch.cat([pooled, globals_], dim=1)


class GRUPolicyValueModel(TrainablePolicyValueModel):
    """Gated recurrent unit over each relator's reserved token slot.

    Every relator row of the fixed ``(rank, max_relator_tokens)`` grid is embedded
    with a shared table and consumed by a GRU as an independent packed sequence, so
    padding stays out of the recurrence; the per-relator final hidden states are
    mean-pooled and concatenated with global features for the heads. Training
    propagates gradients through the full recurrence (backpropagation through time).
    """

    architecture = "gru"

    def __init__(self, *, seed: int = 0, embed_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__(seed=seed, embed_dim=embed_dim, hidden_dim=hidden_dim)

    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        hidden = self._hp["hidden_dim"]
        trunk = _GRUTrunk(vocabulary_size(encoding), embed, hidden)
        return trunk, hidden + GLOBAL_FEATURE_COUNT
