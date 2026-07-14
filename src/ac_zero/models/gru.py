from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ac_zero.models.batch import EncodedBatch
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, vocabulary_size
from ac_zero.models.trainable import TrainablePolicyValueModel


class _GRUTrunk(nn.Module):
    """Run a GRU over each relator's reserved slot, pool, and add global features.

    The input is the fixed ``(rank, max_relator_tokens)`` padded grid. Every relator
    of every state in the batch is one packed sequence, so a batch of ``B`` states at
    rank ``n`` runs as ``B * n`` sequences in a single recurrence; packing to the real
    lengths keeps trailing padding out of it. The per-relator final hidden states are
    mean-pooled back per state (empty relators contribute nothing) and concatenated
    with the global Markov features for the heads.
    """

    def __init__(self, vocab: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, batch: EncodedBatch) -> torch.Tensor:
        size, rank, slots = batch.tokens.shape
        tokens = batch.tokens.reshape(size * rank, slots)
        lengths = batch.mask.reshape(size * rank, slots).sum(dim=1)
        packed = pack_padded_sequence(
            self.embedding(tokens),
            # pack_padded_sequence reads the lengths on the host, whatever device the
            # sequences themselves live on.
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        # Zero the final state of any empty relator so it drops out of the mean.
        per_relator = hidden[-1] * (lengths > 0).unsqueeze(1)
        pooled = per_relator.reshape(size, rank, -1).mean(dim=1)
        return torch.cat([pooled, batch.globals], dim=1)


class GRUPolicyValueModel(TrainablePolicyValueModel):
    """Gated recurrent unit over each relator's reserved token slot.

    Every relator row of the fixed ``(rank, max_relator_tokens)`` grid is embedded
    with a shared table and consumed by a GRU as an independent packed sequence, so
    padding stays out of the recurrence; the per-relator final hidden states are
    mean-pooled and concatenated with global features for the heads. Training
    propagates gradients through the full recurrence (backpropagation through time).
    """

    architecture = "gru"

    def __init__(
        self, *, seed: int = 0, device: str = "cpu", embed_dim: int = 8, hidden_dim: int = 16
    ) -> None:
        super().__init__(seed=seed, device=device, embed_dim=embed_dim, hidden_dim=hidden_dim)

    def _build_trunk(self, batch: EncodedBatch) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        hidden = self._hp["hidden_dim"]
        trunk = _GRUTrunk(vocabulary_size(batch.rank), embed, hidden)
        return trunk, hidden + GLOBAL_FEATURE_COUNT
