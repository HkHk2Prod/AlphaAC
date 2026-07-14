from __future__ import annotations

import torch
from torch import nn

from ac_zero.models.batch import EncodedBatch
from ac_zero.models.features import GLOBAL_FEATURE_COUNT, vocabulary_size
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

    def forward(self, batch: EncodedBatch) -> torch.Tensor:
        embeds = self.embedding(batch.tokens).reshape(batch.size, -1)
        return torch.cat([embeds, batch.globals], dim=1)


class LinearPolicyValueModel(TrainablePolicyValueModel):
    """Linear policy/value model over the embedded relator sequence matrix.

    The trunk only embeds the relator letters and flattens them alongside the
    global features, so the trainable heads form a linear model over the same
    word-aware features the residual MLP consumes. This is the deterministic
    baseline: a single learned embedding plus linear heads, no hidden trunk layers.
    """

    architecture = "linear_policy_value"

    def __init__(self, *, seed: int = 0, device: str = "cpu", embed_dim: int = 8) -> None:
        super().__init__(seed=seed, device=device, embed_dim=embed_dim)

    def _build_trunk(self, batch: EncodedBatch) -> tuple[nn.Module, int]:
        embed = self._hp["embed_dim"]
        feature_dim = batch.token_slots * embed + GLOBAL_FEATURE_COUNT
        return _LinearTrunk(vocabulary_size(batch.rank), embed), feature_dim
