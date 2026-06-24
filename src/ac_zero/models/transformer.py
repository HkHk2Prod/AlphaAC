from __future__ import annotations

import numpy as np

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.autograd import Node, concat_cols, embedding_lookup
from ac_zero.models.features import (
    GLOBAL_FEATURE_COUNT,
    global_features,
    token_sequence,
    vocabulary_size,
)
from ac_zero.models.trainable import TrainablePolicyValueModel


class TransformerPolicyValueModel(TrainablePolicyValueModel):
    """Single-block self-attention encoder over the relator token sequence.

    Token plus learned positional embeddings feed one scaled dot-product
    self-attention block with a residual feed-forward network. The attended
    tokens are mean-pooled and concatenated with global features for the heads.
    Gradients flow through the attention softmax during training.
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

    def _build_trunk(self, rng: np.random.Generator, encoding: PaddedEncoding) -> int:
        embed = self._hp["embed_dim"]
        ff = self._hp["ff_dim"]
        vocab = vocabulary_size(encoding)
        scale = 1.0 / np.sqrt(embed)
        self._param("embed", rng.normal(0.0, 0.1, (vocab, embed)))
        self._param("pos", rng.normal(0.0, 0.1, (self._hp["max_steps"], embed)))
        for name in ("q", "k", "v", "o"):
            self._param(f"w_{name}", rng.normal(0.0, scale, (embed, embed)))
        self._param("ff_w1", rng.normal(0.0, scale, (embed, ff)))
        self._param("ff_b1", np.zeros((1, ff)))
        self._param("ff_w2", rng.normal(0.0, 1.0 / np.sqrt(ff), (ff, embed)))
        self._param("ff_b2", np.zeros((1, embed)))
        return embed + GLOBAL_FEATURE_COUNT

    def _forward_trunk(self, encoding: PaddedEncoding) -> Node:
        tokens = token_sequence(encoding, self._hp["max_steps"])
        steps = tokens.shape[0]
        embeds = embedding_lookup(self._params["embed"], tokens)
        positions = embedding_lookup(self._params["pos"], np.arange(steps, dtype=np.int64))
        sequence = embeds + positions

        attended = self._attention(sequence)
        residual = sequence + attended
        pooled = self._feed_forward(residual).mean(axis=0, keepdims=True)
        return concat_cols([pooled, Node(global_features(encoding)[np.newaxis, :])])

    def _attention(self, sequence: Node) -> Node:
        query = sequence @ self._params["w_q"]
        key = sequence @ self._params["w_k"]
        value = sequence @ self._params["w_v"]
        scale = 1.0 / float(np.sqrt(self._hp["embed_dim"]))
        weights = ((query @ key.transpose()) * scale).softmax_rows()
        attended: Node = (weights @ value) @ self._params["w_o"]
        return attended

    def _feed_forward(self, residual: Node) -> Node:
        hidden = (residual @ self._params["ff_w1"] + self._params["ff_b1"]).relu()
        projected = hidden @ self._params["ff_w2"] + self._params["ff_b2"]
        return residual + projected
