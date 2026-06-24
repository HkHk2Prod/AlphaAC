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


class GRUPolicyValueModel(TrainablePolicyValueModel):
    """Gated recurrent unit over the flattened relator token sequence.

    Tokens are embedded and consumed by a standard GRU cell; the final hidden
    state is concatenated with global features for the heads. Training propagates
    gradients through the full recurrence (backpropagation through time).
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

    def _build_trunk(self, rng: np.random.Generator, encoding: PaddedEncoding) -> int:
        embed = self._hp["embed_dim"]
        hidden = self._hp["hidden_dim"]
        vocab = vocabulary_size(encoding)
        self._param("embed", rng.normal(0.0, 0.1, (vocab, embed)))
        scale_in = 1.0 / np.sqrt(embed)
        scale_hidden = 1.0 / np.sqrt(hidden)
        for gate in ("z", "r", "n"):
            self._param(f"w_{gate}", rng.normal(0.0, scale_in, (embed, hidden)))
            self._param(f"u_{gate}", rng.normal(0.0, scale_hidden, (hidden, hidden)))
            self._param(f"b_{gate}", np.zeros((1, hidden)))
        return hidden + GLOBAL_FEATURE_COUNT

    def _forward_trunk(self, encoding: PaddedEncoding) -> Node:
        tokens = token_sequence(encoding, self._hp["max_steps"])
        embeds = embedding_lookup(self._params["embed"], tokens)
        hidden = Node(np.zeros((1, self._hp["hidden_dim"])))
        for step in range(tokens.shape[0]):
            current = embedding_lookup(embeds, np.asarray([step], dtype=np.int64))
            hidden = self._cell(current, hidden)
        return concat_cols([hidden, Node(global_features(encoding)[np.newaxis, :])])

    def _cell(self, x: Node, hidden: Node) -> Node:
        update = self._gate("z", x, hidden).sigmoid()
        reset = self._gate("r", x, hidden).sigmoid()
        candidate = (
            x @ self._params["w_n"] + (reset * hidden) @ self._params["u_n"] + self._params["b_n"]
        ).tanh()
        return (1.0 - update) * candidate + update * hidden

    def _gate(self, name: str, x: Node, hidden: Node) -> Node:
        return (
            x @ self._params[f"w_{name}"]
            + hidden @ self._params[f"u_{name}"]
            + self._params[f"b_{name}"]
        )
