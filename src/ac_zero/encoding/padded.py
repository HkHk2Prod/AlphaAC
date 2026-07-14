from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ac_zero.environment.state import ACSearchState


@dataclass(frozen=True, slots=True)
class PaddedEncoding:
    """Fixed-shape NumPy representation of an `ACSearchState`.

    `tokens` stores shifted signed-generator IDs with zero reserved for padding.
    `mask` marks real token positions. `scalar_features` carries non-word Markov
    information such as remaining horizon and normalized lengths.
    """

    tokens: NDArray[np.int64]
    mask: NDArray[np.bool_]
    scalar_features: NDArray[np.float64]

    def as_observation(self) -> dict[str, NDArray[np.generic]]:
        """Return a Gymnasium-space-conformant dict of the encoded arrays."""
        return {
            "tokens": self.tokens,
            "mask": self.mask.astype(np.int8),
            "scalar_features": self.scalar_features,
        }


class StateEncoder:
    """Encode search states into padded arrays for model consumption.

    Every state is laid out on the same fixed ``(rank, max_relator_tokens)`` grid, so
    encodings stack into a minibatch without further alignment. The capacity is a
    hard contract, not a truncation point: a relator too long to fit raises rather
    than being silently clipped, since a clipped relator is a different -- and
    mathematically wrong -- presentation. Size the capacity to the data
    (``ac_zero.datasets.supervised_store`` reads the longest relator in a dataset)
    or to the environment's ``total_length_cap``, which no episode can exceed.
    """

    def __init__(self, max_relator_tokens: int = 32) -> None:
        """Create an encoder with a per-relator token capacity."""
        self.max_relator_tokens = max_relator_tokens

    def encode(self, state: ACSearchState) -> PaddedEncoding:
        """Convert one immutable search state into padded token arrays.

        Raises ``ValueError`` when a relator exceeds the encoder's capacity.
        """
        rank = state.presentation.rank
        rows = []
        masks = []
        for relator in state.presentation.relators:
            if len(relator.letters) > self.max_relator_tokens:
                raise ValueError(
                    f"relator of length {len(relator.letters)} exceeds the encoder capacity "
                    f"of {self.max_relator_tokens} tokens; raise max_relator_tokens so the "
                    "presentation fits instead of being truncated"
                )
            row = [letter + rank + 1 for letter in relator.letters]
            pad = self.max_relator_tokens - len(row)
            rows.append(row + [0] * pad)
            masks.append([True] * len(row) + [False] * pad)
        scalars = np.asarray(
            [
                state.moves_remaining,
                state.best_length / max(1, state.initial_length),
                state.presentation.total_length / max(1, state.initial_length),
                state.initial_length,
            ],
            dtype=np.float64,
        )
        return PaddedEncoding(
            np.asarray(rows, dtype=np.int64),
            np.asarray(masks, dtype=np.bool_),
            scalars,
        )
