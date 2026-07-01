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

    This smoke encoder is intentionally simple and NumPy-based. Production JAX
    encoders should preserve the same information contract and reject or mask
    over-capacity states instead of silently losing mathematical data.
    """

    def __init__(self, max_word_length: int = 32) -> None:
        """Create an encoder with a per-relator token capacity."""
        self.max_word_length = max_word_length

    def encode(self, state: ACSearchState) -> PaddedEncoding:
        """Convert one immutable search state into padded token arrays."""
        rank = state.presentation.rank
        rows = []
        masks = []
        for relator in state.presentation.relators:
            row = [letter + rank + 1 for letter in relator.letters[: self.max_word_length]]
            pad = self.max_word_length - len(row)
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
