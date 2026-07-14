"""One minibatch of encoded search states, as torch tensors on a device.

Every trunk consumes this instead of a single :class:`PaddedEncoding`, so the same
architecture serves one-state inference inside search (a batch of one) and the
thousand-state minibatches supervised pretraining pushes through a GPU. The
handcrafted feature blocks are computed once per batch in vectorized NumPy and
handed over as tensors, so no trunk recomputes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.features import global_features, relator_features


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    """A ``(batch, rank, max_relator_tokens)`` grid of states and its feature blocks."""

    tokens: torch.Tensor  # (batch, rank, tokens) int64 -- shifted signed generators
    mask: torch.Tensor  # (batch, rank, tokens) bool -- real (non-padding) letters
    globals: torch.Tensor  # (batch, GLOBAL_FEATURE_COUNT) float32
    relators: torch.Tensor  # (batch, rank, RELATOR_FEATURE_COUNT) float32

    @property
    def size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def rank(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def token_slots(self) -> int:
        """Reserved token slots per state: ``rank * max_relator_tokens``."""
        return int(self.tokens.shape[1] * self.tokens.shape[2])

    @property
    def device(self) -> torch.device:
        return self.tokens.device


def encode_batch(encodings: Sequence[PaddedEncoding], device: torch.device) -> EncodedBatch:
    """Stack encoded states into one device-resident batch.

    Every encoding must share the encoder's ``(rank, max_relator_tokens)`` grid --
    :class:`ac_zero.encoding.padded.StateEncoder` guarantees that, and refuses to
    truncate a relator that would not fit, so a batch never silently loses letters.
    """
    if not encodings:
        raise ValueError("cannot encode an empty batch")
    tokens = np.stack([encoding.tokens for encoding in encodings])
    mask = np.stack([encoding.mask for encoding in encodings])
    scalars = np.stack([encoding.scalar_features for encoding in encodings])
    return EncodedBatch(
        tokens=torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.int64)).to(device),
        mask=torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_)).to(device),
        globals=_float(global_features(tokens, mask, scalars), device),
        relators=_float(relator_features(tokens, mask), device),
    )


def _float(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)
