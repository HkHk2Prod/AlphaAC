from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

GLOBAL_FEATURE_COUNT = 8
RELATOR_FEATURE_COUNT = 6

Tokens = NDArray[np.int64]  # (batch, rank, max_relator_tokens)
Mask = NDArray[np.bool_]  # (batch, rank, max_relator_tokens)
Scalars = NDArray[np.float64]  # (batch, 4)


def _moments(
    tokens: Tokens, mask: Mask, axis: tuple[int, ...] | int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the `(count, mean, population std)` of the real tokens along ``axis``.

    Padding slots are zeroed rather than removed, so the statistics of a whole
    batch come out of two sums instead of one ragged gather per row. Rows with no
    real token get zeros, matching what ``np.mean``/``np.std`` of an empty slice
    contributed in the per-state implementation this vectorizes.
    """
    counts = mask.sum(axis=axis).astype(np.float64)
    real = np.where(mask, tokens, 0).astype(np.float64)
    safe = np.maximum(counts, 1.0)
    mean = real.sum(axis=axis) / safe
    second = np.square(real).sum(axis=axis) / safe
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    nonempty = counts > 0
    return counts, np.where(nonempty, mean, 0.0), np.where(nonempty, std, 0.0)


def global_features(tokens: Tokens, mask: Mask, scalars: Scalars) -> NDArray[np.float64]:
    """Whole-presentation Markov features shared by the linear and MLP trunks.

    The layout matches the original CPU baseline: a bias term, normalized
    remaining horizon, best/current length ratios, a length scale, and three
    aggregate token statistics. Keeping it stable preserves checkpoint meaning.
    """
    capacity = float(max(1, tokens.shape[1] * tokens.shape[2]))
    counts, mean, std = _moments(tokens, mask, axis=(1, 2))
    initial_length = np.maximum(1.0, scalars[:, 3])
    return np.stack(
        [
            np.ones_like(initial_length),
            scalars[:, 0] / initial_length,
            scalars[:, 1],
            scalars[:, 2],
            initial_length / 100.0,
            counts / capacity,
            mean / 10.0,
            std / 10.0,
        ],
        axis=1,
    )


def relator_features(tokens: Tokens, mask: Mask) -> NDArray[np.float64]:
    """Per-relator descriptors for the permutation-invariant DeepSets trunk.

    Returns one row per relator with length, token statistics, and the fraction
    of positively signed generators. Rows are produced in relator order, but the
    DeepSets pooling that consumes them is order invariant by construction.
    """
    rank = tokens.shape[1]
    capacity = float(max(1, tokens.shape[2]))
    counts, mean, std = _moments(tokens, mask, axis=2)
    positive = np.where(mask & (tokens > rank + 1), 1.0, 0.0).sum(axis=2) / np.maximum(counts, 1.0)
    return np.stack(
        [
            np.ones_like(counts),
            counts / capacity,
            mean / 10.0,
            std / 10.0,
            np.where(counts > 0, positive, 0.0),
            (counts > 0).astype(np.float64),
        ],
        axis=2,
    )


def vocabulary_size(rank: int) -> int:
    """Embedding table size: padding slot plus signed generators for the rank.

    For a balanced presentation the rank equals the relator count, and signed
    generator IDs run from 1 to ``2 * rank`` with 0 reserved for padding.
    """
    return int(2 * max(1, rank) + 2)
