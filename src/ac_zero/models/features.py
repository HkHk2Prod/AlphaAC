from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ac_zero.encoding.padded import PaddedEncoding

GLOBAL_FEATURE_COUNT = 8
RELATOR_FEATURE_COUNT = 6


def global_features(encoding: PaddedEncoding) -> NDArray[np.float64]:
    """Whole-presentation Markov features shared by the linear and MLP trunks.

    The layout matches the original CPU baseline: a bias term, normalized
    remaining horizon, best/current length ratios, a length scale, and three
    aggregate token statistics. Keeping it stable preserves checkpoint meaning.
    """
    tokens = encoding.tokens[encoding.mask].astype(np.float64)
    token_capacity = float(max(1, encoding.tokens.size))
    mean_token = float(np.mean(tokens)) if tokens.size else 0.0
    std_token = float(np.std(tokens)) if tokens.size else 0.0
    scalars = encoding.scalar_features
    initial_length = max(1.0, float(scalars[3]))
    return np.asarray(
        [
            1.0,
            float(scalars[0]) / initial_length,
            float(scalars[1]),
            float(scalars[2]),
            initial_length / 100.0,
            float(tokens.size) / token_capacity,
            mean_token / 10.0,
            std_token / 10.0,
        ],
        dtype=np.float64,
    )


def relator_features(encoding: PaddedEncoding) -> NDArray[np.float64]:
    """Per-relator descriptors for the permutation-invariant DeepSets trunk.

    Returns one row per relator with length, token statistics, and the fraction
    of positively signed generators. Rows are produced in relator order, but the
    DeepSets pooling that consumes them is order invariant by construction.
    """
    tokens = encoding.tokens
    mask = encoding.mask
    rank = tokens.shape[0]
    capacity = float(max(1, tokens.shape[1]))
    rows = []
    for relator, relator_mask in zip(tokens, mask, strict=True):
        real = relator[relator_mask].astype(np.float64)
        length = float(real.size)
        mean_token = float(np.mean(real)) if real.size else 0.0
        std_token = float(np.std(real)) if real.size else 0.0
        positive = float(np.mean(real > (rank + 1))) if real.size else 0.0
        rows.append(
            [
                1.0,
                length / capacity,
                mean_token / 10.0,
                std_token / 10.0,
                positive,
                1.0 if real.size else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def token_sequence(encoding: PaddedEncoding, max_steps: int) -> NDArray[np.int64]:
    """Flatten real (unpadded) token IDs in relator order, capped at ``max_steps``."""
    flat = encoding.tokens[encoding.mask].astype(np.int64)
    if flat.size == 0:
        return np.zeros(1, dtype=np.int64)
    return flat[:max_steps]


def vocabulary_size(encoding: PaddedEncoding) -> int:
    """Embedding table size: padding slot plus signed generators for the rank.

    For a balanced presentation the rank equals the relator count, and signed
    generator IDs run from 1 to ``2 * rank`` with 0 reserved for padding.
    """
    rank = max(1, encoding.tokens.shape[0])
    return int(2 * rank + 2)
