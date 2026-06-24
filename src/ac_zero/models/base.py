from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ac_zero.encoding.padded import PaddedEncoding


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    """Policy logits and scalar value predicted for one encoded search state.

    `logits` has one entry for every action in the deterministic action catalog.
    Callers apply legal-action masks before sampling or taking an argmax. `value`
    predicts normalized future improvement from the current Markov state, not
    reward already collected earlier in the episode.
    """

    logits: NDArray[np.float64]
    value: float


class PolicyValueModel(Protocol):
    """Common interface implemented by all policy-value architectures.

    Search code depends only on this protocol, so JAX/Flax models, smoke-test
    NumPy models, and checkpoint-backed wrappers can be swapped without changing
    MCTS or training orchestration.
    """

    def apply(self, encoding: PaddedEncoding, action_count: int) -> PolicyValueOutput: ...


class UniformPolicyValueModel:
    """Deterministic baseline model used by smoke tests and fallbacks.

    The model intentionally returns zero logits and zero value. Once an action
    mask is applied, the induced policy is uniform over legal moves. This keeps
    CLI and search paths runnable while the full neural architectures mature.
    """

    def apply(self, encoding: PaddedEncoding, action_count: int) -> PolicyValueOutput:
        """Return neutral logits/value for a single encoded state."""
        del encoding
        return PolicyValueOutput(np.zeros(action_count, dtype=np.float64), 0.0)
