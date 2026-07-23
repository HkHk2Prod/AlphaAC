from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ac_zero.encoding.padded import PaddedEncoding


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    """Policy logits and the value heads predicted for one encoded search state.

    `logits` has one entry for every action in the deterministic action catalog.
    Callers apply legal-action masks before sampling or taking an argmax.

    The value comes as three heads, of which a run uses two. `value` is the legacy
    scalar critic every non-navigation reward mode reads directly. The navigation
    reward instead splits its value along the seam that makes it `alpha`-invariant:
    `success` predicts the (discounted) probability of reaching the destination and
    `progress` the `alpha`-free shaping-return-to-go normalized by the start
    distance, and the environment recombines them as
    `L0 * (destination_scale * success + alpha * progress)` (see
    `ACEnvironment.leaf_value`). Splitting there means moving `alpha` never
    invalidates the critic -- it rescales a head that was never refit.
    """

    logits: NDArray[np.float64]
    value: float
    success: float = 0.0
    progress: float = 0.0


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
