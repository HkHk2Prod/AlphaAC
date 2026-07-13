from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def use_single_torch_thread() -> None:
    """Pin torch to one intra-op thread, in this process.

    The policy/value nets are tiny -- tens of thousands of parameters, evaluated one
    state at a time -- so torch's default intra-op pool (one thread per core) spends
    far more on synchronization barriers than the arithmetic it splits up. Training
    already fans out over *processes*, so leaving the default in place has every
    worker spawn its own full pool and oversubscribe the machine by a factor of the
    core count. Call this in the main process and in every worker initializer.
    """
    torch.set_num_threads(1)


def float_tensor(array: NDArray[np.float64]) -> torch.Tensor:
    """Convert a NumPy feature array to a contiguous float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def long_tensor(array: NDArray[np.int64]) -> torch.Tensor:
    """Convert NumPy token IDs to a contiguous int64 tensor for embedding lookups."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64))
