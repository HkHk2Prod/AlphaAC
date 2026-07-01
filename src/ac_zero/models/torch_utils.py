from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def float_tensor(array: NDArray[np.float64]) -> torch.Tensor:
    """Convert a NumPy feature array to a contiguous float32 tensor."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def long_tensor(array: NDArray[np.int64]) -> torch.Tensor:
    """Convert NumPy token IDs to a contiguous int64 tensor for embedding lookups."""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64))
