from __future__ import annotations

import torch


def use_single_torch_thread() -> None:
    """Pin torch to one intra-op thread, in this process.

    Self-play evaluates the policy/value net one state at a time, so torch's default
    intra-op pool (one thread per core) spends far more on synchronization barriers
    than the arithmetic it splits up. Self-play already fans out over *processes*, so
    leaving the default in place has every worker spawn its own full pool and
    oversubscribe the machine by a factor of the core count. Call this in the main
    process and in every worker initializer. Supervised training is the exception --
    it feeds the net large minibatches, which do profit from the pool -- so it does
    not call this.
    """
    torch.set_num_threads(1)


def select_device(request: str) -> torch.device:
    """Resolve a configured device name to a real torch device.

    ``"auto"`` takes CUDA when the machine offers it and CPU otherwise -- so the same
    config trains on a laptop and on a Kaggle GPU without an edit. Any other value is
    passed to torch verbatim, so a requested-but-absent accelerator raises here rather
    than silently costing a run its speed.
    """
    if request != "auto":
        return torch.device(request)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
