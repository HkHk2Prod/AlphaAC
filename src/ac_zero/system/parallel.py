from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# CPU-bound pure-Python/NumPy work (search expansion, the reverse-mode autodiff
# model) is held back by the GIL, so threads cannot run it in parallel; these
# helpers fan work out across *processes* instead. Results are always returned in
# input order, so a run's output is independent of the worker count and the
# project's determinism guarantees survive parallelization.


def _physical_core_count() -> int | None:
    """Count physical CPU cores from /proc/cpuinfo, or None when undeterminable.

    A physical core is a unique ``(physical id, core id)`` pair; hyperthreads on
    the same core share that pair and so are collapsed. Returns None on non-Linux
    systems or kernels that omit those fields, leaving the caller to fall back.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    cores: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in text.splitlines():
        if line.startswith("physical id"):
            physical_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core_id = line.split(":", 1)[1].strip()
        elif not line.strip():  # a blank line terminates one processor block
            if physical_id is not None and core_id is not None:
                cores.add((physical_id, core_id))
            physical_id = core_id = None
    if physical_id is not None and core_id is not None:  # trailing block, no newline
        cores.add((physical_id, core_id))
    return len(cores) or None


def detect_core_count() -> int:
    """Best-effort physical CPU core count, falling back to logical processors.

    Physical cores are the better default for CPU-bound work: the second
    hyperthread on a core adds little throughput and oversubscribing it can hurt.
    When the physical topology cannot be read (non-Linux, some kernels) this falls
    back to ``os.cpu_count()`` (logical processors), and finally to 1.
    """
    return _physical_core_count() or os.cpu_count() or 1


def resolve_worker_count(workers: int | None) -> int:
    """Translate a configured worker count into a concrete positive process count.

    ``None`` or ``0`` means "use every physical CPU core" (see
    :func:`detect_core_count`); a negative count leaves that many cores free (so
    ``-2`` uses ``cores - 2``). The result is always at least 1, so a sequential,
    in-process run is the safe fallback.
    """
    cores = detect_core_count()
    if workers is None or workers == 0:
        return cores
    if workers < 0:
        return max(1, cores + workers)
    return workers


def describe_worker_pool(
    workers: int | None,
) -> tuple[int, str, dict[str, float | int | bool | str]]:
    """Resolve a worker count and describe it for the run log.

    Returns the concrete worker count alongside a human-readable message and a
    metrics dict (``workers``, detected ``cores``, and a ``parallel`` flag).
    Callers emit these through the project's event/progress logs so every run
    records whether it fanned work out across multiple worker processes or stayed
    in-process, making the degree of parallelism auditable after the fact.
    """
    resolved = resolve_worker_count(workers)
    cores = detect_core_count()
    if resolved > 1:
        message = f"fanning out across {resolved} worker processes"
    else:
        message = "running in-process (single worker)"
    return resolved, message, {"workers": resolved, "cores": cores, "parallel": resolved > 1}


def imap_ordered(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> Iterator[R]:
    """Yield ``func(item)`` for each item in input order, fanning out over processes.

    Runs inline in the current process when the resolved worker count is 1 (or
    there is at most one item), which keeps the sequential path allocation-free,
    easy to debug, and free of pickling. Otherwise the work is spread across a
    process pool. Either way results are yielded lazily in input order, so a
    consumer can report incremental progress and still see deterministic,
    worker-count-independent output.
    """
    resolved = resolve_worker_count(workers)
    materialized = list(items)
    if resolved <= 1 or len(materialized) <= 1:
        if initializer is not None:
            initializer(*initargs)
        for item in materialized:
            yield func(item)
        return
    # Split on initializer so the type checker picks the right ProcessPoolExecutor
    # overload (initargs is only meaningful alongside an initializer).
    if initializer is None:
        executor = ProcessPoolExecutor(max_workers=resolved)
    else:
        executor = ProcessPoolExecutor(
            max_workers=resolved, initializer=initializer, initargs=initargs
        )
    with executor:
        yield from executor.map(func, materialized)


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[R]:
    """Eagerly map ``func`` over ``items`` across processes, preserving input order.

    A thin :func:`imap_ordered` wrapper for callers that need every result at
    once rather than streaming.
    """
    return list(
        imap_ordered(func, items, workers=workers, initializer=initializer, initargs=initargs)
    )
