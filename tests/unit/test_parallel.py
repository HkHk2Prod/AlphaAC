import os

from ac_zero.system.parallel import (
    detect_core_count,
    imap_ordered,
    parallel_map,
    resolve_worker_count,
)


def _square(value: int) -> int:
    return value * value


_PREFIX = ""


def _init_prefix(prefix: str) -> None:
    global _PREFIX
    _PREFIX = prefix


def _label(value: int) -> str:
    return f"{_PREFIX}{value}"


def test_detect_core_count_is_physical_and_bounded() -> None:
    cores = detect_core_count()
    # At least one core, and never more than the logical-processor count.
    assert cores >= 1
    assert cores <= (os.cpu_count() or 1)


def test_resolve_worker_count_handles_auto_and_relative() -> None:
    cores = detect_core_count()
    # 0/None autodetect to the physical core count.
    assert resolve_worker_count(0) == cores
    assert resolve_worker_count(None) == cores
    assert resolve_worker_count(3) == 3
    assert resolve_worker_count(-1) == max(1, cores - 1)
    # Never drops below a single (in-process) worker.
    assert resolve_worker_count(-10 * cores) == 1


def test_parallel_map_preserves_order_in_process() -> None:
    assert parallel_map(_square, [1, 2, 3, 4], workers=1) == [1, 4, 9, 16]


def test_parallel_map_preserves_order_across_processes() -> None:
    items = list(range(8))
    assert parallel_map(_square, items, workers=2) == [v * v for v in items]


def test_imap_ordered_runs_initializer_in_each_mode() -> None:
    sequential = list(
        imap_ordered(_label, [1, 2, 3], workers=1, initializer=_init_prefix, initargs=("s",))
    )
    assert sequential == ["s1", "s2", "s3"]
    parallel = list(
        imap_ordered(_label, [1, 2, 3], workers=2, initializer=_init_prefix, initargs=("p",))
    )
    assert parallel == ["p1", "p2", "p3"]


def test_imap_ordered_is_lazy_for_single_item() -> None:
    # A single item short-circuits to the in-process path even when many workers
    # are requested, so no pool is spun up.
    assert list(imap_ordered(_square, [5], workers=8)) == [25]
