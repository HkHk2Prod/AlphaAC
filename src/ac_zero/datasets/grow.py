from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.datasets.expand import BatchHandle, ExpansionPool
from ac_zero.datasets.groups import SCHEMA_VERSION, GroupNode, GroupStore, SelectStrategy
from ac_zero.system.parallel import describe_worker_pool

__all__ = ["SCHEMA_VERSION", "GrowConfig", "GrowReport", "grow_dataset"]

# Emitted incrementally during long grow runs: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]

# Batches kept in flight at once. With the next batch already expanding in the
# worker pool, the main process's serial select+merge overlaps expansion instead
# of stalling on a round barrier; two is enough to hide the gap on any core count.
_LOOKAHEAD = 2


@dataclass(frozen=True, slots=True)
class GrowConfig:
    """Parameters for one persistent grow run.

    `select` picks which open group to expand next: ``smallest`` always takes the
    shortest (by total relator length), giving one deterministic canonical
    frontier; ``weighted-random`` samples with a bias toward short groups (steered
    by `short_bias` and `seed`) so independent machines explore divergent paths.
    """

    rank: int
    target: int
    select: SelectStrategy = "smallest"
    seed: int = 0
    total_length_cap: int = 48
    short_bias: float = 2.0
    workers: int = 0
    # Groups claimed and expanded per round. Kept independent of the worker count
    # so the dataset a run produces is reproducible regardless of how many cores
    # expand it; larger batches keep the persistent worker pool saturated and let
    # expansion overlap the main process's serial merge.
    batch_size: int = 256
    # Dump the dataset to disk every this many newly added groups so an
    # interrupted long run keeps its progress; 0 dumps only at the end. Each
    # checkpoint rewrites the whole file, so keep it well above a handful.
    checkpoint_every: int = 5000
    # Emit an incremental progress log every this many newly added groups; 0
    # silences the per-round updates (the start/finish lines still fire). Kept
    # separate from checkpointing so disk writes stay rarer than status lines.
    log_every: int = 1000
    # Soft wall-clock budget in seconds; None runs until `target` is reached or
    # the frontier is exhausted. When set, the run stops submitting new batches
    # once the budget is spent, drains the in-flight ones, and flushes -- a clean
    # exit at a round boundary, not a mid-batch kill.
    time_limit_s: float | None = None


@dataclass(frozen=True, slots=True)
class GrowReport:
    """Summary of one grow run."""

    total: int
    added: int
    expanded: int
    frontier: int
    max_length: int


def grow_dataset(
    path: str | Path, config: GrowConfig, *, progress: ProgressCallback | None = None
) -> GrowReport:
    """Expand a persistent dataset outward from the trivial group.

    Loads the dataset at `path` (seeding the trivial root on the first run),
    repeatedly expands non-exhausted groups by every catalog move, and records
    each novel group with its co-optimal construction edges until `config.target`
    new groups have been added, the reachable frontier (within the length cap) is
    exhausted, or the optional `config.time_limit_s` wall-clock budget runs out.
    The file is rewritten atomically -- both at the end and every
    `config.checkpoint_every` added groups -- so a run interrupted mid-way resumes
    from its last checkpoint, and every run only ever grows the database.

    Expansion is software-pipelined: the next batch is already expanding in the
    worker pool while the main process merges the current one, so the serial merge
    overlaps expansion instead of stalling on a round barrier. A `claimed` set
    keeps an in-flight group from being re-selected, while groups whose expansion
    has not been merged yet stay open, so an interrupted run still resumes cleanly.
    """
    path = Path(path)
    rng = random.Random(config.seed)
    graph = GroupStore.load_or_seed(path, config.rank)
    if progress is not None:
        progress("growing dataset", _start_metrics(path, config, len(graph.nodes)))
        _, message, metrics = describe_worker_pool(config.workers)
        progress(message, metrics)

    added = 0
    expanded = 0
    checkpointed = 0
    logged = 0
    timed_out = False
    deadline = None if config.time_limit_s is None else time.monotonic() + config.time_limit_s
    claimed: set[str] = set()
    inflight: deque[tuple[list[GroupNode], BatchHandle]] = deque()
    with ExpansionPool(config.rank, config.total_length_cap, config.workers) as pool:

        def submit_next() -> bool:
            batch = graph.select_batch(
                config.select, rng, config.batch_size, config.short_bias, claimed
            )
            if not batch:
                return False
            claimed.update(node.content_hash for node in batch)
            inflight.append((batch, pool.submit_batch([node.presentation for node in batch])))
            return True

        def refill() -> None:
            nonlocal timed_out
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True  # stop submitting; drain what is already in flight
                return
            while added < config.target and len(inflight) < _LOOKAHEAD:
                if not submit_next():
                    break

        refill()  # prime the pipeline
        while inflight:
            batch, handle = inflight.popleft()
            records_list = handle.result()
            refill()  # top up before merging, so the next batch expands during this merge
            for parent, records in zip(batch, records_list, strict=True):
                claimed.discard(parent.content_hash)
                expanded += 1
                added += graph.merge(parent, records)
            refill()  # the merge may have revealed new frontier; never let the pipe drain
            if progress is not None and config.log_every > 0 and added - logged >= config.log_every:
                progress(
                    "growing dataset",
                    {
                        "added": added,
                        "target": config.target,
                        "expanded": expanded,
                        "groups": len(graph.nodes),
                    },
                )
                logged = added
            # Snapshot to disk between merges (a consistent point) so an interrupted
            # run resumes from the last checkpoint rather than losing everything.
            if config.checkpoint_every > 0 and added - checkpointed >= config.checkpoint_every:
                graph.write(path)
                checkpointed = added
                if progress is not None:
                    progress("checkpoint", {"groups": len(graph.nodes), "added": added})

    graph.write(path)
    report = GrowReport(len(graph.nodes), added, expanded, graph.frontier(), graph.max_length())
    if progress is not None:
        progress(
            "grow complete" if not timed_out else "grow stopped: time limit",
            {
                "groups": report.total,
                "added": added,
                "expanded": expanded,
                "frontier": report.frontier,
            },
        )
    return report


def _start_metrics(path: Path, config: GrowConfig, start_groups: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "rank": config.rank,
        "target": config.target,
        "select": config.select,
        "seed": config.seed,
        "start_groups": start_groups,
    }
