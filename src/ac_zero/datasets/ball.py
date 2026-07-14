"""Generate the exact ball of groups around the trivial group, closest first.

``dataset grow`` expands whichever group is *shortest*, under every universal move. The
distances of what it produces then have to be searched for afterwards, and can only ever
be found as upper bounds: a group's true shortest path to the origin may run through a
group the run never expanded. Under ``strict-ac`` barely half of a grown dataset can
reach the origin at all.

This generator inverts the construction. It walks outward from the trivial group by the
*inverses* of one move set's moves, in breadth-first order, so a group is discovered
exactly when the first inverse-move path reaches it -- and that path, reversed, is a
shortest path of forward moves from the group back to the origin. Two properties follow,
and they are the whole point:

* **Exact distances.** Every ``distance_to_origin`` is a proven optimum, so no annotation
  pass is needed: the generator emits the annotation file itself, with the co-optimal
  first moves a supervised policy is trained on.
* **Complete shells.** Groups are expanded in discovery order, so once the last group at
  distance ``d`` has been expanded, *every* group at distance ``d + 1`` is in the
  dataset. The deepest such ``d`` is recorded as ``complete_depth``.

Nothing is dropped for being long: a length cap would silently reroute (or delete) the
shortest paths that run through a long group, and forfeit the exactness above. The groups
too long for a model's encoder are filtered where they are consumed, not where they are
made.

Shells grow by roughly sevenfold a layer at rank 2, so a run is bounded by a group budget
rather than a depth, and reports the depth it completed.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.datasets.annotate import annotation_path
from ac_zero.datasets.ball_store import OriginBall
from ac_zero.datasets.expand import BatchHandle, ExpansionPool
from ac_zero.system.parallel import describe_worker_pool

__all__ = ["BallConfig", "BallReport", "grow_ball"]

# Emitted incrementally during long runs: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]

# Batches kept in flight, so the worker pool expands the next one while the main process
# merges the current -- the same software pipelining `dataset grow` uses.
_LOOKAHEAD = 2


@dataclass(frozen=True, slots=True)
class BallConfig:
    """Parameters for one closest-first generation run."""

    rank: int
    moveset: str = "strict-ac"
    target: int = 100  # new groups to add this run
    workers: int = 0
    batch_size: int = 256
    # Rewrite both files every this many newly added groups so an interrupted run keeps
    # its progress; 0 writes only at the end.
    checkpoint_every: int = 50_000
    log_every: int = 10_000
    # Soft wall-clock budget in seconds; None runs until `target` is reached.
    time_limit_s: float | None = None


@dataclass(frozen=True, slots=True)
class BallReport:
    """Summary of one closest-first generation run."""

    total: int
    added: int
    expanded: int
    # Every group whose distance to the origin is at most this is in the dataset.
    complete_depth: int
    max_distance: int
    max_length: int


def grow_ball(
    groups_path: str | Path, config: BallConfig, *, progress: ProgressCallback | None = None
) -> BallReport:
    """Expand the ball around the trivial group until ``config.target`` groups are added.

    Writes the groups to ``groups_path`` and their proven distances to the companion
    ``.<moveset>.annotations.json``, checkpointing both every ``config.checkpoint_every``
    groups. Because the file is the queue -- groups are expanded in the order they were
    discovered -- an interrupted run resumes from the count of expanded groups it left
    behind, without rebuilding a frontier.
    """
    groups_path = Path(groups_path)
    annotations_path = annotation_path(groups_path, config.moveset)
    ball = OriginBall.load_or_seed(groups_path, annotations_path, config.rank, config.moveset)
    if progress is not None:
        progress(
            "growing ball",
            {
                "path": str(groups_path),
                "rank": config.rank,
                "moveset": config.moveset,
                "target": config.target,
                "start_groups": len(ball.nodes),
                "complete_depth": ball.complete_depth,
            },
        )
        _, message, metrics = describe_worker_pool(config.workers)
        progress(message, metrics)

    added = expanded = checkpointed = logged = 0
    timed_out = False
    deadline = None if config.time_limit_s is None else time.monotonic() + config.time_limit_s
    # No length cap: dropping a long group would reroute the shortest paths that run
    # through it, and its distances would stop being optimal.
    with ExpansionPool(config.rank, 0, config.workers, ball.inverse_ids) as pool:
        inflight: deque[tuple[list[int], BatchHandle]] = deque()
        claimed = ball.expanded

        def submit_next() -> bool:
            nonlocal claimed
            if claimed >= len(ball.nodes):
                return False
            batch = list(range(claimed, min(claimed + config.batch_size, len(ball.nodes))))
            claimed = batch[-1] + 1
            inflight.append((batch, pool.submit_batch([ball.nodes[i].presentation for i in batch])))
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
            records = handle.result()
            refill()  # top up before merging, so the next batch expands during this merge
            for parent, neighbours in zip(batch, records, strict=True):
                added += ball.merge(parent, neighbours)
                expanded += 1
            # Batches are merged in the order they were submitted, so the expanded prefix
            # advances one whole batch at a time and never runs past an unmerged group.
            ball.expanded = batch[-1] + 1
            refill()  # the merge revealed new frontier; never let the pipe drain
            if progress is not None and config.log_every > 0 and added - logged >= config.log_every:
                progress(
                    "growing ball",
                    {
                        "added": added,
                        "target": config.target,
                        "groups": len(ball.nodes),
                        "complete_depth": ball.complete_depth,
                        "distance": ball.max_distance(),
                    },
                )
                logged = added
            if config.checkpoint_every > 0 and added - checkpointed >= config.checkpoint_every:
                ball.write(groups_path, annotations_path)
                checkpointed = added
                if progress is not None:
                    progress("checkpoint", {"groups": len(ball.nodes), "added": added})

    ball.write(groups_path, annotations_path)
    report = BallReport(
        total=len(ball.nodes),
        added=added,
        expanded=expanded,
        complete_depth=ball.complete_depth,
        max_distance=ball.max_distance(),
        max_length=ball.max_length(),
    )
    if progress is not None:
        progress(
            "ball stopped: time limit" if timed_out else "ball complete",
            {
                "groups": report.total,
                "added": report.added,
                "complete_depth": report.complete_depth,
                "max_distance": report.max_distance,
                "max_length": report.max_length,
            },
        )
    return report
