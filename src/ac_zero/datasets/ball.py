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

``max_relator_length`` bounds the ball, and it bounds it the same way the environment
bounds an episode: a move that would make a relator longer than the bound is not played,
so a group with an over-long relator is not in the graph at all. That is what keeps the
exactness above *for the model trained on it* -- the distances are shortest paths through
the very graph the model moves in, not through long groups its encoder could never hold
and its environment would never let it enter. The bound is on each relator, never on the
sum of them, and it belongs to the dataset rather than to the consumer: it is recorded in
the file and carried in its name, and a run whose encoder capacity disagrees with it is
refused. ``0`` leaves the ball unbounded -- the whole graph, no model attached.

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

__all__ = ["BallConfig", "BallReport", "ball_groups_path", "grow_ball"]

# Emitted incrementally during long runs: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]

# Batches kept in flight, so the worker pool expands the next one while the main process
# merges the current -- the same software pipelining `dataset grow` uses.
_LOOKAHEAD = 2


def ball_groups_path(directory: str | Path, rank: int, max_relator_length: int) -> Path:
    """Name a ball's group file after the two things that define its graph.

    ``ball_rank2_rel48.groups.json`` -- the bound is in the name because a ball grown
    under a different one is a *different dataset*, not a longer run of the same one,
    and the model trained on it has a different input shape. An unbounded ball keeps
    the plain ``ball_rank2.groups.json``. The companion annotation and split files
    derive from this stem, so they inherit the bound for free.
    """
    suffix = f"_rel{max_relator_length}" if max_relator_length > 0 else ""
    return Path(directory) / f"ball_rank{rank}{suffix}.groups.json"


@dataclass(frozen=True, slots=True)
class BallConfig:
    """Parameters for one closest-first generation run."""

    rank: int
    moveset: str = "strict-ac"
    target: int = 100  # new groups to add this run
    # Longest relator a group may carry to be in the ball; 0 grows it unbounded. A move
    # that would overshoot it is one the environment masks, so the ball holds exactly
    # the groups a model with this encoder capacity can reach (see the module docstring).
    max_relator_length: int = 0
    workers: int = 0
    batch_size: int = 256
    # Rewrite both files every this many hours so an interrupted run keeps its progress;
    # 0 writes only at the end. Timed rather than counted in groups because what a
    # checkpoint buys is a bound on the *work* an interruption can destroy, and the cost
    # of taking one is a full rewrite of both documents -- which grows with the ball while
    # a group count does not. Whoever runs the ball pairs this with pushing the checkpoint
    # somewhere durable (`aczero dataset ball` uploads it), so the interval is really the
    # answer to "how much progress am I willing to lose?".
    checkpoint_hours: float = 0.0
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
    ball = OriginBall.load_or_seed(
        groups_path, annotations_path, config.rank, config.moveset, config.max_relator_length
    )
    if progress is not None:
        progress(
            "growing ball",
            {
                "path": str(groups_path),
                "rank": config.rank,
                "moveset": config.moveset,
                "max_relator_length": config.max_relator_length,
                "target": config.target,
                "start_groups": len(ball),
                "complete_depth": ball.complete_depth,
            },
        )
        _, message, metrics = describe_worker_pool(config.workers)
        progress(message, metrics)

    added = expanded = logged = 0
    timed_out = False
    deadline = None if config.time_limit_s is None else time.monotonic() + config.time_limit_s
    checkpoint_s = config.checkpoint_hours * 3600
    next_checkpoint = time.monotonic() + checkpoint_s if checkpoint_s > 0 else None
    with ExpansionPool(
        config.rank, config.max_relator_length, config.workers, ball.inverse_ids
    ) as pool:
        inflight: deque[tuple[list[int], BatchHandle]] = deque()
        claimed = ball.expanded

        def submit_next() -> bool:
            nonlocal claimed
            if claimed >= len(ball):
                return False
            batch = list(range(claimed, min(claimed + config.batch_size, len(ball))))
            claimed = batch[-1] + 1
            inflight.append((batch, pool.submit_batch([ball.presentation(i) for i in batch])))
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
                        "groups": len(ball),
                        "complete_depth": ball.complete_depth,
                        "distance": ball.max_distance(),
                    },
                )
                logged = added
            # A checkpoint is taken between merges -- a consistent point, with nothing in
            # flight unmerged -- so the pair of documents it leaves is always resumable.
            if next_checkpoint is not None and time.monotonic() >= next_checkpoint:
                ball.write(groups_path, annotations_path)
                next_checkpoint = time.monotonic() + checkpoint_s
                if progress is not None:
                    progress("checkpoint", {"groups": len(ball), "added": added})

    ball.write(groups_path, annotations_path)
    report = BallReport(
        total=len(ball),
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
