from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.labels import known_solution
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import (
    ConjugateRelatorMove,
    InvertRelatorMove,
    MultiplyRelatorsMove,
    PrimitiveMove,
)
from ac_zero.system.parallel import describe_worker_pool, resolve_worker_count

# Emitted incrementally during long generation runs: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class GeneratedInstance:
    """Synthetic solvable presentation plus a private reverse certificate path.

    `difficulty` is the number of scramble moves that actually changed the
    presentation; it is an upper bound on the strict-AC solution length and a
    useful per-instance training/evaluation label.
    """

    presentation: BalancedPresentation
    reverse_moves: tuple[PrimitiveMove, ...]
    difficulty: int


def generate_solvable(rank: int, depth: int, seed: int) -> GeneratedInstance:
    """Scramble the standard presentation with seeded strict primitive moves.

    The returned presentation is guaranteed to be AC-trivial because it was
    produced from the exact standard presentation. The reverse path is returned
    for validation and fixture generation, but training code should not expose
    it as an observation.
    """

    rng = random.Random(seed)
    catalog = ActionCatalog(rank)
    pres = BalancedPresentation.standard(rank)
    applied: list[PrimitiveMove] = []
    for _ in range(depth):
        move = rng.choice(catalog.moves)
        nxt = move.apply(pres)
        if nxt.content_hash == pres.content_hash:
            continue
        pres = nxt
        applied.append(move)
    reverse_parts: list[PrimitiveMove] = []
    for move in reversed(applied):
        reverse_parts.extend(_inverse_primitive_sequence(move))
    reverse = tuple(reverse_parts)
    return GeneratedInstance(
        BalancedPresentation.from_letters(
            rank,
            [r.letters for r in pres.relators],
            presentation_id=f"synthetic-r{rank}-d{depth}-s{seed}",
            provenance={
                "family": "seeded_strict_ac_scramble",
                "seed": seed,
                "depth": depth,
                "difficulty": len(applied),
            },
        ),
        reverse,
        len(applied),
    )


def _inverse_primitive_sequence(move: PrimitiveMove) -> tuple[PrimitiveMove, ...]:
    """Expand the inverse of one primitive move into strict primitive moves."""
    if isinstance(move, InvertRelatorMove):
        return (move,)
    if isinstance(move, ConjugateRelatorMove):
        return (ConjugateRelatorMove(move.target, -move.generator),)
    if isinstance(move, MultiplyRelatorsMove):
        return (
            InvertRelatorMove(move.source),
            MultiplyRelatorsMove(move.target, move.source),
            InvertRelatorMove(move.source),
        )
    raise TypeError(f"unsupported primitive move {move!r}")


def _generate_candidate(args: tuple[int, int, int]) -> GeneratedInstance:
    """Worker entry point: build the candidate for one attempt index's parameters."""
    rank, depth, seed = args
    return generate_solvable(rank, depth, seed)


def _candidate_stream(
    *,
    rank: int,
    depth_cycle: Sequence[int],
    seed: int,
    budget: int,
    workers: int,
) -> Iterator[GeneratedInstance]:
    """Yield candidates for attempt indices ``0..budget-1`` in order.

    Attempt ``i`` is the pure function ``generate_solvable(rank, depth_cycle[i %
    len], seed + i)``, so candidates can be produced out of order and re-sorted.
    With a single worker they are generated lazily in-process; otherwise a reused
    process pool fills the stream in chunks. Either way candidates are yielded in
    attempt order, so the downstream accept/reject pass — and thus the dataset —
    is identical regardless of the worker count.
    """
    cycle = len(depth_cycle)
    resolved = resolve_worker_count(workers)
    if resolved <= 1:
        for index in range(budget):
            yield generate_solvable(rank, depth_cycle[index % cycle], seed + index)
        return
    # A bounded chunk keeps wasted work small when the consumer stops early: at
    # most one in-flight chunk is generated past the point the target is reached.
    round_size = max(resolved * 256, 1)
    chunksize = max(1, round_size // (resolved * 8))
    with ProcessPoolExecutor(max_workers=resolved) as executor:
        start = 0
        while start < budget:
            stop = min(start + round_size, budget)
            args = [
                (rank, depth_cycle[index % cycle], seed + index) for index in range(start, stop)
            ]
            yield from executor.map(_generate_candidate, args, chunksize=chunksize)
            start = stop


def generate_dataset(
    *,
    rank: int,
    count: int,
    depth: int = 0,
    seed: int,
    min_total_length: int = 0,
    min_relator_length: int = 0,
    unique: bool = True,
    max_attempts: int | None = None,
    depths: Sequence[int] | None = None,
    workers: int = 0,
    progress: ProgressCallback | None = None,
) -> list[GeneratedInstance]:
    """Generate `count` distinct guaranteed-solvable instances with difficulty labels.

    Duplicate presentations (by content hash) and the trivial standard
    presentation are skipped when `unique` is set, so the dataset scales to large
    counts without repetition. Each returned instance carries its scramble
    difficulty. When `depths` is given, scramble depth is cycled across that list
    so the set spans an easy-to-hard difficulty range; otherwise the single
    `depth` is used. Raises if the constraints cannot be satisfied within the
    attempt budget.

    Candidate construction fans out across `workers` processes; the default `0`
    autodetects and uses every CPU core (set 1 for in-process, or a negative count
    to leave that many free). The deterministic accept/reject pass runs on
    candidates in attempt order, so the dataset is identical regardless of the
    worker count.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if min_total_length < 0:
        raise ValueError("min_total_length must be non-negative")
    if min_relator_length < 0:
        raise ValueError("min_relator_length must be non-negative")
    depth_cycle = list(depths) if depths else [depth]
    if any(d < 1 for d in depth_cycle):
        raise ValueError("each scramble depth must be positive")

    budget = max_attempts if max_attempts is not None else max(count * 200, 2000)
    if progress is not None:
        # Describe the task up front with every parameter that shapes the output,
        # so a run is reproducible from its log alone.
        progress(
            "generating dataset",
            {
                "rank": rank,
                "count": count,
                "depths": str(depth_cycle),
                "seed": seed,
                "min_total_length": min_total_length,
                "min_relator_length": min_relator_length,
                "unique": unique,
                "max_attempts": budget,
            },
        )
    trivial_hash = BalancedPresentation.standard(rank).content_hash
    seen: set[str] = set()
    instances: list[GeneratedInstance] = []
    attempt = 0
    skipped_length = 0
    skipped_duplicate = 0
    # Report progress at ~10 checkpoints so large runs stay visible without spam.
    interval = max(1, count // 10)
    if progress is not None:
        _, message, metrics = describe_worker_pool(workers)
        progress(message, metrics)
    candidates = _candidate_stream(
        rank=rank, depth_cycle=depth_cycle, seed=seed, budget=budget, workers=workers
    )
    while len(instances) < count:
        instance = next(candidates, None)
        if instance is None:
            break
        attempt += 1
        pres = instance.presentation
        relator_lengths = [len(relator) for relator in pres.relators]
        if sum(relator_lengths) < min_total_length or min(relator_lengths) < min_relator_length:
            skipped_length += 1
            continue
        if unique:
            content = pres.content_hash
            if content == trivial_hash or content in seen:
                skipped_duplicate += 1
                continue
            seen.add(content)
        instances.append(instance)
        if progress is not None and len(instances) % interval == 0:
            progress(
                "generating instances",
                {"generated": len(instances), "target": count, "attempts": attempt},
            )
    if len(instances) < count:
        raise ValueError(
            "could not generate enough distinct presentations matching dataset constraints; "
            "increase depth or max_attempts, or relax length/uniqueness constraints"
        )
    if progress is not None:
        progress(
            "generation complete",
            {
                "generated": len(instances),
                "attempts": attempt,
                "skipped_duplicate": skipped_duplicate,
                "skipped_length": skipped_length,
            },
        )
    return instances


def write_dataset(
    path: str | Path,
    *,
    rank: int,
    count: int,
    depth: int = 0,
    seed: int,
    min_total_length: int = 0,
    min_relator_length: int = 0,
    unique: bool = True,
    max_attempts: int | None = None,
    depths: Sequence[int] | None = None,
    workers: int = 0,
    progress: ProgressCallback | None = None,
) -> None:
    """Write a versioned JSON dataset of distinct seeded solvable presentations."""
    instances = generate_dataset(
        rank=rank,
        count=count,
        depth=depth,
        seed=seed,
        min_total_length=min_total_length,
        min_relator_length=min_relator_length,
        unique=unique,
        max_attempts=max_attempts,
        depths=depths,
        workers=workers,
        progress=progress,
    )
    difficulties = [instance.difficulty for instance in instances]
    provenance: dict[str, int | str | bool | list[int]] = {
        "seed": seed,
        "generator": "strict_ac_scramble",
        "unique": unique,
        "count": len(instances),
    }
    if depths:
        provenance["depths"] = list(depths)
    else:
        provenance["depth"] = depth
    if difficulties:
        provenance["min_difficulty"] = min(difficulties)
        provenance["max_difficulty"] = max(difficulties)
    if min_total_length:
        provenance["min_total_length"] = min_total_length
    if min_relator_length:
        provenance["min_relator_length"] = min_relator_length
    data = {
        "schema_version": "aczero-dataset-v2",
        "rank": rank,
        "instances": [
            {
                **instance.presentation.to_json(),
                "difficulty": instance.difficulty,
                # The reverse scramble path is a known, but not proven-minimal,
                # strict-AC trivialization of every guaranteed-solvable instance.
                **known_solution(len(instance.reverse_moves), optimal=False).to_json(),
            }
            for instance in instances
        ],
        "provenance": provenance,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if progress is not None:
        progress("dataset written", {"path": str(path), "instances": len(instances)})
