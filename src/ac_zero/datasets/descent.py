from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.datasets.io import atomic_write_json
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.search.descent import DescentConfig, descent_distance
from ac_zero.system.parallel import describe_worker_pool, imap_ordered

# Emitted incrementally during long annotation passes: (message, metrics).
ProgressCallback = Callable[[str, dict[str, Any]], None]

# Per-entry fields written by this pass. ``descent_distance`` is the fewest AC
# moves that strictly shorten the presentation (the training-difficulty label);
# ``descent_proven`` marks whether that value is exact (see search/descent.py).
DISTANCE_FIELD = "descent_distance"
PROVEN_FIELD = "descent_proven"


@dataclass(frozen=True, slots=True)
class DescentAnnotateConfig:
    """Parameters for one dataset descent-annotation pass."""

    total_length_cap: int = 48
    max_depth: int = 32
    max_expansions: int = 20_000
    workers: int = 0
    # Rewrite the whole file every this many freshly computed entries so an
    # interrupted long pass keeps its progress; 0 writes only at the end.
    checkpoint_every: int = 5000

    def search(self) -> DescentConfig:
        return DescentConfig(self.total_length_cap, self.max_depth, self.max_expansions)


@dataclass(frozen=True, slots=True)
class DescentReport:
    """Summary of one descent-annotation pass."""

    total: int
    computed: int
    with_descent: int
    proven: int
    max_distance: int


def annotate_descent(
    path: str | Path,
    config: DescentAnnotateConfig,
    *,
    output: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> DescentReport:
    """Annotate every dataset entry with its length-descent distance.

    For each presentation the pass runs a bounded breadth-first search for the
    fewest AC moves that strictly reduce the total length, writing that count to
    ``descent_distance`` (``None`` when none is found) and whether it is exact to
    ``descent_proven``. Entries already carrying a *proven* answer are skipped, so
    repeated passes are cheap and a later pass with a bigger budget only resolves
    the still-unknown ones. Entries are searched easiest-first (shortest, then
    lowest construction depth) so early checkpoints cover the easy majority, and
    the file is rewritten atomically -- at the end and every
    ``checkpoint_every`` computed entries -- so an interrupted pass resumes.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = data.get("instances", [])
    destination = Path(output) if output is not None else Path(path)

    if progress is not None:
        _, worker_message, worker_metrics = describe_worker_pool(config.workers)
        progress(
            "annotating descent distance",
            {
                "input": str(path),
                "output": str(destination),
                "total": len(entries),
                "total_length_cap": config.total_length_cap,
                "max_depth": config.max_depth,
            },
        )
        progress(worker_message, worker_metrics)

    todo = sorted((e for e in entries if _needs_compute(e)), key=_difficulty_key)
    results = imap_ordered(
        _compute_entry,
        todo,
        workers=config.workers,
        initializer=_init_worker,
        initargs=(config.search(),),
    )

    computed = 0
    checkpointed = 0
    for entry, (distance, proven) in zip(todo, results, strict=True):
        entry[DISTANCE_FIELD] = distance
        entry[PROVEN_FIELD] = proven
        computed += 1
        if config.checkpoint_every > 0 and computed - checkpointed >= config.checkpoint_every:
            _refresh_provenance(data, entries)
            atomic_write_json(destination, data)
            checkpointed = computed
            if progress is not None:
                progress("checkpoint", {"computed": computed, "total": len(entries)})

    _refresh_provenance(data, entries)
    atomic_write_json(destination, data)

    found = [d for e in entries if isinstance((d := e.get(DISTANCE_FIELD)), int)]
    report = DescentReport(
        total=len(entries),
        computed=computed,
        with_descent=len(found),
        proven=sum(1 for e in entries if e.get(PROVEN_FIELD) is True),
        max_distance=max(found, default=0),
    )
    if progress is not None:
        progress(
            "descent annotation complete",
            {
                "total": report.total,
                "computed": report.computed,
                "with_descent": report.with_descent,
                "proven": report.proven,
            },
        )
    return report


def _needs_compute(entry: dict[str, Any]) -> bool:
    # A proven answer (a proven minimum or a proven-absent descent) never changes,
    # so re-searching it only wastes work; anything else is recomputed, letting a
    # later pass with a larger budget settle a previously unknown entry.
    return entry.get(PROVEN_FIELD) is not True


def _difficulty_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    """Sort key placing the easiest (shortest, shallowest) presentations first."""
    length = sum(len(relator) for relator in entry.get("relators", []))
    difficulty = entry.get("difficulty")
    depth = difficulty if isinstance(difficulty, int) else length
    return (length, depth, str(entry.get("content_hash")))


def _refresh_provenance(data: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        return
    distances = [d for e in entries if isinstance((d := e.get(DISTANCE_FIELD)), int)]
    provenance["descent_annotated"] = sum(1 for e in entries if PROVEN_FIELD in e)
    provenance["max_descent_distance"] = max(distances, default=0)


# Per-worker search config and rank-keyed catalog cache, built once per process so
# the hot search path never re-allocates move objects.
_WORKER_CONFIG: DescentConfig | None = None
_WORKER_CATALOGS: dict[int, ActionCatalog] = {}


def _init_worker(config: DescentConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    _WORKER_CATALOGS.clear()


def _compute_entry(entry: dict[str, Any]) -> tuple[int | None, bool]:
    assert _WORKER_CONFIG is not None
    presentation = BalancedPresentation.from_json(entry)
    catalog = _WORKER_CATALOGS.get(presentation.rank)
    if catalog is None:
        catalog = ActionCatalog(presentation.rank)
        _WORKER_CATALOGS[presentation.rank] = catalog
    result = descent_distance(presentation, catalog, _WORKER_CONFIG)
    return result.distance, result.proven
