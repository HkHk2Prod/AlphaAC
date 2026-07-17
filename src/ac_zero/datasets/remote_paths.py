"""Map local dataset filenames to their folders in the Hugging Face bucket.

The bucket used to hold every dataset file flat at its root, mixed in with the
scheduler state and the model checkpoints. This module places each dataset file
under a folder derived from the dataset it belongs to, so a ball's graph, its
per-move-set annotations, and their summaries sit together::

    datasets/rank{R}/rel-{N}/            ball_rank{R}_rel{N}.groups.json
    datasets/rank{R}/rel-{N}/{moveset}/  ball_rank{R}_rel{N}.{moveset}.annotations.json
    datasets/rank{R}/rel-{N}/{moveset}/  ball_rank{R}_rel{N}.{moveset}.annotations.summary.md

The graph (``groups``) file is defined by ``(rank, relator bound)`` alone and is
shared across move sets, so it sits at the ball root; annotations and their
summaries are per-move-set and sit under a ``{moveset}`` subfolder. An unbounded
ball uses ``rel-unbounded`` so a bounded and an unbounded ball of the same rank
never share a folder.

The remote path is derived from the filename so every caller keeps passing a bare
local path (``download_dataset(groups)``) and the folder falls out of the name --
the standard names carry the rank, relator bound, and move set the folder needs.
A filename that does not follow the ``*_rank{R}[_rel{N}].*`` convention (a
hand-named upload) keeps its bare name at the bucket root.
"""

from __future__ import annotations

import re
from pathlib import Path

# Top-level bucket folder holding every dataset (grouped by rank / relator bound).
DATASETS_PREFIX = "datasets"

# ``<something>_rank{R}`` with an optional ``_rel{N}`` bound, then the ``.`` that
# starts the file's role suffix (``groups.json``, ``strict-ac.annotations.json``).
# Non-greedy up to ``_rank`` so ``ball``/``train`` prefixes are matched, not skipped.
_RANK_RE = re.compile(r"^.+?_rank(?P<rank>\d+)(?:_rel(?P<rel>\d+))?\.(?P<rest>.+)$")

# Annotations (and their summaries) carry the move set as the segment before this
# marker: ``strict-ac.annotations.json`` -> move set ``strict-ac``. Graph files and
# their summaries have no move set and stay at the ball root.
_ANNOTATION_MARKER = ".annotations"


def ball_remote_dir(rank: int, max_relator_length: int) -> str:
    """The bucket folder for a ball's files: ``datasets/rank{R}/rel-{N}``.

    An unbounded ball (``max_relator_length <= 0``) uses ``rel-unbounded``, matching
    the unbounded ball's ``ball_rank{R}.groups.json`` name that carries no ``_rel``.
    """
    rel = f"rel-{max_relator_length}" if max_relator_length > 0 else "rel-unbounded"
    return f"{DATASETS_PREFIX}/rank{rank}/{rel}"


def _moveset_of(rest: str) -> str | None:
    """The move set an annotation/summary belongs to, or ``None`` for ball-level files."""
    index = rest.find(_ANNOTATION_MARKER)
    return rest[:index] if index > 0 else None


def dataset_remote_name(filename: str | Path) -> str:
    """Map a dataset filename to its full path in the bucket.

    Graph files and their summaries go to the ball root; annotation files and their
    summaries go under a ``{moveset}`` subfolder. A name that does not follow the
    ``*_rank{R}[_rel{N}].*`` convention is returned unchanged, so a hand-named upload
    still lands at the bucket root.
    """
    name = Path(filename).name
    match = _RANK_RE.match(name)
    if match is None:
        return name
    rank = int(match.group("rank"))
    rel = int(match.group("rel")) if match.group("rel") else 0
    folder = ball_remote_dir(rank, rel)
    moveset = _moveset_of(match.group("rest"))
    return f"{folder}/{moveset}/{name}" if moveset else f"{folder}/{name}"
