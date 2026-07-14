"""Assign every group in a dataset to the train, validation, or test split.

The split lives in its own ``<name>.split.json`` next to the group file, alongside
the annotation files, and syncs to the Hugging Face bucket the same way. Keeping it
separate means a split is stated once and read identically by every consumer, rather
than each training run inventing its own shuffle and quietly evaluating on groups it
trained on.

A group's split is a deterministic function of its content hash -- ``sha256(salt ||
hash)`` folded into 10,000 buckets and cut at the configured ratios. Two properties
follow, and both matter for a database that only ever grows:

* **Stable under growth.** A group's assignment depends on nothing but its own hash,
  so re-running ``dataset split`` after a ``dataset grow`` assigns the new groups and
  cannot move an existing one out of the split it was evaluated on.
* **Reproducible.** The same dataset and salt regenerate the same file byte for byte,
  from scratch, on any machine -- so the file is an artifact to audit, not a secret.

The hash is uniform, so the ratios come out on the whole population; there is no
stratification by length or distance. What the split does *not* do is separate the
graph: validation groups are neighbours of training groups, because in a dense Cayley
graph grown from one root they unavoidably are. The split measures generalization to
unseen *groups*, not to an unseen region.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ac_zero.datasets.io import atomic_write_json
from ac_zero.datasets.json_stream import iter_json_array

SCHEMA_VERSION = "aczero-split-v1"
_GROUPS_SUFFIX = ".groups.json"
# Bucket resolution: ratios are honoured to one part in ten thousand.
_BUCKETS = 10_000

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
# The integer each split is stored as in the supervised sidecar's split column.
SPLIT_CODES: dict[str, int] = {name: code for code, name in enumerate(SPLITS)}


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """The ratios and salt that define one split of a dataset."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    # Changing the salt reshuffles every group into a fresh split -- which invalidates
    # any model already evaluated against the old one. It exists so a second,
    # independent split can be drawn deliberately, not as a knob to turn casually.
    salt: str = "aczero-split-v1"

    def validate(self) -> None:
        if min(self.train, self.val, self.test) < 0.0:
            raise ValueError("split ratios must be non-negative")
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        if self.val <= 0.0 or self.test <= 0.0:
            raise ValueError("the val and test splits must each get a positive share")

    @property
    def cuts(self) -> tuple[int, int]:
        """Bucket boundaries `(train_end, val_end)` in ``[0, _BUCKETS)``."""
        train_end = round(self.train * _BUCKETS)
        return train_end, train_end + round(self.val * _BUCKETS)


@dataclass(frozen=True, slots=True)
class SplitReport:
    """How many groups landed in each split."""

    path: str
    total: int
    train: int
    val: int
    test: int


def split_path(groups_path: str | Path) -> Path:
    """Derive the split filename ``<base>.split.json`` from a group dataset's path."""
    groups = Path(groups_path)
    name = groups.name
    base = name[: -len(_GROUPS_SUFFIX)] if name.endswith(_GROUPS_SUFFIX) else groups.stem
    return groups.with_name(f"{base}.split.json")


def assign(content_hash: str, config: SplitConfig) -> Split:
    """Return the split a group belongs to, from its content hash alone."""
    digest = hashlib.sha256(f"{config.salt}:{content_hash}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % _BUCKETS
    train_end, val_end = config.cuts
    if bucket < train_end:
        return "train"
    return "val" if bucket < val_end else "test"


def write_split(groups_path: str | Path, config: SplitConfig) -> SplitReport:
    """Assign every group in ``groups_path`` and write the split file beside it.

    The group file is streamed rather than parsed, so a multi-gigabyte dataset splits
    in bounded memory.
    """
    config.validate()
    groups = Path(groups_path)
    assignments = [
        {"hash": entry["hash"], "split": assign(entry["hash"], config)}
        for entry in iter_json_array(groups, "groups")
    ]
    if not assignments:
        raise ValueError(f"{groups}: dataset has no groups to split")
    counts = Counter(str(entry["split"]) for entry in assignments)
    destination = split_path(groups)
    atomic_write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "salt": config.salt,
            "ratios": {"train": config.train, "val": config.val, "test": config.test},
            "assignments": assignments,
            "provenance": {
                "source": groups.name,
                "count": len(assignments),
                **{name: counts.get(name, 0) for name in SPLITS},
            },
        },
    )
    return SplitReport(
        path=str(destination),
        total=len(assignments),
        train=counts.get("train", 0),
        val=counts.get("val", 0),
        test=counts.get("test", 0),
    )
