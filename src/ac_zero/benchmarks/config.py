"""Configuration for one benchmark evaluation run.

The knobs exist because the right amount of effort per presentation is not yet
known: the defaults are a starting point, and every budget is overridable from
the Kaggle queue task's ``config:`` block exactly like a training config.

Budgets are stated as *node* budgets rather than seconds. The solvers have no
mid-search time callback, so a wall-clock budget could only be enforced by
killing a search partway -- which throws away the work rather than bounding it.
``max_total_minutes`` is the one wall-clock knob and it is checked at entry
boundaries, so a run stops between presentations with everything it has scored
already recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ac_zero.datasets.hub import DEFAULT_BUCKET

SCAN_AGENT = "greedy-best-first"
DEEP_AGENT = "puct"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Budgets and environment settings for a benchmark run."""

    catalog_path: str = ""
    checkpoint_name: str = ""
    checkpoint_path: str = ""
    bucket: str = DEFAULT_BUCKET

    # Environment: must match what the checkpoint was trained under, since the
    # encoder capacity is also the environment's relator capacity.
    max_relator_tokens: int = 48
    moveset: str = "strict-ac"
    goal_mode: str = "exact_standard"
    max_moves: int = 128

    # Pass 1 -- classical scan over every entry.
    scan_expansions: int = 256
    scan_generated: int = 10_000

    # Pass 2 -- model-guided search over what the scan missed. Skipped entirely
    # when no checkpoint is available, leaving the scan as the whole run.
    deep_simulations: int = 128
    deep_moves: int = 64

    max_total_minutes: float = 0.0  # 0 -> no wall-clock cap

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Build from a queue/YAML mapping, ignoring unrelated keys.

        Accepts the budgets nested under ``benchmark:`` or flat at the top level,
        matching how the training config accepts both shapes.
        """
        merged = {**data, **(data.get("benchmark") or {})}
        fields = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in merged.items() if k in fields})

    @property
    def deadline_seconds(self) -> float | None:
        return self.max_total_minutes * 60 if self.max_total_minutes > 0 else None
