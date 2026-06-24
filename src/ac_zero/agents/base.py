from __future__ import annotations

from dataclasses import dataclass, field

from ac_zero.algebra.presentation import BalancedPresentation


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Shared result object returned by search algorithms and agents.

    `path` stores stable action IDs, not move objects, so results can be logged
    compactly and replayed against the deterministic `ActionCatalog`.
    """

    best_state: BalancedPresentation
    best_reduction: int
    path: tuple[int, ...]
    expanded_nodes: int
    generated_nodes: int
    peak_frontier_size: int
    termination_reason: str
    success: bool
    certificate_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class Agent:
    """Minimal legal-action selection interface for online agents."""

    def select_action(self, mask: tuple[bool, ...]) -> int:
        """Choose one legal action ID from a boolean mask."""
        raise NotImplementedError
