from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Typed subset of experiment configuration used by smoke workflows."""

    rank: int = 2
    max_moves: int = 8
    total_length_cap: int = 128
    model: str = "residual_mlp"

    def validate(self) -> None:
        """Reject basic impossible environment/model settings early."""
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.max_moves < 0:
            raise ValueError("max_moves must be non-negative")
        if self.total_length_cap <= 0:
            raise ValueError("total_length_cap must be positive")
