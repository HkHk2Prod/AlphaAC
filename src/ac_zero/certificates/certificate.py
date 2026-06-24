from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import PrimitiveMove


@dataclass(frozen=True, slots=True)
class Certificate:
    """Self-contained replay artifact for a claimed AC trivialization.

    Certificates store primitive moves and intermediate hashes so verification
    can replay the transformation without loading a model, trainer, or search
    implementation.
    """

    schema_version: str
    presentation_id: str | None
    initial_presentation: BalancedPresentation
    action_catalog_version: str
    moves: tuple[PrimitiveMove, ...]
    intermediate_hashes: tuple[str, ...]
    final_presentation: BalancedPresentation
    goal_mode: str
    success: bool
    experiment_id: str
    seed: int

    @classmethod
    def from_path(cls, path: str | Path) -> Certificate:
        """Load a canonical JSON certificate from disk."""
        from ac_zero.moves.primitive import move_from_json

        data = json.loads(Path(path).read_text())
        return cls(
            schema_version=data["schema_version"],
            presentation_id=data.get("presentation_id"),
            initial_presentation=BalancedPresentation.from_json(data["initial_presentation"]),
            action_catalog_version=data["action_catalog_version"],
            moves=tuple(move_from_json(m) for m in data["moves"]),
            intermediate_hashes=tuple(data["intermediate_presentation_hashes"]),
            final_presentation=BalancedPresentation.from_json(data["final_presentation"]),
            goal_mode=data["goal_mode"],
            success=bool(data["success"]),
            experiment_id=data.get("experiment_id", "unknown"),
            seed=int(data.get("seed", 0)),
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize the certificate with derived length statistics."""
        initial_len = self.initial_presentation.total_length
        lengths = [initial_len]
        pres = self.initial_presentation
        for move in self.moves:
            pres = move.apply(pres)
            lengths.append(pres.total_length)
        min_len = min(lengths)
        return {
            "schema_version": self.schema_version,
            "presentation_id": self.presentation_id,
            "initial_presentation": self.initial_presentation.to_json(),
            "initial_presentation_hash": self.initial_presentation.content_hash,
            "rank": self.initial_presentation.rank,
            "action_catalog_version": self.action_catalog_version,
            "moves": [m.to_json() for m in self.moves],
            "intermediate_presentation_hashes": list(self.intermediate_hashes),
            "final_presentation": self.final_presentation.to_json(),
            "final_presentation_hash": self.final_presentation.content_hash,
            "goal_mode": self.goal_mode,
            "number_of_moves": len(self.moves),
            "initial_total_length": initial_len,
            "minimum_total_length_reached": min_len,
            "step_of_minimum": lengths.index(min_len),
            "success": self.success,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "resolved_configuration_hash": "unavailable-smoke",
            "dataset_checksum": "unavailable-smoke",
            "git_commit": "unavailable",
            "dirty_working_tree": True,
            "uv_lock_checksum": "unavailable",
            "package_versions": {},
            "hardware_metadata": {},
        }

    def write(self, path: str | Path) -> None:
        """Write canonical pretty-printed JSON to `path`."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")


def build_certificate(
    initial: BalancedPresentation,
    moves: tuple[PrimitiveMove, ...],
    *,
    goal_mode: str,
    experiment_id: str,
    seed: int,
) -> Certificate:
    """Build a certificate by replaying moves and recording intermediate hashes."""
    pres = initial
    hashes: list[str] = []
    for move in moves:
        pres = move.apply(pres)
        hashes.append(pres.content_hash)
    return Certificate(
        schema_version="aczero-certificate-v1",
        presentation_id=initial.presentation_id,
        initial_presentation=initial,
        action_catalog_version=ActionCatalog(initial.rank).version,
        moves=moves,
        intermediate_hashes=tuple(hashes),
        final_presentation=pres,
        goal_mode=goal_mode,
        success=True,
        experiment_id=experiment_id,
        seed=seed,
    )
