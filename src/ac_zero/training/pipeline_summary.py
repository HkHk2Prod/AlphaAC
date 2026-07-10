"""Output scaffolding for a training run: its metrics rows, directory layout, and summary.

Split out of :mod:`ac_zero.training.pipeline` so the run orchestrator stays focused
on control flow rather than the shapes of what it writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# One row of the run's metrics.jsonl / event payloads: a flat map of scalars.
MetricsRow = dict[str, float | int | bool | str]


@dataclass(frozen=True, slots=True)
class TrainingPipelineSummary:
    """High-level result of the config-driven training pipeline."""

    run_directory: str
    checkpoint_path: str
    certificate_path: str
    model_name: str
    checkpoint_name: str
    checkpoint_bundle_dir: str
    run_id: str
    best_return: float | None
    iterations: int
    episodes: int
    replay_size: int
    optimizer_updates: int
    final_total_loss: float
    checkpoint_restored: bool
    certificate_verified: bool
    event_log_path: str
    progress_log_path: str
    live_graph_path: str
    final_graph_path: str
    plot_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RunDirectories:
    """The output subdirectories of a single training run."""

    run: Path
    checkpoints: Path
    certificates: Path
    artifacts: Path
    logs: Path

    @classmethod
    def create(cls, run_directory: str) -> _RunDirectories:
        run = Path(run_directory)
        dirs = cls(
            run=run,
            checkpoints=run / "checkpoints",
            certificates=run / "certificates",
            artifacts=run / "artifacts",
            logs=run / "logs",
        )
        for directory in (dirs.checkpoints, dirs.certificates, dirs.artifacts, dirs.logs):
            directory.mkdir(parents=True, exist_ok=True)
        return dirs
