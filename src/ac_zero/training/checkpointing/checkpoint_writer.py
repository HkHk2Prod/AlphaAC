"""Writes a run's checkpoints: the legacy ``latest.json`` plus the HF bundle.

Owns the two checkpoint sinks a run keeps in step -- the backward-compatible
``checkpoints/latest.json`` (read by the notebook report and older tooling) and
the best-by-metric :class:`CheckpointBundle` under ``model_checkpoint/`` -- and
builds the self-describing payload (config + model weights + optimizer step) and
the provenance meta the two share. Keeping this here leaves the training pipeline
to decide *when* to checkpoint, not *how* to serialize one.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.training.checkpointing.checkpoint_bundle import CheckpointBundle
from ac_zero.training.checkpointing.checkpointing import CheckpointManager
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
from ac_zero.training.ppo.losses import PolicyValueLoss

_SCHEMA_VERSION = "aczero-training-checkpoint-v1"


class RunCheckpointer:
    """Serializes checkpoints for one run to both ``latest.json`` and the bundle."""

    def __init__(
        self,
        run_dir: Path,
        *,
        config: TrainingPipelineConfig,
        checkpoint_name: str,
        run_id: str,
        seed: int,
        warm_started_from: str | None,
    ) -> None:
        self._legacy = CheckpointManager(run_dir / "checkpoints")
        self.bundle = CheckpointBundle(run_dir / "model_checkpoint")
        self._config = config
        self.checkpoint_name = checkpoint_name
        self.run_id = run_id
        self._seed = seed
        self.warm_started_from = warm_started_from

    @property
    def best_metric(self) -> float | None:
        """The best metric seen so far (mirrors the bundle's best model)."""
        return self.bundle.best_metric

    def load_latest(self) -> dict[str, Any]:
        """Read back the last-written ``latest.json`` payload."""
        return self._legacy.load_json("latest")

    def save(
        self,
        *,
        model: TrainablePolicyValueModel,
        iteration: int,
        optimizer_step: int,
        loss: PolicyValueLoss,
        replay_size: int,
        metric: float | None,
        mean_return: float,
        success_rate: float,
        metrics_rows: list[dict[str, Any]],
        learning_state: dict[str, Any],
    ) -> Path:
        """Write ``latest.json`` and refresh the bundle (best/latest/metrics/meta)."""
        payload = self._payload(
            model,
            iteration,
            optimizer_step,
            loss,
            replay_size,
            metric,
            mean_return,
            success_rate,
            learning_state,
        )
        path = self._legacy.save_json("latest", payload)
        self.bundle.save_checkpoint(payload, metric=metric)
        self.bundle.save_metrics(metrics_rows)
        self.bundle.save_meta(self._meta(iteration, optimizer_step, mean_return, success_rate))
        return path

    def _payload(
        self,
        model: TrainablePolicyValueModel,
        iteration: int,
        optimizer_step: int,
        loss: PolicyValueLoss,
        replay_size: int,
        metric: float | None,
        mean_return: float,
        success_rate: float,
        learning_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "seed": self._seed,
            "iteration": iteration,
            "config": asdict(self._config),
            "model_state": model.to_json(),
            "optimizer_state": {
                "step": optimizer_step,
                "learning_rate": self._config.learning_rate,
            },
            # Adaptive across-episode state (shaping alpha + distance curriculum)
            # so a warm-started run resumes them continuously instead of resetting
            # to config initials. Empty when neither mechanism is active.
            "learning_state": learning_state,
            "replay_size": replay_size,
            "loss": asdict(loss),
            # Metrics that let a cross-run rollup rank this checkpoint.
            "checkpoint_metric": metric,
            "mean_return": mean_return,
            "success_rate": success_rate,
        }

    def _meta(
        self, iteration: int, optimizer_step: int, mean_return: float, success_rate: float
    ) -> dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "run_id": self.run_id,
            "seed": self._seed,
            "iteration": iteration,
            "optimizer_step": optimizer_step,
            "best_metric": self.bundle.best_metric,
            "mean_return": mean_return,
            "success_rate": success_rate,
            "warm_started_from": self.warm_started_from,
            "updated_at": int(time.time()),
            "config": asdict(self._config),
        }
