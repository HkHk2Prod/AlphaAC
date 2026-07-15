"""The supervised pretraining run: epochs of Adam over the dataset's descent labels.

Structurally this is the same run as the RL backends -- same run directory, event log,
metrics rows, plots, and Hugging Face checkpoint bundle -- with self-play replaced by a
labelled dataset. Keeping the artifacts identical is the point: the ``best.json`` this
writes is exactly what ``training.warm_start`` in an AlphaZero or PPO config consumes,
so pretraining a small model and fine-tuning it with RL is two configs and no glue.

An epoch is ``optimizer_updates`` minibatches, scored afterwards on a fixed sample of
the validation split. The best checkpoint is the one with the highest validation
descent accuracy. The test split is touched exactly once, after the final epoch.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, cast

from ac_zero.datasets.instance_store import InstanceStore
from ac_zero.datasets.split import split_path
from ac_zero.datasets.supervised_store import SupervisedStore
from ac_zero.encoding.padded import StateEncoder
from ac_zero.models.registry import create_trainable_model, model_from_json
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.moves.universal import moveset_catalog
from ac_zero.training.checkpointing.checkpoint_name import derive_checkpoint_name
from ac_zero.training.checkpointing.checkpoint_writer import RunCheckpointer
from ac_zero.training.logging.callbacks import CallbackManager, default_training_callbacks
from ac_zero.training.logging.events import LogLevel
from ac_zero.training.pipeline.instance_source import require_dataset_bound
from ac_zero.training.pipeline.pipeline_artifacts import render_plots, write_fixture_certificate
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig, run_description
from ac_zero.training.pipeline.pipeline_summary import (
    MetricsRow,
    TrainingPipelineSummary,
    _RunDirectories,
)
from ac_zero.training.ppo.losses import PolicyValueLoss
from ac_zero.training.supervised.batches import LabelledBatch, SupervisedBatches
from ac_zero.training.supervised.supervised import SupervisedLoss, SupervisedTrainer


def run_supervised_pipeline(
    config: TrainingPipelineConfig,
    seed: int,
    callbacks: CallbackManager | None = None,
) -> TrainingPipelineSummary:
    """Pretrain a policy/value model on the dataset's known descent directions."""
    return _SupervisedRun(config, seed, callbacks).execute()


class _SupervisedRun:
    """One supervised run: labelled epochs, validation, and the usual artifacts."""

    def __init__(
        self, config: TrainingPipelineConfig, seed: int, callbacks: CallbackManager | None
    ) -> None:
        config.validate()
        self.seed = seed
        self.rng = random.Random(seed)
        self.dirs = _RunDirectories.create(config.run_directory)
        self.manager = callbacks or default_training_callbacks(
            self.dirs.run, verbosity=config.verbosity
        )
        groups = Path(str(config.dataset_path))
        annotations = Path(str(config.dataset_annotations_path))
        split_file = _split_file(config, groups)
        # The dataset has to have been generated under this run's bound: its labels point
        # down descents, and a descent is only a descent in the graph it was proven in.
        require_dataset_bound(groups, config.max_relator_tokens)

        # The capacity decides which moves the environment would let the model play, so
        # the labels are built for it: it is a source of the sidecar, not a reader of it.
        # Building the sidecar over a large ball streams gigabytes and applies a move to
        # every group, so it reports progress and fans that work out across `workers`.
        def on_build(message: str, metrics: dict[str, Any]) -> None:
            self.manager.emit(0, "sidecar", message, metrics)

        self.labels = SupervisedStore.open(
            groups,
            annotations,
            split_file,
            config.moveset,
            config.max_relator_tokens,
            workers=config.workers,
            progress=on_build,
        )
        self.instances = InstanceStore.open(groups, annotations, progress=on_build)
        self.config = config
        self.encoder = StateEncoder(self.config.max_relator_tokens)
        self.model = self._build_model()
        self.warm_started_from = self._warm_start()
        self.batches = SupervisedBatches(
            self.instances,
            self.labels,
            self.encoder,
            temperature=self.config.target_temperature,
            gamma=self.config.gamma,
            catalog_version=moveset_catalog(self.config.moveset, self.labels.rank).version,
        )
        self.trainer = SupervisedTrainer(
            self.model,
            self.batches,
            actions=self.labels.actions,
            learning_rate=self.config.learning_rate,
            value_loss_weight=self.config.value_loss_weight,
            grad_clip=self.config.grad_clip,
        )
        self.checkpointer = RunCheckpointer(
            self.dirs.run,
            config=self.config,
            checkpoint_name=self.config.checkpoint_name or derive_checkpoint_name(self.config),
            run_id=f"{int(time.time())}-{seed}",
            warm_started_from=self.warm_started_from,
            seed=seed,
        )
        self.metrics_rows: list[MetricsRow] = []
        self.optimizer_step = 0
        self.completed_epochs = 0
        self.final_loss = SupervisedLoss(0.0, 0.0, 0.0)
        self.best_accuracy: float | None = None
        # Drawn once in `execute`, so every epoch scores the same validation groups.
        self.validation: list[LabelledBatch] = []
        self.deadline = (
            None
            if self.config.time_limit_s is None
            else time.monotonic() + self.config.time_limit_s
        )

    def _build_model(self) -> TrainablePolicyValueModel:
        return create_trainable_model(
            self.config.model,
            seed=self.seed,
            device=self.config.device,
            # The sizes are architecture hyperparameters, never a `device` override.
            **cast(dict[str, Any], self.config.model_config),
        )

    def _warm_start(self) -> str | None:
        """Initialize from a prior checkpoint -- a longer pretraining lineage."""
        if not self.config.warm_start:
            return None
        data = json.loads(Path(self.config.warm_start).read_text(encoding="utf-8"))
        self.model = model_from_json(data.get("model_state", data), device=self.config.device)
        return self.config.warm_start

    def _describe_data(self) -> dict[str, float | int | bool | str]:
        return {
            "groups": self.labels.count,
            "actions": self.labels.actions,
            "moveset": self.labels.moveset,
            "train": self.batches.size("train"),
            "val": self.batches.size("val"),
            "test": self.batches.size("test"),
            "max_relator_tokens": self.encoder.max_relator_tokens,
            "labels": str(self.labels.path),
        }

    def _epoch(self, epoch: int) -> None:
        """Run one epoch of minibatch updates, then score the validation sample."""
        for _ in range(self.config.optimizer_updates):
            self.final_loss = self.trainer.step("train", self.config.batch_size, self.rng)
            self.optimizer_step += 1
        metrics = self.trainer.evaluate(self.validation)
        row: MetricsRow = {
            "iteration": epoch,
            "optimizer_step": self.optimizer_step,
            "batch_size": self.config.batch_size,
            "policy_loss": self.final_loss.policy_loss,
            "value_loss": self.final_loss.value_loss,
            "total_loss": self.final_loss.total_loss,
            **metrics.as_metrics("val"),
        }
        self.metrics_rows.append(row)
        self.manager.emit(epoch, "epoch", "trained on labelled groups", row)
        self.best_accuracy = (
            metrics.descent_accuracy
            if self.best_accuracy is None
            else max(self.best_accuracy, metrics.descent_accuracy)
        )
        self._save_checkpoint(epoch, metrics.descent_accuracy)

    def _save_checkpoint(self, epoch: int, accuracy: float) -> None:
        if epoch % self.config.checkpoint_every:
            return
        self.checkpointer.save(
            model=self.model,
            iteration=epoch,
            optimizer_step=self.optimizer_step,
            loss=PolicyValueLoss(
                policy_loss=self.final_loss.policy_loss,
                value_loss=self.final_loss.value_loss,
                total_loss=self.final_loss.total_loss,
            ),
            replay_size=0,  # supervised training reads a dataset; it fills no replay
            # The best model is the one that most often picks a move that actually
            # reduces the distance to the origin -- not the one with the lowest loss.
            metric=accuracy,
            mean_return=0.0,
            success_rate=accuracy,
            metrics_rows=self.metrics_rows,
            learning_state={},
        )

    def _write_metrics(self) -> None:
        (self.dirs.run / "metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.metrics_rows),
            encoding="utf-8",
        )

    def execute(self) -> TrainingPipelineSummary:
        try:
            self.manager.emit(
                0,
                "start",
                "starting supervised pretraining",
                run_description(self.config, self.seed, self.model.architecture),
            )
            self.manager.emit(1, "dataset", "labelled group dataset", self._describe_data())
            # Drawn once, so every epoch is scored against the same groups and a moved
            # metric is a moved model.
            self.validation = self.trainer.sample_batches(
                "val", self.config.eval_batches, self.config.batch_size, self.seed
            )
            for epoch in range(1, self.config.iterations + 1):
                self._epoch(epoch)
                self.completed_epochs = epoch
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.manager.emit(
                        epoch,
                        "budget",
                        "wall-clock budget spent; stopping at epoch boundary",
                        {"epoch": epoch, "optimizer_step": self.optimizer_step},
                    )
                    break
            self.manager.emit(
                self.completed_epochs + 1,
                "model",
                "trained model size",
                {"parameters": self.model.parameter_count},
            )
            test = self.trainer.evaluate(self.batches.epoch("test", self.config.batch_size))
            self.manager.emit(
                self.completed_epochs + 2,
                "test",
                "held-out test split",
                test.as_metrics("test"),
                level=LogLevel.INFO,
            )
            self._write_metrics()
            certificate_path = self.dirs.certificates / "example.json"
            verified = write_fixture_certificate(self.config, self.seed, certificate_path)
            plots = render_plots(
                self.metrics_rows, self.dirs.artifacts, self.manager, self.completed_epochs + 3
            )
            return TrainingPipelineSummary(
                run_directory=str(self.dirs.run),
                checkpoint_path=str(self.dirs.checkpoints / "latest.json"),
                certificate_path=str(certificate_path),
                model_name=self.model.architecture,
                checkpoint_name=self.checkpointer.checkpoint_name,
                checkpoint_bundle_dir=str(self.checkpointer.bundle.directory),
                run_id=self.checkpointer.run_id,
                best_return=self.checkpointer.best_metric,
                iterations=self.completed_epochs,
                episodes=0,
                replay_size=0,
                optimizer_updates=self.optimizer_step,
                final_total_loss=self.final_loss.total_loss,
                checkpoint_restored=self.warm_started_from is not None,
                certificate_verified=verified,
                event_log_path=str(self.dirs.logs / "training_events.jsonl"),
                progress_log_path=str(self.dirs.logs / "progress.log"),
                live_graph_path=str(self.dirs.artifacts / "live_graphs.txt"),
                final_graph_path=str(self.dirs.artifacts / "final_graphs.txt"),
                plot_paths=plots,
            )
        finally:
            self.manager.close()


def _split_file(config: TrainingPipelineConfig, groups: Path) -> Path:
    """Locate the split file, defaulting to the one beside the group dataset."""
    configured = config.dataset_split_path
    path = Path(configured) if configured else split_path(groups)
    if not path.exists():
        raise FileNotFoundError(
            f"the supervised stage needs a train/val/test split at {path}; "
            f"create it with `aczero dataset split --input {groups}`"
        )
    return path
