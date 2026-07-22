"""Tests for supervised pretraining: move targets, the optimizer, and a whole run."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from ac_zero.datasets.annotate import AnnotateConfig, annotate, annotation_path
from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.datasets.instance_store import InstanceStore
from ac_zero.datasets.split import SplitConfig, split_path, write_split
from ac_zero.datasets.supervised_store import DELTA_UNKNOWN, SupervisedStore
from ac_zero.encoding.padded import StateEncoder
from ac_zero.models.registry import create_trainable_model
from ac_zero.moves.universal import moveset_catalog
from ac_zero.training.pipeline.pipeline import run_training_pipeline
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
from ac_zero.training.supervised.batches import SupervisedBatches, policy_targets
from ac_zero.training.supervised.supervised import SupervisedTrainer

MOVESET = "strict-ac"
# The dataset is generated under the run's relator bound; a run whose max_relator_tokens
# differs is refused, because the dataset's descents are only descents in the graph that
# bound defines.
BOUND = 6


def _dataset(tmp_path: Path, target: int = 300) -> Path:
    groups = tmp_path / "toy.groups.json"
    grow_dataset(groups, GrowConfig(rank=2, target=target, max_relator_length=BOUND, workers=1))
    annotate(groups, AnnotateConfig(moveset=MOVESET, workers=1))
    write_split(groups, SplitConfig())
    return groups


def _batches(groups: Path, temperature: float = 1.0) -> SupervisedBatches:
    annotations = annotation_path(groups, MOVESET)
    labels = SupervisedStore.open(groups, annotations, split_path(groups), MOVESET, BOUND)
    return SupervisedBatches(
        InstanceStore.open(groups, annotations),
        labels,
        StateEncoder(BOUND),
        temperature=temperature,
        gamma=0.99,
        catalog_version=moveset_catalog(MOVESET, labels.rank).version,
    )


def _config(tmp_path: Path, groups: Path, **overrides: object) -> TrainingPipelineConfig:
    settings: dict[str, object] = {
        "rank": 2,
        "agent": "supervised",
        "model": "linear_policy_value",
        "moveset": MOVESET,
        "max_relator_tokens": BOUND,  # the bound the dataset was generated under
        "dataset_path": str(groups),
        "dataset_annotations_path": str(annotation_path(groups, MOVESET)),
        "dataset_split_path": str(split_path(groups)),
        "iterations": 2,
        "optimizer_updates": 3,
        "batch_size": 8,
        "eval_batches": 2,
        "learning_rate": 0.01,
        "run_directory": str(tmp_path / "run"),
    }
    settings.update(overrides)
    return TrainingPipelineConfig(**settings)  # type: ignore[arg-type]


# -- targets ---------------------------------------------------------------


def test_the_target_ranks_moves_by_what_they_do_to_the_distance() -> None:
    """A descent outranks a stall, which outranks a climb; an unknown move gets nothing."""
    deltas = np.asarray([[-1, 0, 1, DELTA_UNKNOWN]], dtype=np.int16)
    target = policy_targets(deltas, temperature=1.0)[0]

    assert target[0] > target[1] > target[2] > 0.0
    assert target[3] == 0.0
    assert float(target.sum()) == pytest.approx(1.0)


def test_co_optimal_descents_share_the_mass_equally() -> None:
    deltas = np.asarray([[-1, -1, 2]], dtype=np.int16)
    target = policy_targets(deltas, temperature=1.0)[0]
    assert target[0] == pytest.approx(target[1])


def test_a_low_temperature_sharpens_the_target_onto_the_descents() -> None:
    deltas = np.asarray([[-1, 0, 1]], dtype=np.int16)
    sharp = policy_targets(deltas, temperature=0.05)[0]
    soft = policy_targets(deltas, temperature=2.0)[0]

    assert sharp[0] == pytest.approx(1.0, abs=1e-6)
    assert sharp[0] > soft[0]
    assert soft[2] > sharp[2]


def test_targets_are_a_distribution_over_every_row() -> None:
    deltas = np.asarray(
        [[-1, 0, DELTA_UNKNOWN], [DELTA_UNKNOWN, 3, 3], [-1, -1, -1]], dtype=np.int16
    )
    targets = policy_targets(deltas, temperature=1.0)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert np.all(targets[deltas == DELTA_UNKNOWN] == 0.0)


# -- batches ---------------------------------------------------------------


def test_a_batch_pairs_each_group_with_its_own_labels(tmp_path: Path) -> None:
    batches = _batches(_dataset(tmp_path))
    batch = batches.sample("train", 8, random.Random(0))

    assert batch.size == 8
    assert batch.policy_targets.shape == (8, 12)
    assert batch.deltas.shape == (8, 12)
    assert np.allclose(batch.policy_targets.sum(axis=1), 1.0)
    # Every group is a real problem short of the origin, so its success target
    # gamma**d is below the origin's 1.0, and its progress target B~ is positive.
    assert np.all(batch.success_targets < 1.0)
    assert np.all(batch.success_targets > 0.0)
    assert np.all(batch.progress_targets > 0.0)


def test_the_head_targets_track_the_distance_to_the_origin(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    batches = _batches(groups)
    labels = SupervisedStore.open(
        groups, annotation_path(groups, MOVESET), split_path(groups), MOVESET, 0
    )
    rows = labels.trainable("train")
    near = int(rows[np.argmin(labels.distances[rows])])
    far = int(rows[np.argmax(labels.distances[rows])])

    batch = batches.rows([near, far])
    d_near = labels.distances[near]
    assert d_near < labels.distances[far]
    # success = gamma**(d-1) falls off with distance; the nearer group is more likely
    # to reach the origin. The goal reward lands on the d-th transition, discounted
    # from the start by gamma**(d-1) under the RL collectors' convention.
    assert batch.success_targets[0] > batch.success_targets[1]
    assert batch.success_targets[0] == pytest.approx(0.99 ** (d_near - 1))
    # progress = B~ = (1 - gamma**d) / ((1 - gamma) * d), the descent's normalized
    # shaping return.
    assert batch.progress_targets[0] == pytest.approx(
        (1.0 - 0.99**d_near) / ((1.0 - 0.99) * d_near)
    )


def test_an_epoch_sweep_sees_every_group_in_the_split_once(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    batches = _batches(groups)
    seen = sum(batch.size for batch in batches.epoch("test", 4))
    assert seen == batches.size("test")


# -- trainer ---------------------------------------------------------------


def _trainer(groups: Path) -> SupervisedTrainer:
    return SupervisedTrainer(
        create_trainable_model("linear_policy_value", seed=0),
        _batches(groups),
        actions=12,
        learning_rate=0.05,
        value_loss_weight=1.0,
        grad_clip=1.0,
    )


def test_training_reduces_the_loss_on_the_data_it_is_shown(tmp_path: Path) -> None:
    trainer = _trainer(_dataset(tmp_path))
    rng = random.Random(0)
    first = trainer.step("train", 32, rng)
    for _ in range(40):
        last = trainer.step("train", 32, rng)
    assert last.total_loss < first.total_loss
    assert last.policy_loss > 0.0


def test_evaluation_scores_the_move_the_model_actually_picks(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    trainer = _trainer(groups)
    batches = trainer.sample_batches("val", 3, 16, seed=0)
    metrics = trainer.evaluate(batches)

    assert metrics.groups == 48
    assert 0.0 <= metrics.descent_accuracy <= 1.0
    assert 0.0 <= metrics.unknown_rate <= 1.0
    assert metrics.policy_loss > 0.0
    # The picked move's delta is a real distance change, so it sits in a sane range.
    assert -1.0 <= metrics.mean_delta <= 10.0
    assert set(metrics.as_metrics("val")) >= {"val_descent_accuracy", "val_mean_delta"}


def test_the_validation_sample_is_the_same_on_every_epoch(tmp_path: Path) -> None:
    """A moved metric must mean a moved model, not a freshly drawn sample."""
    trainer = _trainer(_dataset(tmp_path))
    first = trainer.sample_batches("val", 2, 8, seed=7)
    second = trainer.sample_batches("val", 2, 8, seed=7)
    assert np.array_equal(first[0].deltas, second[0].deltas)


def test_training_improves_the_descent_accuracy(tmp_path: Path) -> None:
    """The whole point: after fitting, the top-ranked move reduces the distance more often."""
    trainer = _trainer(_dataset(tmp_path, target=600))
    validation = trainer.sample_batches("val", 4, 32, seed=0)
    before = trainer.evaluate(validation).descent_accuracy

    rng = random.Random(0)
    for _ in range(150):
        trainer.step("train", 64, rng)
    after = trainer.evaluate(validation)

    assert after.descent_accuracy > before
    assert after.mean_delta < 0.0  # the average pick now moves toward the origin


def test_evaluating_nothing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no batches"):
        _trainer(_dataset(tmp_path)).evaluate([])


# -- the whole run ---------------------------------------------------------


def test_a_supervised_run_writes_the_usual_artifacts(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    summary = run_training_pipeline(_config(tmp_path, groups), seed=0)

    assert summary.iterations == 2
    assert summary.optimizer_updates == 6
    assert summary.episodes == 0  # no self-play: this stage reads a dataset
    assert Path(summary.checkpoint_path).exists()
    assert (Path(summary.checkpoint_bundle_dir) / "best.json").exists()
    assert summary.best_return is not None  # the best validation descent accuracy

    rows = [
        json.loads(line)
        for line in Path(summary.run_directory, "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert {"val_descent_accuracy", "val_mean_delta", "val_policy_loss"} <= set(rows[0])


def test_the_checkpoint_warm_starts_an_rl_run(tmp_path: Path) -> None:
    """The pretrain-then-fine-tune path: an RL config loads the supervised weights."""
    groups = _dataset(tmp_path)
    pretrained = run_training_pipeline(_config(tmp_path, groups), seed=0)
    best = Path(pretrained.checkpoint_bundle_dir) / "best.json"

    payload = json.loads(best.read_text())
    assert payload["model_state"]["built"] is True
    # The checkpoint records the capacity it was built with -- without that the fine-tune
    # could not reconstruct the network's input shape, nor know which dataset it may
    # continue training on.
    capacity = payload["config"]["max_relator_tokens"]
    assert capacity == BOUND

    rl = TrainingPipelineConfig(
        rank=2,
        agent="ppo",
        model="linear_policy_value",
        moveset=MOVESET,
        max_relator_tokens=capacity,
        # Seed self-play from the same dataset the model was pretrained on -- the one
        # grown under this very bound. A random scramble could hand the episode a
        # presentation longer than the capacity the pretrained network was built around,
        # which the encoder rightly refuses.
        dataset_path=str(groups),
        dataset_annotations_path=str(annotation_path(groups, MOVESET)),
        iterations=1,
        episodes_per_iteration=1,
        optimizer_updates=1,
        batch_size=2,
        warm_start=str(best),
        run_directory=str(tmp_path / "finetune"),
        workers=1,
    )
    summary = run_training_pipeline(rl, seed=0)
    assert summary.checkpoint_restored


def test_early_stopping_ends_the_run_before_the_epoch_cap(tmp_path: Path) -> None:
    """A `min_delta` no epoch can clear makes every epoch past the first a non-improvement,
    so `patience` of those in a row stops the run well short of `iterations`."""
    groups = _dataset(tmp_path)
    config = _config(
        tmp_path,
        groups,
        iterations=20,
        early_stopping_patience=2,
        early_stopping_min_delta=1.0,  # unreachable: accuracy is in [0, 1]
    )
    summary = run_training_pipeline(config, seed=0)
    # epoch 1 sets the best; epochs 2 and 3 fail to improve -> stop at epoch 3.
    assert summary.iterations == 3
    assert (Path(summary.checkpoint_bundle_dir) / "best.json").exists()


def test_early_stopping_off_runs_every_epoch(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    summary = run_training_pipeline(_config(tmp_path, groups, iterations=4), seed=0)
    assert summary.iterations == 4  # patience defaults to 0, i.e. disabled


def test_early_stopping_settings_are_validated(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    with pytest.raises(ValueError, match="early_stopping_patience must be non-negative"):
        _config(tmp_path, groups, early_stopping_patience=-1).validate()
    with pytest.raises(ValueError, match="early_stopping_min_delta must be non-negative"):
        _config(tmp_path, groups, early_stopping_min_delta=-0.1).validate()


def test_a_supervised_config_needs_its_labels() -> None:
    with pytest.raises(ValueError, match=r"set dataset\.path"):
        TrainingPipelineConfig(agent="supervised").validate()
    with pytest.raises(ValueError, match=r"needs dataset\.annotations"):
        TrainingPipelineConfig(agent="supervised", dataset_path="groups.json").validate()
    with pytest.raises(ValueError, match=r"remove dataset\.max_difficulty"):
        TrainingPipelineConfig(
            agent="supervised",
            dataset_path="g.json",
            dataset_annotations_path="a.json",
            dataset_max_difficulty=4,
        ).validate()


def test_the_encoder_capacity_must_be_stated() -> None:
    """There is no `derive it from the data` any more: the data is generated *for* it."""
    with pytest.raises(ValueError, match="max_relator_tokens must be positive"):
        TrainingPipelineConfig(agent="ppo", max_relator_tokens=0).validate()


def test_a_run_refuses_a_dataset_generated_under_a_different_bound(tmp_path: Path) -> None:
    """The dataset's bound and the model's capacity are one number, and it is checked.

    A ball grown to `BOUND` proves shortest paths through the graph a `BOUND`-token
    encoder moves in. Train a wider model on it and its labels point down descents that
    are not the shortest ones in *its* graph -- so the run refuses to start rather than
    silently learning the wrong target.
    """
    groups = _dataset(tmp_path)
    config = _config(tmp_path, groups, max_relator_tokens=BOUND + 2)
    with pytest.raises(ValueError, match="was generated max_relator_length=6"):
        run_training_pipeline(config, seed=0)


def test_a_supervised_run_refuses_to_start_without_a_split(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    split_path(groups).unlink()
    config = _config(tmp_path, groups, dataset_split_path=str(split_path(groups)))
    with pytest.raises(FileNotFoundError, match="aczero dataset split"):
        run_training_pipeline(config, seed=0)


def test_supervised_settings_are_validated(tmp_path: Path) -> None:
    groups = _dataset(tmp_path)
    with pytest.raises(ValueError, match="target_temperature must be positive"):
        _config(tmp_path, groups, target_temperature=0.0).validate()
    with pytest.raises(ValueError, match="grad_clip must be non-negative"):
        _config(tmp_path, groups, grad_clip=-1.0).validate()
    with pytest.raises(ValueError, match="eval_batches must be positive"):
        _config(tmp_path, groups, eval_batches=0).validate()
