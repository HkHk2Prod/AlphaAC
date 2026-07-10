import json
from pathlib import Path

import pytest

from ac_zero.datasets.generator import generate_solvable
from ac_zero.datasets.groups import MOVE_CATALOG, SCHEMA_VERSION, group_entry
from ac_zero.environment.navigation_reward import AlphaUpdater, EpisodeStats, RewardConfig
from ac_zero.training.callbacks import CallbackManager
from ac_zero.training.events import TrainingEvent
from ac_zero.training.navigation_metrics import fold_alpha, navigation_eval_metrics
from ac_zero.training.pipeline import TrainingPipelineConfig, run_training_pipeline
from ac_zero.training.pipeline_episodes import EpisodeMetrics

_ANNOTATIONS_SCHEMA = "aczero-annotations-v1"


def _annotated_dataset(tmp_path: Path) -> tuple[str, str]:
    """Write a tiny grown dataset whose groups carry small known distances.

    Navigation requires distance annotations, and the distance curriculum starts at
    ``L_max = 2``, so every group is annotated with a distance in ``{1, 2}`` to keep
    the sampler non-empty.
    """
    presentations = [
        generate_solvable(rank=2, depth=1 + i % 2, seed=i).presentation for i in range(6)
    ]
    groups = [group_entry(p, ac_trivial=True, source="universal_expansion") for p in presentations]
    dataset = tmp_path / "train.groups.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "rank": 2,
                "move_catalog": MOVE_CATALOG,
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )
    annotations = tmp_path / "train.strict-ac.annotations.json"
    rows = [
        {
            "hash": p.content_hash,
            "distance_to_origin": 1 + i % 2,
            "optimal_moves_to_origin": [],
            "distance_to_shorter": None,
            "optimal_moves_to_shorter": [],
            "shorter_proven": False,
            "optimal": True,
        }
        for i, p in enumerate(presentations)
    ]
    annotations.write_text(
        json.dumps(
            {
                "schema_version": _ANNOTATIONS_SCHEMA,
                "rank": 2,
                "moveset": "strict-ac",
                "annotations": rows,
            }
        ),
        encoding="utf-8",
    )
    return str(dataset), str(annotations)


class _CapturingSink:
    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


def _episode(progress: float, success: bool, *, start: int = 10) -> EpisodeMetrics:
    min_reached = round(start * (1.0 - progress))
    stats = EpisodeStats(
        start_distance=start,
        min_distance_reached=min_reached,
        final_distance=min_reached,
        success=success,
        length=3,
        revisit_count=1,
        alpha=0.3,
        destination_reward=5.0 if success else 0.0,
        shaping_reward=0.6,
        move_fee=-0.03,
        revisit_fee=-0.02,
        total_reward=5.55 if success else 0.55,
    )
    return EpisodeMetrics(
        total_return=stats.total_reward,
        normalized_return=0.5,
        success=success,
        moves=3,
        nav=stats,
    )


def test_fold_alpha_advances_once_per_episode_in_order() -> None:
    updater = AlphaUpdater(RewardConfig(alpha_initial=0.3, progress_low=0.3, increase_factor=1.1))
    rows = fold_alpha(updater, [_episode(0.1, False), _episode(0.1, False)])
    assert len(rows) == 2
    # Both low-progress episodes push alpha up multiplicatively.
    assert rows[0]["alpha"] == pytest.approx(0.3 * 1.1)
    assert rows[1]["alpha"] == pytest.approx(0.3 * 1.1 * 1.1)
    assert updater.alpha == pytest.approx(0.3 * 1.1 * 1.1)
    for row in rows:
        assert set(row) == {"alpha", "progress_rate", "progress_ema", "success", "success_ema"}


def test_fold_alpha_ignores_episodes_without_nav_stats() -> None:
    updater = AlphaUpdater(RewardConfig())
    plain = EpisodeMetrics(total_return=1.0, normalized_return=1.0, success=True, moves=2)
    assert fold_alpha(updater, [plain]) == []
    assert updater.alpha == RewardConfig().alpha_initial


def test_navigation_eval_metrics_average_the_batch() -> None:
    metrics = navigation_eval_metrics([_episode(0.5, True), _episode(0.1, False)])
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["average_revisit_count"] == pytest.approx(1.0)
    assert metrics["average_destination_reward"] == pytest.approx(2.5)
    assert metrics["average_episode_length"] == pytest.approx(3.0)
    assert set(metrics) == {
        "success_rate",
        "progress_rate",
        "average_min_distance_reached",
        "average_final_distance",
        "average_episode_length",
        "average_revisit_count",
        "average_destination_reward",
        "average_shaping_reward",
        "average_move_fee",
        "average_revisit_fee",
        "average_total_reward",
    }


def test_navigation_eval_metrics_empty_without_nav_episodes() -> None:
    plain = EpisodeMetrics(total_return=1.0, normalized_return=1.0, success=True, moves=2)
    assert navigation_eval_metrics([plain]) == {}


def _navigation_config(tmp_path: Path, agent: str) -> TrainingPipelineConfig:
    dataset_path, annotations_path = _annotated_dataset(tmp_path)
    return TrainingPipelineConfig(
        max_moves=4,
        model="residual_mlp",
        agent=agent,
        mcts_simulations=4,
        iterations=2,
        episodes_per_iteration=2,
        optimizer_updates=2,
        batch_size=2,
        workers=1,
        reward_mode="navigation",
        reward_config=RewardConfig(alpha_initial=0.3),
        dataset_path=dataset_path,
        dataset_annotations_path=annotations_path,
        run_directory=str(tmp_path / f"nav-{agent}"),
    )


@pytest.mark.parametrize("agent", ["alphazero", "ppo"])
def test_navigation_pipeline_emits_alpha_and_eval_metrics(tmp_path: Path, agent: str) -> None:
    config = _navigation_config(tmp_path, agent)
    sink = _CapturingSink()
    summary = run_training_pipeline(config, seed=5, callbacks=CallbackManager((sink,)))
    assert Path(summary.checkpoint_path).exists()

    nav_events = [e for e in sink.events if e.phase == "navigation"]
    assert nav_events, "expected a navigation evaluation-metrics event"
    metrics = nav_events[-1].metrics
    for key in ("success_rate", "progress_rate", "alpha", "progress_ema", "success_ema"):
        assert key in metrics
    # Per-episode alpha logging (item 1 of the reward spec).
    episode_events = [e for e in sink.events if e.phase == "navigation_episode"]
    assert episode_events


def test_config_parses_reward_block_and_validates_navigation() -> None:
    config = TrainingPipelineConfig.from_mapping(
        {
            "reward_mode": "navigation",
            "reward": {"alpha_initial": 0.5, "move_fee_scale": 0.05, "increase_factor": 1.2},
            "dataset": {"path": "train.groups.json", "annotations": "train.annotations.json"},
        }
    )
    assert config.reward_config.alpha_initial == 0.5
    assert config.reward_config.move_fee_scale == 0.05
    assert config.reward_config.increase_factor == 1.2
    # Untouched keys keep their defaults.
    assert config.reward_config.anneal_factor == RewardConfig().anneal_factor
    config.validate()
    # Navigation without distance annotations is rejected: L0 would be undefined.
    with pytest.raises(ValueError, match=r"needs dataset\.annotations"):
        TrainingPipelineConfig(reward_mode="navigation").validate()
    # A bad navigation reward config is rejected by the pipeline validator.
    bad = TrainingPipelineConfig(
        reward_mode="navigation",
        dataset_annotations_path="train.annotations.json",
        reward_config=RewardConfig(ema_rate=0.0),
    )
    with pytest.raises(ValueError, match="ema_rate"):
        bad.validate()


def test_navigation_pipeline_stores_reward_components_in_replay(tmp_path: Path) -> None:
    config = _navigation_config(tmp_path, "alphazero")
    from ac_zero.encoding.padded import StateEncoder
    from ac_zero.models.registry import create_trainable_model
    from ac_zero.training.instance_source import build_instance_source
    from ac_zero.training.pipeline_episodes import collect_episodes

    source = build_instance_source(config)
    model = create_trainable_model(config.model, seed=5)
    collected = collect_episodes(
        config,
        StateEncoder(config.max_word_length),
        model,
        seed=5,
        iteration=1,
        source=source,
        alpha=0.3,
    )
    examples = [example for batch, _ in collected for example in batch]
    assert examples, "expected replay examples"
    assert all(example.components is not None for example in examples)
    for example in examples:
        components = example.components
        assert components is not None
        parts = (
            components.reward_destination
            + components.reward_shaping
            + components.reward_move_fee
            + components.reward_revisit_fee
        )
        assert parts == pytest.approx(components.reward_total)
