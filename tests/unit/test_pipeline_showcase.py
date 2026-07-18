from pathlib import Path

import pytest

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.encoding.padded import StateEncoder
from ac_zero.models.registry import create_trainable_model
from ac_zero.moves.primitive import (
    ConcatRelatorMove,
    ConjugateRelatorMove,
    InvertRelatorMove,
    MultiplyRelatorsMove,
)
from ac_zero.training.logging.callbacks import CallbackManager
from ac_zero.training.logging.events import TrainingEvent, Verbosity
from ac_zero.training.pipeline.instance_source import build_instance_source
from ac_zero.training.pipeline.pipeline import TrainingPipelineConfig, run_training_pipeline
from ac_zero.training.pipeline.pipeline_showcase import (
    EpisodeShowcase,
    ShowcaseEpisode,
    ShowcaseStep,
    format_move,
    render_episode,
)


class _CapturingSink:
    """Collect every emitted event so a test can inspect the run log."""

    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def on_event(self, event: TrainingEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


def _config(tmp_path: Path, **overrides: object) -> TrainingPipelineConfig:
    defaults: dict[str, object] = {
        "scramble_depth": 2,
        "unknown_distance_max_moves": 4,
        "model": "residual_mlp",
        "mcts_simulations": 2,
        "iterations": 1,
        "episodes_per_iteration": 2,
        "optimizer_updates": 1,
        "batch_size": 2,
        "workers": 1,
        "run_directory": str(tmp_path / "train"),
    }
    return TrainingPipelineConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


def _showcase(config: TrainingPipelineConfig, every_hours: float = 3.0) -> EpisodeShowcase:
    return EpisodeShowcase(
        config,
        StateEncoder(config.max_relator_tokens),
        build_instance_source(config),
        every_hours=every_hours,
    )


def test_format_move_renders_each_move_as_its_rewrite_rule() -> None:
    names = ("x1", "x2")
    assert format_move(MultiplyRelatorsMove(0, 1), names) == "AC1 r0 <- r0 r1"
    assert format_move(InvertRelatorMove(1), names) == "AC2 r1 <- r1^-1"
    # AC3 conjugates by a signed generator: `r <- g r g^-1`, so a negative
    # generator conjugates by the inverse letter and closes with the positive one.
    assert format_move(ConjugateRelatorMove(0, 2), names) == "AC3 r0 <- x2 r0 x2^-1"
    assert format_move(ConjugateRelatorMove(0, -1), names) == "AC3 r0 <- x1^-1 r0 x1"
    assert format_move(ConcatRelatorMove(1, 0, "left", True), names) == "CAT r1 <- r0^-1 r1"
    assert format_move(ConcatRelatorMove(1, 0, "right", False), names) == "CAT r1 <- r1 r0"


def test_format_move_uses_the_presentation_generator_names() -> None:
    assert format_move(ConjugateRelatorMove(0, 1), ("a", "b")) == "AC3 r0 <- a r0 a^-1"


def _episode(steps: int, *, solved: bool = True) -> ShowcaseEpisode:
    start = BalancedPresentation.standard(2)
    return ShowcaseEpisode(
        start=start,
        start_distance=3,
        steps=tuple(
            ShowcaseStep(f"AC2 r{index % 2} <- r{index % 2}^-1", "<x1, x2 | x1, x2>", 2)
            for index in range(steps)
        ),
        solved=solved,
        reason="goal" if solved else "horizon",
        best_length=2,
    )


def test_render_episode_lists_every_move_of_a_short_episode() -> None:
    rendered = render_episode(_episode(3), iteration=4)

    assert "iteration 4" in rendered
    assert "distance=3" in rendered
    assert "moves=3" in rendered
    assert "SOLVED" in rendered
    assert rendered.count("AC2") == 3
    assert "  1. AC2 r0 <- r0^-1" in rendered
    assert "elided" not in rendered
    assert "best length reached: 2" in rendered


def test_render_episode_elides_the_middle_of_a_long_episode_and_says_how_much() -> None:
    rendered = render_episode(_episode(60, solved=False), iteration=1)

    # 20 head + 10 tail steps are shown; the 30 in between are named, never dropped
    # silently, and the numbering resumes at the real step index.
    assert "... 30 moves elided ..." in rendered
    assert rendered.count("AC2") == 30
    assert "  20. AC2" in rendered
    assert "  51. AC2" in rendered
    assert "unsolved (horizon)" in rendered


def test_showcase_is_due_on_the_first_check_then_throttled_by_the_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    showcase = _showcase(config, every_hours=3.0)
    model = create_trainable_model(config.model, seed=1)
    manager = CallbackManager(())

    # Owed at the first check so a fresh run shows an episode at its first
    # checkpoint rather than three hours in.
    assert showcase.due() is True
    showcase.show(manager, model, event_id=1, iteration=1, seed=5, alpha=None)
    assert showcase.due() is False
    capsys.readouterr()


@pytest.mark.parametrize("agent", ["alphazero", "ppo"])
def test_showcase_plays_a_full_episode_and_reports_its_shape(tmp_path: Path, agent: str) -> None:
    config = _config(tmp_path, agent=agent)
    showcase = _showcase(config)
    model = create_trainable_model(config.model, seed=1)
    sink = _CapturingSink()

    episode = showcase.show(
        CallbackManager((sink,)),
        model,
        event_id=7,
        iteration=2,
        seed=11,
        alpha=None,
    )

    assert episode.steps, "an episode with legal moves available takes at least one"
    # The horizon for an unannotated scramble is the unknown-distance fallback.
    assert len(episode.steps) <= config.unknown_distance_max_moves
    assert episode.reason in ("goal", "horizon", "no_legal_action")
    assert episode.solved is (episode.reason == "goal")

    (event,) = sink.events
    assert event.phase == "showcase"
    assert event.metrics["iteration"] == 2
    assert event.metrics["moves"] == len(episode.steps)
    assert event.metrics["solved"] is episode.solved


def test_training_run_prints_one_showcase_episode_at_the_first_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Three iterations, each a checkpoint: the three-hour throttle lets exactly the
    # first one through, so a run prints an episode once per interval and no more.
    config = _config(tmp_path, iterations=3, checkpoint_every=1)
    sink = _CapturingSink()
    run_training_pipeline(config, seed=3, callbacks=CallbackManager((sink,)))

    printed = capsys.readouterr().out
    assert printed.count("showcase: self-play episode") == 1
    assert "best length reached:" in printed
    showcase_events = [event for event in sink.events if event.phase == "showcase"]
    assert len(showcase_events) == 1
    # It runs after the checkpoint it accompanies, the event the HF uploader fires on.
    phases = [event.phase for event in sink.events]
    assert phases.index("showcase") == phases.index("checkpoint") + 1


def test_showcase_is_off_when_disabled_or_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_training_pipeline(
        _config(tmp_path / "off", showcase_every_hours=0.0),
        seed=3,
        callbacks=CallbackManager(()),
    )
    assert "showcase" not in capsys.readouterr().out

    run_training_pipeline(
        _config(tmp_path / "quiet", verbosity=Verbosity.QUIET),
        seed=3,
        callbacks=CallbackManager(()),
    )
    assert "showcase" not in capsys.readouterr().out


def test_config_rejects_a_negative_showcase_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="showcase_every_hours"):
        _config(tmp_path, showcase_every_hours=-1.0).validate()


def test_config_reads_the_showcase_interval_from_the_training_block() -> None:
    config = TrainingPipelineConfig.from_mapping({"training": {"showcase_every_hours": 0.5}})
    assert config.showcase_every_hours == 0.5
    assert TrainingPipelineConfig.from_mapping({}).showcase_every_hours == 3.0
