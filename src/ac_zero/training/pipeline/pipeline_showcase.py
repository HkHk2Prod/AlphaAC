"""Periodic self-play showcase: one played-out episode, printed move by move.

The scalar training logs say how well the policy scores; they never say what it
actually *does*. This plays a single extra episode with the current weights every
few hours -- the same cadence as the checkpoint upload -- and prints the rewrite
it walks: each move as a rule (``r0 <- r0 r1``) next to the presentation it
produces. It is a read-only view: the episode feeds no replay buffer and no
optimizer step, so it cannot perturb the run it reports on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment
from ac_zero.models.base import PolicyValueModel
from ac_zero.moves.primitive import (
    ConcatRelatorMove,
    ConjugateRelatorMove,
    InvertRelatorMove,
    MultiplyRelatorsMove,
    PrimitiveMove,
)
from ac_zero.search.puct import PUCTMCTS, PUCTConfig
from ac_zero.training.logging.callbacks import CallbackManager
from ac_zero.training.pipeline.instance_source import InstanceSource
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
from ac_zero.training.pipeline.pipeline_episodes import build_env, episode_distance_and_moves
from ac_zero.training.ppo.losses import masked_softmax, visit_count_policy

# A long episode is elided in the middle rather than dumped whole: the opening and
# closing moves are what a reader learns from, and the horizon (3L+6) can run to
# hundreds of steps on a far problem.
_HEAD_STEPS = 20
_TAIL_STEPS = 10
# Each relator is bounded by `max_relator_tokens` letters, but their sum is not, so a
# presentation can still render to a few hundred characters; keep a line terminal-sized.
_MAX_PRESENTATION_CHARS = 120


@dataclass(frozen=True, slots=True)
class ShowcaseStep:
    """One move of the showcased episode and the presentation it produced."""

    move: str
    presentation: str
    length: int


@dataclass(frozen=True, slots=True)
class ShowcaseEpisode:
    """A played-out episode: where it started, what it did, how it ended."""

    start: BalancedPresentation
    start_distance: int | None
    steps: tuple[ShowcaseStep, ...]
    solved: bool
    reason: str
    best_length: int


class EpisodeShowcase:
    """Play one episode with the current model every ``every_hours`` and print it.

    The throttle is checked at checkpoint boundaries (the pipeline drives it), so
    it fires on the first checkpoint of a run -- an early sanity view of the
    untrained policy -- and then once per interval. Action selection mirrors the
    run's backend: the PUCT search's most-visited move for AlphaZero, the policy
    head's most likely legal move for PPO. Both take the argmax rather than
    sampling, so the transcript shows the agent's committed choice.
    """

    def __init__(
        self,
        config: TrainingPipelineConfig,
        encoder: StateEncoder,
        source: InstanceSource,
        *,
        every_hours: float,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.source = source
        self.interval_s = every_hours * 3600.0
        self._last: float | None = None

    def due(self) -> bool:
        """Whether an episode is owed: at the first check, then once per interval."""
        return self._last is None or time.monotonic() - self._last >= self.interval_s

    def show(
        self,
        manager: CallbackManager,
        model: PolicyValueModel,
        *,
        event_id: int,
        iteration: int,
        seed: int,
        alpha: float | None,
        max_distance: int | None,
    ) -> ShowcaseEpisode:
        """Play one episode, print its transcript, and record its shape as an event."""
        self._last = time.monotonic()
        episode = self._play(model, seed, alpha, max_distance)
        print(render_episode(episode, iteration), flush=True)
        manager.emit(
            event_id,
            "showcase",
            "played one self-play episode",
            {
                "iteration": iteration,
                "moves": len(episode.steps),
                "solved": episode.solved,
                "termination_reason": episode.reason,
                "start_length": episode.start.total_length,
                "best_length": episode.best_length,
                **({} if episode.start_distance is None else {"distance": episode.start_distance}),
            },
        )
        return episode

    def _play(
        self,
        model: PolicyValueModel,
        seed: int,
        alpha: float | None,
        max_distance: int | None,
    ) -> ShowcaseEpisode:
        presentation = self.source.sample(seed, max_distance)
        start_distance, max_moves = episode_distance_and_moves(
            self.source, presentation, self.config.curriculum_config.unknown_distance_max_moves
        )
        env = build_env(self.config, presentation, self.source, alpha, max_moves)
        names = presentation.generator_names
        mcts = (
            None
            if self.config.agent == "ppo"
            else PUCTMCTS(
                model,
                self.encoder,
                PUCTConfig(simulations=self.config.mcts_simulations, c_puct=self.config.c_puct),
            )
        )
        steps: list[ShowcaseStep] = []
        terminated = truncated = False
        reason = "no_legal_action"
        while not terminated and not truncated:
            mask = env.legal_action_mask()
            if not any(mask):
                break
            action = self._best_action(env, model, mask, mcts)
            move = env.catalog.move(action)
            _, _, terminated, truncated, info = env.step(action)
            state = env.state
            steps.append(
                ShowcaseStep(
                    move=format_move(move, names),
                    presentation=state.presentation.format(),
                    length=state.presentation.total_length,
                )
            )
            reason = str(info["termination_reason"])
        return ShowcaseEpisode(
            start=presentation,
            start_distance=start_distance,
            steps=tuple(steps),
            solved=terminated,
            reason=reason,
            best_length=env.state.best_length,
        )

    def _best_action(
        self,
        env: ACEnvironment,
        model: PolicyValueModel,
        mask: tuple[bool, ...],
        mcts: PUCTMCTS | None,
    ) -> int:
        """The move the agent commits to here: most-visited (PUCT) or most likely (PPO)."""
        if mcts is None:
            encoding = self.encoder.encode(env.state)
            policy = masked_softmax(model.apply(encoding, len(mask)).logits, mask)
        else:
            policy = visit_count_policy(mcts.search(env).visit_counts, mask)
        return int(np.argmax(policy))


def format_move(move: PrimitiveMove, generator_names: tuple[str, ...]) -> str:
    """Render one move as the rewrite rule it applies, e.g. ``r0 <- r0 r1^-1``."""
    match move:
        case MultiplyRelatorsMove(target=target, source=source):
            return f"AC1 r{target} <- r{target} r{source}"
        case InvertRelatorMove(target=target):
            return f"AC2 r{target} <- r{target}^-1"
        case ConjugateRelatorMove(target=target, generator=generator):
            left = _letter(generator, generator_names)
            right = _letter(-generator, generator_names)
            return f"AC3 r{target} <- {left} r{target} {right}"
        case ConcatRelatorMove(target=target, source=source, side=side, invert_source=invert):
            other = f"r{source}^-1" if invert else f"r{source}"
            body = f"{other} r{target}" if side == "left" else f"r{target} {other}"
            return f"CAT r{target} <- {body}"
    raise TypeError(f"unsupported primitive move {move!r}")


def render_episode(episode: ShowcaseEpisode, iteration: int) -> str:
    """Render the episode as an indented, move-by-move transcript block."""
    distance = "?" if episode.start_distance is None else str(episode.start_distance)
    outcome = "SOLVED" if episode.solved else f"unsolved ({episode.reason})"
    lines = [
        f"showcase: self-play episode at iteration {iteration} "
        f"| distance={distance} moves={len(episode.steps)} {outcome}",
        f"  start     {_ellipsize(episode.start.format())}  (length {episode.start.total_length})",
    ]
    width = max((len(step.move) for step in episode.steps), default=0)
    for number, step in _numbered_steps(episode.steps):
        if step is None:
            lines.append(f"  ... {number} moves elided ...")
            continue
        lines.append(
            f"  {number:>3}. {step.move:<{width}}  {_ellipsize(step.presentation)}"
            f"  (length {step.length})"
        )
    lines.append(f"  best length reached: {episode.best_length}")
    return "\n".join(lines)


def _numbered_steps(
    steps: tuple[ShowcaseStep, ...],
) -> list[tuple[int, ShowcaseStep | None]]:
    """Number the steps, replacing a long middle stretch with an elision marker.

    The marker is carried as a ``None`` step whose number is how many moves it
    stands for, so the transcript never silently hides part of the episode.
    """
    if len(steps) <= _HEAD_STEPS + _TAIL_STEPS + 1:
        return [(index, step) for index, step in enumerate(steps, 1)]
    head = [(index, step) for index, step in enumerate(steps[:_HEAD_STEPS], 1)]
    elided = len(steps) - _HEAD_STEPS - _TAIL_STEPS
    tail_start = len(steps) - _TAIL_STEPS + 1
    tail = [(index, step) for index, step in enumerate(steps[-_TAIL_STEPS:], tail_start)]
    return [*head, (elided, None), *tail]


def _letter(letter: int, generator_names: tuple[str, ...]) -> str:
    name = generator_names[abs(letter) - 1]
    return name if letter > 0 else f"{name}^-1"


def _ellipsize(text: str) -> str:
    if len(text) <= _MAX_PRESENTATION_CHARS:
        return text
    return text[: _MAX_PRESENTATION_CHARS - 1] + "…"
