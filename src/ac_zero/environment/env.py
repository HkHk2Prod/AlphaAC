from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.encoding.padded import StateEncoder, within_capacity
from ac_zero.environment.goals import exact_standard_goal, signed_permuted_basis_goal
from ac_zero.environment.navigation_reward import (
    EpisodeStats,
    RewardComponents,
    RewardComputer,
    RewardConfig,
)
from ac_zero.environment.rewards import RewardSignal, step_reward
from ac_zero.environment.state import ACSearchState
from ac_zero.moves.universal import moveset_catalog

ENV_ID = "ACZero-v0"


@dataclass(frozen=True, slots=True)
class ACEnvironmentConfig:
    """Runtime limits and goal semantics for an AC search episode."""

    max_moves: int = 16
    mask_noops: bool = True
    goal_mode: str = "exact_standard"
    # Reaching a goal is rewarded on top of length reduction so the solved state
    # is the unique optimum; pure length reduction saturates at non-goal states.
    reward_mode: str = "length_reduction_and_goal"
    goal_reward: float = 1.0
    # Config for the "navigation" reward mode (ignored by other modes).
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    # Shaping weight this episode runs at under the "navigation" mode; the
    # training loop advances it between episodes with an `AlphaUpdater`. `None`
    # falls back to `reward_config.alpha_initial`.
    alpha: float | None = None
    # Which named move set (`ac_zero.moves.universal.MOVE_SET_NAMES`) the episode
    # steps with.
    moveset: str = "strict-ac"


class ACEnvironment(gymnasium.Env[dict[str, Any], int]):
    """Gymnasium environment for strict Andrews-Curtis transformations.

    Observations are the padded encoder arrays (a `spaces.Dict`), so standard RL
    libraries can consume the env directly; the rich `ACSearchState` Markov state
    is carried in `info["state"]` and on `self.state` for the project's tree
    searches. `terminated` is reserved for a true goal state, while horizons and
    action-capacity failures are truncations.

    The one length bound is `relator_capacity`: a move that would make a relator
    longer than the encoder can hold is masked out, never played. Stepping one
    anyway is a caller bug, and the encoder raises on it rather than truncating a
    presentation into a different -- wrong -- one.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 — gymnasium.Env convention

    def __init__(
        self,
        presentation: BalancedPresentation,
        config: ACEnvironmentConfig | None = None,
        encoder: StateEncoder | None = None,
        potentials: Mapping[str, int] | None = None,
    ) -> None:
        """Create an episode beginning at `presentation` with a strict catalog.

        `potentials` maps a presentation's content hash to its distance to the
        trivial group, used only by the "potential" reward mode; states missing
        from it are off the annotated graph (see `_potential_step`).
        """
        self.initial = presentation
        self.config = config or ACEnvironmentConfig()
        self.encoder = encoder or StateEncoder()
        self.potentials = potentials or {}
        self.catalog = moveset_catalog(self.config.moveset, presentation.rank)
        self.action_space = spaces.Discrete(len(self.catalog))
        self.observation_space = self._build_observation_space()
        # The "navigation" reward keeps within-episode state (visited set, running
        # minimum distance), so the env drives a RewardComputer instead of the
        # pure `step_reward`. `None` for every other mode.
        self._reward = (
            RewardComputer(self.config.reward_config)
            if self.config.reward_mode == "navigation"
            else None
        )
        self._last_components: RewardComponents | None = None
        # Last known distance to the destination, carried so the navigation reward
        # can defer an off-graph excursion's shaping credit until the search
        # re-enters a node whose distance is known (see `_navigation_step`).
        self._nav_anchor: int = 0
        self.state = self._initial_state()
        if self._reward is not None:
            self._start_navigation_episode()

    def _build_observation_space(self) -> spaces.Dict:
        rank = self.initial.rank
        relators = len(self.initial.relators)
        width = self.encoder.max_relator_tokens
        return spaces.Dict(
            {
                "tokens": spaces.Box(0, 2 * rank + 1, (relators, width), dtype=np.int64),
                "mask": spaces.MultiBinary([relators, width]),
                "scalar_features": spaces.Box(0.0, np.inf, (4,), dtype=np.float64),
            }
        )

    def _initial_state(self) -> ACSearchState:
        anchor = (
            self._known_potential(self.initial, self._is_goal(self.initial))
            if self.config.reward_mode == "potential"
            else None
        )
        return ACSearchState(
            presentation=self.initial,
            initial_length=self.initial.total_length,
            best_length=self.initial.total_length,
            moves_used=0,
            moves_remaining=self.config.max_moves,
            catalog_version=self.catalog.version,
            last_known_potential=anchor,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the initial Markov state and return `(observation, info)`."""
        super().reset(seed=seed)
        self.state = self._initial_state()
        self._last_components = None
        if self._reward is not None:
            self._start_navigation_episode()
        return self._observation(), self._info(self.state, "running", False)

    def _observation(self) -> dict[str, Any]:
        return self.encoder.encode(self.state).as_observation()

    @property
    def relator_capacity(self) -> int:
        """The longest relator an episode may reach: the encoder's grid width.

        The searches that expand presentations themselves (breadth-first, iterative
        deepening, bidirectional, greedy) prune by this rather than re-deriving it, so
        they explore the same graph the environment lets a model move in -- and the
        same one its dataset was generated under.
        """
        return self.encoder.max_relator_tokens

    def legal_action_mask(self, state: ACSearchState | None = None) -> tuple[bool, ...]:
        """Compute which strict primitive actions are currently allowed.

        A move is legal when every relator it produces fits the encoder's per-relator
        capacity. That bound keeps the episode within the states the model can actually
        represent -- the encoder refuses to truncate an over-long relator, so a move
        that would make one is masked out here rather than left to fail at the next
        observation -- and it is the *only* length bound: nothing caps the relators'
        sum. It is also the bound the training dataset was generated under, so the
        moves masked out here are exactly the ones the data left unlabelled.
        """
        st = state or self.state
        current = st.presentation.relators
        capacity = self.relator_capacity
        mask: list[bool] = []
        for move in self.catalog.moves:
            nxt = move.apply(st.presentation)
            legal = within_capacity(nxt, capacity)
            # Moves only ever rewrite relators, leaving rank and generator names
            # intact, so an unchanged relator tuple is the same no-op test as an
            # unchanged content hash -- without a SHA-256 of a JSON dump per move.
            if self.config.mask_noops and nxt.relators == current:
                legal = False
            mask.append(legal)
        return tuple(mask)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one catalog action and return the Gymnasium 5-tuple."""
        move = self.catalog.move(action)
        prev = self.state
        nxt_pres = move.apply(prev.presentation)
        best = min(prev.best_length, nxt_pres.total_length)
        terminated = self._is_goal(nxt_pres)
        potential_delta, anchor = self._potential_step(prev, nxt_pres, terminated)
        if self._reward is not None:
            reward = self._navigation_step(nxt_pres, terminated)
        else:
            reward = step_reward(
                self.config.reward_mode,
                RewardSignal(
                    previous_best_length=prev.best_length,
                    new_best_length=best,
                    goal_reached=terminated,
                    goal_reward=self.config.goal_reward,
                    potential_delta=potential_delta,
                ),
            )
        remaining = max(0, prev.moves_remaining - 1)
        nxt = ACSearchState(
            presentation=nxt_pres,
            initial_length=prev.initial_length,
            best_length=best,
            moves_used=prev.moves_used + 1,
            moves_remaining=remaining,
            catalog_version=self.catalog.version,
            last_action=action,
            last_known_potential=anchor,
        )
        truncated = False
        reason = "running"
        # The mask is the dominant cost of a step (it applies every catalog move), so
        # the one computed for the no-legal-action check is handed to `_info` rather
        # than recomputed there.
        mask: tuple[bool, ...] | None = None
        if terminated:
            reason = "goal"
        elif remaining == 0:
            truncated = True
            reason = "horizon"
        else:
            mask = self.legal_action_mask(nxt)
            if not any(mask):
                truncated = True
                reason = "no_legal_action"
        self.state = nxt
        info = self._info(nxt, reason, terminated, mask)
        return self._observation(), reward, terminated, truncated, info

    def _info(
        self,
        state: ACSearchState,
        reason: str,
        success: bool,
        mask: tuple[bool, ...] | None = None,
    ) -> dict[str, Any]:
        pres = state.presentation
        if mask is None:
            mask = self.legal_action_mask(state)
        info: dict[str, Any] = {
            "state": state,
            "action_mask": np.asarray(mask, dtype=np.int8),
            "current_total_length": pres.total_length,
            "best_total_length": state.best_length,
            "raw_episode_reduction": state.initial_length - state.best_length,
            "normalized_reduction": (state.initial_length - state.best_length)
            / max(1, state.initial_length),
            "move_count": state.moves_used,
            "success": success,
            "termination_reason": reason,
        }
        if self._last_components is not None:
            info["reward_components"] = self._last_components
        return info

    def _known_potential(self, presentation: BalancedPresentation, is_goal: bool) -> float | None:
        """Distance to the trivial group, or `None` off the annotated graph.

        The goal is the trivial group itself, so it always has a known potential of
        zero even when it is not carried as a dataset group in `potentials`.
        """
        if is_goal:
            return 0.0
        distance = self.potentials.get(presentation.content_hash)
        return float(distance) if distance is not None else None

    def _potential_step(
        self, prev: ACSearchState, nxt_pres: BalancedPresentation, terminated: bool
    ) -> tuple[float, float | None]:
        """Return `(potential_delta, new_anchor)` for one step of the potential reward.

        While off the annotated graph the delta is zero and the exit potential
        (``prev.last_known_potential``) is carried forward unchanged; re-entering the
        known region credits the whole ``exit - entry`` change against that anchor.
        """
        if self.config.reward_mode != "potential":
            return 0.0, prev.last_known_potential
        next_potential = self._known_potential(nxt_pres, terminated)
        if next_potential is None:
            return 0.0, prev.last_known_potential  # off-graph: hold the exit potential
        anchor = prev.last_known_potential
        delta = 0.0 if anchor is None else anchor - next_potential
        return delta, next_potential

    def _navigation_distance(self, presentation: BalancedPresentation, is_goal: bool) -> int | None:
        """Shortest-path distance to the destination for the navigation reward.

        The destination (goal) is always distance zero; annotated groups use their
        exact ``distance_to_origin`` from ``potentials``. A presentation off the
        annotated graph has no known distance and returns ``None`` -- navigation
        never invents a length proxy; it defers the excursion's shaping credit to
        the re-entry step instead (see `_navigation_step`). Navigation runs require
        annotations, so the start node and the goal are always known.
        """
        if is_goal:
            return 0
        distance = self.potentials.get(presentation.content_hash)
        return int(distance) if distance is not None else None

    def _start_navigation_episode(self) -> None:
        """Reset the RewardComputer and distance anchor at the current ``alpha``."""
        assert self._reward is not None
        start = self.initial
        alpha = self.config.alpha
        if alpha is None:
            alpha = self.config.reward_config.alpha_initial
        known = self._navigation_distance(start, self._is_goal(start))
        # Navigation requires annotations, so the start is a known node; the guard
        # only defends a misconfigured off-graph start (anchor 0 -> no shaping/bonus).
        self._nav_anchor = known if known is not None else 0
        self._reward.start_episode(
            alpha=alpha, start_node=start.content_hash, start_distance=self._nav_anchor
        )

    def _navigation_step(self, next_node: BalancedPresentation, terminated: bool) -> float:
        """Score one transition with the RewardComputer, deferring off-graph credit.

        The step is scored against the anchor (the last known distance): re-entering
        a known node credits the whole ``anchor - distance`` descent at once, while an
        off-graph step holds the anchor and so scores zero shaping.
        """
        assert self._reward is not None
        known_after = self._navigation_distance(next_node, terminated)
        distance_after = self._nav_anchor if known_after is None else known_after
        components = self._reward.step(
            next_node=next_node.content_hash,
            distance_before=self._nav_anchor,
            distance_after=distance_after,
            reached_destination=terminated,
        )
        if known_after is not None:
            self._nav_anchor = known_after
        self._last_components = components
        return components.reward_total

    def navigation_episode_stats(self) -> EpisodeStats:
        """Aggregate stats for the finished navigation episode (alpha updater input)."""
        if self._reward is None:
            raise ValueError("navigation_episode_stats requires reward_mode 'navigation'")
        return self._reward.episode_stats()

    def _is_goal(self, presentation: BalancedPresentation) -> bool:
        if self.config.goal_mode == "exact_standard":
            return exact_standard_goal(presentation)
        if self.config.goal_mode == "signed_permuted_basis":
            return signed_permuted_basis_goal(presentation)
        raise ValueError(f"unknown goal mode {self.config.goal_mode!r}")


if ENV_ID not in gymnasium.registry:
    gymnasium.register(id=ENV_ID, entry_point="ac_zero.environment.env:ACEnvironment")
