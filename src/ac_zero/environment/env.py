from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.goals import exact_standard_goal, signed_permuted_basis_goal
from ac_zero.environment.rewards import RewardSignal, step_reward
from ac_zero.environment.state import ACSearchState
from ac_zero.moves.universal import moveset_catalog

ENV_ID = "ACZero-v0"


@dataclass(frozen=True, slots=True)
class ACEnvironmentConfig:
    """Runtime limits and goal semantics for an AC search episode."""

    max_moves: int = 16
    total_length_cap: int = 128
    mask_noops: bool = True
    goal_mode: str = "exact_standard"
    # Reaching a goal is rewarded on top of length reduction so the solved state
    # is the unique optimum; pure length reduction saturates at non-goal states.
    reward_mode: str = "length_reduction_and_goal"
    goal_reward: float = 1.0
    # Which named move set (`ac_zero.moves.universal.MOVE_SET_NAMES`) the episode
    # steps with.
    moveset: str = "strict-ac"


class ACEnvironment(gymnasium.Env[dict[str, Any], int]):
    """Gymnasium environment for strict Andrews-Curtis transformations.

    Observations are the padded encoder arrays (a `spaces.Dict`), so standard RL
    libraries can consume the env directly; the rich `ACSearchState` Markov state
    is carried in `info["state"]` and on `self.state` for the project's tree
    searches. `terminated` is reserved for a true goal state, while horizons,
    safety caps, and action-capacity failures are truncations.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 — gymnasium.Env convention

    def __init__(
        self,
        presentation: BalancedPresentation,
        config: ACEnvironmentConfig | None = None,
        encoder: StateEncoder | None = None,
    ) -> None:
        """Create an episode beginning at `presentation` with a strict catalog."""
        self.initial = presentation
        self.config = config or ACEnvironmentConfig()
        self.encoder = encoder or StateEncoder()
        self.catalog = moveset_catalog(self.config.moveset, presentation.rank)
        self.action_space = spaces.Discrete(len(self.catalog))
        self.observation_space = self._build_observation_space()
        self.state = self._initial_state()

    def _build_observation_space(self) -> spaces.Dict:
        rank = self.initial.rank
        relators = len(self.initial.relators)
        width = self.encoder.max_word_length
        return spaces.Dict(
            {
                "tokens": spaces.Box(0, 2 * rank + 1, (relators, width), dtype=np.int64),
                "mask": spaces.MultiBinary([relators, width]),
                "scalar_features": spaces.Box(0.0, np.inf, (4,), dtype=np.float64),
            }
        )

    def _initial_state(self) -> ACSearchState:
        return ACSearchState(
            presentation=self.initial,
            initial_length=self.initial.total_length,
            best_length=self.initial.total_length,
            moves_used=0,
            moves_remaining=self.config.max_moves,
            catalog_version=self.catalog.version,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the initial Markov state and return `(observation, info)`."""
        super().reset(seed=seed)
        self.state = self._initial_state()
        return self._observation(), self._info(self.state, "running", False)

    def _observation(self) -> dict[str, Any]:
        return self.encoder.encode(self.state).as_observation()

    def legal_action_mask(self, state: ACSearchState | None = None) -> tuple[bool, ...]:
        """Compute which strict primitive actions are currently allowed."""
        st = state or self.state
        mask: list[bool] = []
        for move in self.catalog.moves:
            nxt = move.apply(st.presentation)
            legal = nxt.total_length <= self.config.total_length_cap
            if self.config.mask_noops and nxt.content_hash == st.presentation.content_hash:
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
        reward = step_reward(
            self.config.reward_mode,
            RewardSignal(
                previous_best_length=prev.best_length,
                new_best_length=best,
                goal_reached=terminated,
                goal_reward=self.config.goal_reward,
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
            safety_truncated=nxt_pres.total_length > self.config.total_length_cap,
        )
        truncated = False
        reason = "running"
        if terminated:
            reason = "goal"
        elif nxt.safety_truncated:
            truncated = True
            reason = "safety_cap"
        elif remaining == 0:
            truncated = True
            reason = "horizon"
        elif not any(self.legal_action_mask(nxt)):
            truncated = True
            reason = "no_legal_action"
        self.state = nxt
        info = self._info(nxt, reason, terminated)
        return self._observation(), reward, terminated, truncated, info

    def _info(self, state: ACSearchState, reason: str, success: bool) -> dict[str, Any]:
        pres = state.presentation
        return {
            "state": state,
            "action_mask": np.asarray(self.legal_action_mask(state), dtype=np.int8),
            "current_total_length": pres.total_length,
            "best_total_length": state.best_length,
            "raw_episode_reduction": state.initial_length - state.best_length,
            "normalized_reduction": (state.initial_length - state.best_length)
            / max(1, state.initial_length),
            "move_count": state.moves_used,
            "success": success,
            "termination_reason": reason,
            "presentation_hash": pres.content_hash,
        }

    def _is_goal(self, presentation: BalancedPresentation) -> bool:
        if self.config.goal_mode == "exact_standard":
            return exact_standard_goal(presentation)
        if self.config.goal_mode == "signed_permuted_basis":
            return signed_permuted_basis_goal(presentation)
        raise ValueError(f"unknown goal mode {self.config.goal_mode!r}")


if ENV_ID not in gymnasium.registry:
    gymnasium.register(id=ENV_ID, entry_point="ac_zero.environment.env:ACEnvironment")
