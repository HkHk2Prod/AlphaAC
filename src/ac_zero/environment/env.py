from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.environment.goals import exact_standard_goal, signed_permuted_basis_goal
from ac_zero.environment.state import ACSearchState
from ac_zero.moves.catalog import ActionCatalog


@dataclass(frozen=True, slots=True)
class ACEnvironmentConfig:
    """Runtime limits and goal semantics for an AC search episode."""

    max_moves: int = 16
    total_length_cap: int = 128
    mask_noops: bool = True
    goal_mode: str = "exact_standard"


class ACEnvironment:
    """Deterministic finite-horizon environment for strict AC transformations.

    The environment keeps Gymnasium-compatible `terminated`/`truncated`
    semantics: `terminated` is reserved for a true goal state, while horizons,
    safety caps, and action-capacity failures are truncations.
    """

    def __init__(
        self, initial: BalancedPresentation, config: ACEnvironmentConfig | None = None
    ) -> None:
        """Create an episode beginning at `initial` with a strict action catalog."""
        self.initial = initial
        self.config = config or ACEnvironmentConfig()
        self.catalog = ActionCatalog(initial.rank)
        self.state = self.reset()

    def reset(self) -> ACSearchState:
        """Reset to the initial Markov state and return it."""
        self.state = ACSearchState(
            presentation=self.initial,
            initial_length=self.initial.total_length,
            best_length=self.initial.total_length,
            moves_used=0,
            moves_remaining=self.config.max_moves,
            catalog_version=self.catalog.version,
        )
        return self.state

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

    def step(self, action_id: int) -> tuple[ACSearchState, int, bool, bool, dict[str, Any]]:
        """Apply one catalog action and return state, reward, termination, and info."""
        move = self.catalog.move(action_id)
        prev = self.state
        nxt_pres = move.apply(prev.presentation)
        best = min(prev.best_length, nxt_pres.total_length)
        reward = prev.best_length - best
        remaining = max(0, prev.moves_remaining - 1)
        nxt = ACSearchState(
            presentation=nxt_pres,
            initial_length=prev.initial_length,
            best_length=best,
            moves_used=prev.moves_used + 1,
            moves_remaining=remaining,
            catalog_version=self.catalog.version,
            last_action=action_id,
            safety_truncated=nxt_pres.total_length > self.config.total_length_cap,
        )
        terminated = self._is_goal(nxt_pres)
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
        info = {
            "current_total_length": nxt_pres.total_length,
            "best_total_length": best,
            "raw_episode_reduction": prev.initial_length - best,
            "normalized_reduction": (prev.initial_length - best) / max(1, prev.initial_length),
            "move_count": nxt.moves_used,
            "success": terminated,
            "termination_reason": reason,
            "presentation_hash": nxt_pres.content_hash,
        }
        return nxt, reward, terminated, truncated, info

    def _is_goal(self, presentation: BalancedPresentation) -> bool:
        if self.config.goal_mode == "exact_standard":
            return exact_standard_goal(presentation)
        if self.config.goal_mode == "signed_permuted_basis":
            return signed_permuted_basis_goal(presentation)
        raise ValueError(f"unknown goal mode {self.config.goal_mode!r}")
