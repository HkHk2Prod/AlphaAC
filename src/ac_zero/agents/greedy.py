from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path

from ac_zero.agents.base import SolverResult
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.certificates.certificate import build_certificate
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.environment.env import ACEnvironment
from ac_zero.environment.goals import exact_standard_goal, signed_permuted_basis_goal
from ac_zero.moves.catalog import ActionCatalog


@dataclass(frozen=True, slots=True)
class GreedyActionEvaluation:
    """One-step evaluation record for a legal greedy action.

    The record is deliberately small and serializable-friendly so tests and
    benchmark reports can explain why the greedy policy chose a move.
    """

    action_id: int
    next_total_length: int
    immediate_reduction: int
    reaches_goal: bool


@dataclass(frozen=True, slots=True)
class GreedySolverConfig:
    """Configuration for the one-step greedy rollout solver.

    `require_strict_progress` makes the solver stop at a local minimum unless
    the chosen action reaches the goal. This avoids wandering through
    length-preserving moves and then misreporting an arbitrary horizon failure.
    """

    require_strict_progress: bool = True
    write_certificate: bool = True


@dataclass(frozen=True, slots=True)
class GreedyBestFirstConfig:
    """Budgets for length-ordered greedy best-first search."""

    max_expansions: int = 256
    max_generated: int = 10_000
    write_certificate: bool = True


class GreedyLengthAgent:
    """Select the legal action with the best immediate length/goal score."""

    def __init__(self, env: ACEnvironment) -> None:
        """Bind the selector to the mutable environment it will inspect."""
        self.env = env

    def rank_actions(self, mask: tuple[bool, ...]) -> tuple[GreedyActionEvaluation, ...]:
        """Return legal actions sorted by goal reachability, length, and action ID."""
        state = self.env.state
        evaluations: list[GreedyActionEvaluation] = []
        for action_id, ok in enumerate(mask):
            if not ok:
                continue
            nxt = self.env.catalog.move(action_id).apply(state.presentation)
            current_length = state.presentation.total_length
            next_length = nxt.total_length
            evaluations.append(
                GreedyActionEvaluation(
                    action_id=action_id,
                    next_total_length=next_length,
                    immediate_reduction=current_length - next_length,
                    reaches_goal=self._is_goal(nxt),
                )
            )
        return tuple(sorted(evaluations, key=_evaluation_sort_key))

    def select_action(self, mask: tuple[bool, ...]) -> int:
        """Return the highest-ranked legal action ID."""
        ranked = self.rank_actions(mask)
        if not ranked:
            raise RuntimeError("no legal actions")
        return ranked[0].action_id

    def best_evaluation(self, mask: tuple[bool, ...]) -> GreedyActionEvaluation:
        """Return the full evaluation record for the selected action."""
        ranked = self.rank_actions(mask)
        if not ranked:
            raise RuntimeError("no legal actions")
        return ranked[0]

    def _is_goal(self, presentation: BalancedPresentation) -> bool:
        if self.env.config.goal_mode == "exact_standard":
            return exact_standard_goal(presentation)
        if self.env.config.goal_mode == "signed_permuted_basis":
            return signed_permuted_basis_goal(presentation)
        raise ValueError(f"unknown goal mode {self.env.config.goal_mode!r}")


class GreedySolver:
    """Deterministic rollout using the best one-step length/goal action."""

    def __init__(self, config: GreedySolverConfig | None = None) -> None:
        """Create a greedy rollout solver with optional stopping behavior."""
        self.config = config or GreedySolverConfig()

    def solve(
        self,
        env: ACEnvironment,
        *,
        certificate_path: str | Path | None = None,
        experiment_id: str = "greedy",
        seed: int = 0,
    ) -> SolverResult:
        """Run a deterministic greedy rollout until goal, horizon, or local minimum.

        When `certificate_path` is provided and a goal is reached, the emitted
        certificate is immediately replay-verified before `success=True` is
        returned in the `SolverResult`.
        """

        initial = env.initial
        best_state = env.state.presentation
        best_reduction = 0
        path: list[int] = []
        expanded_nodes = 0
        generated_nodes = 0
        termination_reason = "running"

        if env._is_goal(env.state.presentation):
            termination_reason = "goal"
            return self._result(
                initial,
                env.state.presentation,
                tuple(path),
                expanded_nodes,
                generated_nodes,
                termination_reason,
                True,
                certificate_path,
                env.config.goal_mode,
                experiment_id,
                seed,
            )

        while env.state.moves_remaining > 0:
            mask = env.legal_action_mask()
            legal_count = sum(1 for ok in mask if ok)
            if legal_count == 0:
                termination_reason = "no_legal_action"
                break
            expanded_nodes += 1
            generated_nodes += legal_count
            agent = GreedyLengthAgent(env)
            evaluation = agent.best_evaluation(mask)
            # Exact-standard cleanup can be length-preserving, so goal-reaching
            # actions are allowed even when strict progress is otherwise needed.
            if (
                self.config.require_strict_progress
                and evaluation.immediate_reduction <= 0
                and not evaluation.reaches_goal
            ):
                termination_reason = "local_minimum"
                break
            path.append(evaluation.action_id)
            _, _, terminated, truncated, info = env.step(evaluation.action_id)
            state = env.state
            current_reduction = state.initial_length - state.best_length
            if current_reduction > best_reduction:
                best_reduction = current_reduction
                best_state = state.presentation
            if terminated or truncated:
                termination_reason = str(info["termination_reason"])
                break

        success = env._is_goal(env.state.presentation)
        if termination_reason == "running":
            termination_reason = "horizon"
        return self._result(
            initial,
            best_state if not success else env.state.presentation,
            tuple(path),
            expanded_nodes,
            generated_nodes,
            termination_reason,
            success,
            certificate_path,
            env.config.goal_mode,
            experiment_id,
            seed,
        )

    def _result(
        self,
        initial: BalancedPresentation,
        best_state: BalancedPresentation,
        path: tuple[int, ...],
        expanded_nodes: int,
        generated_nodes: int,
        termination_reason: str,
        success: bool,
        certificate_path: str | Path | None,
        goal_mode: str,
        experiment_id: str,
        seed: int,
    ) -> SolverResult:
        cert_path: str | None = None
        if success and certificate_path is not None and self.config.write_certificate:
            catalog = ActionCatalog(initial.rank)
            moves = tuple(catalog.move(action_id) for action_id in path)
            certificate = build_certificate(
                initial,
                moves,
                goal_mode=goal_mode,
                experiment_id=experiment_id,
                seed=seed,
            )
            certificate.write(certificate_path)
            verification = CertificateVerifier().verify_path(certificate_path)
            success = verification.ok
            cert_path = str(certificate_path) if verification.ok else None
        return SolverResult(
            best_state=best_state,
            best_reduction=initial.total_length - best_state.total_length,
            path=path,
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            peak_frontier_size=1,
            termination_reason=termination_reason,
            success=success,
            certificate_path=cert_path,
            metrics={
                "path_length": float(len(path)),
                "initial_total_length": float(initial.total_length),
                "best_total_length": float(best_state.total_length),
            },
        )


class GreedyBestFirstSearch:
    """Best-first classical search ordered by total relator length."""

    def __init__(self, config: GreedyBestFirstConfig | None = None) -> None:
        """Create a best-first searcher with explicit node-generation budgets."""
        self.config = config or GreedyBestFirstConfig()

    def solve(
        self,
        initial: BalancedPresentation,
        *,
        env_template: ACEnvironment,
        certificate_path: str | Path | None = None,
        experiment_id: str = "greedy_best_first",
        seed: int = 0,
    ) -> SolverResult:
        """Explore presentations in increasing total-length order.

        The priority queue uses `(total_length, depth, insertion_order)` so ties
        are deterministic. State deduplication includes depth because the same
        presentation at different depths can have different remaining horizons.
        """

        catalog = ActionCatalog(initial.rank)
        counter = itertools.count()
        frontier: list[tuple[int, int, int, BalancedPresentation, tuple[int, ...]]] = []
        heapq.heappush(frontier, (initial.total_length, 0, next(counter), initial, ()))
        seen = {(initial.content_hash, 0)}
        expanded = 0
        generated = 0
        peak = 1
        best = initial
        best_path: tuple[int, ...] = ()
        reason = "budget_exhausted"

        while frontier and expanded < self.config.max_expansions:
            _, depth, _, pres, path = heapq.heappop(frontier)
            if _is_goal_for_mode(pres, env_template.config.goal_mode):
                reason = "goal"
                return self._result(
                    initial,
                    pres,
                    path,
                    expanded,
                    generated,
                    peak,
                    reason,
                    True,
                    certificate_path,
                    env_template.config.goal_mode,
                    experiment_id,
                    seed,
                )
            if pres.total_length < best.total_length:
                best = pres
                best_path = path
            if depth >= env_template.config.max_moves:
                continue
            expanded += 1
            for action_id, move in enumerate(catalog.moves):
                nxt = move.apply(pres)
                if nxt.total_length > env_template.config.total_length_cap:
                    continue
                if env_template.config.mask_noops and nxt.content_hash == pres.content_hash:
                    continue
                key = (nxt.content_hash, depth + 1)
                if key in seen:
                    continue
                seen.add(key)
                generated += 1
                if generated > self.config.max_generated:
                    reason = "generated_budget_exhausted"
                    frontier.clear()
                    break
                heapq.heappush(
                    frontier,
                    (nxt.total_length, depth + 1, next(counter), nxt, (*path, action_id)),
                )
            peak = max(peak, len(frontier))
        return self._result(
            initial,
            best,
            best_path,
            expanded,
            generated,
            peak,
            reason if frontier else "frontier_exhausted",
            False,
            certificate_path,
            env_template.config.goal_mode,
            experiment_id,
            seed,
        )

    def _result(
        self,
        initial: BalancedPresentation,
        best_state: BalancedPresentation,
        path: tuple[int, ...],
        expanded_nodes: int,
        generated_nodes: int,
        peak_frontier_size: int,
        termination_reason: str,
        success: bool,
        certificate_path: str | Path | None,
        goal_mode: str,
        experiment_id: str,
        seed: int,
    ) -> SolverResult:
        cert_path: str | None = None
        if success and certificate_path is not None and self.config.write_certificate:
            catalog = ActionCatalog(initial.rank)
            certificate = build_certificate(
                initial,
                tuple(catalog.move(action_id) for action_id in path),
                goal_mode=goal_mode,
                experiment_id=experiment_id,
                seed=seed,
            )
            certificate.write(certificate_path)
            verification = CertificateVerifier().verify_path(certificate_path)
            success = verification.ok
            cert_path = str(certificate_path) if verification.ok else None
        return SolverResult(
            best_state=best_state,
            best_reduction=initial.total_length - best_state.total_length,
            path=path,
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            peak_frontier_size=peak_frontier_size,
            termination_reason=termination_reason,
            success=success,
            certificate_path=cert_path,
            metrics={
                "path_length": float(len(path)),
                "initial_total_length": float(initial.total_length),
                "best_total_length": float(best_state.total_length),
            },
        )


def _evaluation_sort_key(evaluation: GreedyActionEvaluation) -> tuple[int, int, int, int]:
    """Sort goal actions first, then shorter states, then stable action IDs."""
    return (
        0 if evaluation.reaches_goal else 1,
        evaluation.next_total_length,
        -evaluation.immediate_reduction,
        evaluation.action_id,
    )


def _is_goal_for_mode(presentation: BalancedPresentation, goal_mode: str) -> bool:
    """Evaluate the configured goal predicate for classical search nodes."""
    if goal_mode == "exact_standard":
        return exact_standard_goal(presentation)
    if goal_mode == "signed_permuted_basis":
        return signed_permuted_basis_goal(presentation)
    raise ValueError(f"unknown goal mode {goal_mode!r}")
