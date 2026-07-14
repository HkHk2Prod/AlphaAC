from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ac_zero.agents.base import SolverResult
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.certificates.certificate import build_certificate
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.encoding.padded import within_capacity
from ac_zero.environment.env import ACEnvironment
from ac_zero.environment.goals import exact_standard_goal, signed_permuted_basis_goal
from ac_zero.moves.catalog import ActionCatalog


@dataclass(frozen=True, slots=True)
class BreadthFirstConfig:
    """Budgets for uninformed breadth-first search."""

    max_expansions: int = 50_000
    max_generated: int = 400_000
    write_certificate: bool = True


class BreadthFirstSearch:
    """Uninformed breadth-first search for a shortest strict-AC trivialization.

    States are explored in nondecreasing path length using a FIFO frontier with
    content-hash deduplication, so the first goal popped is reached by a
    fewest-moves certificate. That certificate is provably optimal when the
    search reached the goal without ever pruning a strictly shallower branch by
    the relator bound and without exhausting its node budget first.
    """

    def __init__(self, config: BreadthFirstConfig | None = None) -> None:
        """Create a breadth-first searcher with explicit node budgets."""
        self.config = config or BreadthFirstConfig()

    def solve(
        self,
        initial: BalancedPresentation,
        *,
        env_template: ACEnvironment,
        certificate_path: str | Path | None = None,
        experiment_id: str = "breadth_first",
        seed: int = 0,
    ) -> SolverResult:
        """Search for a shortest move sequence reaching the configured goal."""
        catalog = ActionCatalog(initial.rank)
        goal_mode = env_template.config.goal_mode
        capacity = env_template.relator_capacity
        max_moves = env_template.config.max_moves
        frontier: deque[tuple[int, BalancedPresentation, tuple[int, ...]]] = deque()
        frontier.append((0, initial, ()))
        seen = {initial.content_hash}
        expanded = 0
        generated = 0
        peak = 1
        best = initial
        best_path: tuple[int, ...] = ()
        shallowest_prune: int | None = None
        reason = "frontier_exhausted"

        while frontier:
            depth, pres, path = frontier.popleft()
            if _is_goal(pres, goal_mode):
                optimal = shallowest_prune is None or shallowest_prune >= depth - 1
                return self._result(
                    initial,
                    pres,
                    path,
                    expanded,
                    generated,
                    peak,
                    "goal",
                    True,
                    certificate_path,
                    goal_mode,
                    experiment_id,
                    seed,
                    optimal,
                )
            if pres.total_length < best.total_length:
                best, best_path = pres, path
            if depth >= max_moves:
                continue
            if expanded >= self.config.max_expansions:
                reason = "expansion_budget_exhausted"
                break
            expanded += 1
            for action_id, move in enumerate(catalog.moves):
                nxt = move.apply(pres)
                if not within_capacity(nxt, capacity):
                    shallowest_prune = depth if shallowest_prune is None else shallowest_prune
                    continue
                if env_template.config.mask_noops and nxt.content_hash == pres.content_hash:
                    continue
                if nxt.content_hash in seen:
                    continue
                seen.add(nxt.content_hash)
                generated += 1
                if generated > self.config.max_generated:
                    reason = "generated_budget_exhausted"
                    frontier.clear()
                    break
                frontier.append((depth + 1, nxt, (*path, action_id)))
            peak = max(peak, len(frontier))

        return self._result(
            initial,
            best,
            best_path,
            expanded,
            generated,
            peak,
            reason,
            False,
            certificate_path,
            goal_mode,
            experiment_id,
            seed,
            False,
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
        proved_optimal: bool,
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
                "proved_optimal": 1.0 if (success and proved_optimal) else 0.0,
                "initial_total_length": float(initial.total_length),
                "best_total_length": float(best_state.total_length),
            },
        )


def _is_goal(presentation: BalancedPresentation, goal_mode: str) -> bool:
    if goal_mode == "exact_standard":
        return exact_standard_goal(presentation)
    if goal_mode == "signed_permuted_basis":
        return signed_permuted_basis_goal(presentation)
    raise ValueError(f"unknown goal mode {goal_mode!r}")
