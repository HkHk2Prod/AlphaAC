from __future__ import annotations

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
class IterativeDeepeningConfig:
    """Budgets for iterative-deepening depth-first search."""

    max_generated: int = 200_000
    write_certificate: bool = True


class IterativeDeepeningSearch:
    """Iterative-deepening DFS: BFS-style shortest solutions with DFS memory.

    Depth-first search is re-run with an increasing depth bound up to the
    environment horizon, so the first goal found is reached by a fewest-moves
    certificate while memory stays linear in depth. This is the uninformed search
    suited to the unbounded solution length of Andrews-Curtis trivialization.
    """

    def __init__(self, config: IterativeDeepeningConfig | None = None) -> None:
        """Create an iterative-deepening searcher with an explicit node budget."""
        self.config = config or IterativeDeepeningConfig()

    def solve(
        self,
        initial: BalancedPresentation,
        *,
        env_template: ACEnvironment,
        certificate_path: str | Path | None = None,
        experiment_id: str = "iterative_deepening",
        seed: int = 0,
    ) -> SolverResult:
        """Search depth bounds 0..max_moves for a shortest reaching path."""
        catalog = ActionCatalog(initial.rank)
        goal_mode = env_template.config.goal_mode
        capacity = env_template.relator_capacity
        generated = 0
        expanded = 0
        best = initial
        best_path: tuple[int, ...] = ()
        budget_hit = False
        bound_pruned = False

        for limit in range(env_template.config.max_moves + 1):
            stack: list[tuple[BalancedPresentation, tuple[int, ...], frozenset[str]]] = [
                (initial, (), frozenset({initial.content_hash}))
            ]
            while stack:
                pres, path, on_path = stack.pop()
                if _is_goal(pres, goal_mode):
                    # Minimality holds only if the relator bound never hid a branch.
                    return self._result(
                        initial,
                        pres,
                        path,
                        expanded,
                        generated,
                        "goal",
                        True,
                        certificate_path,
                        goal_mode,
                        experiment_id,
                        seed,
                        not bound_pruned,
                    )
                if pres.total_length < best.total_length:
                    best, best_path = pres, path
                if len(path) >= limit:
                    continue
                expanded += 1
                for action_id in reversed(range(len(catalog.moves))):
                    nxt = catalog.move(action_id).apply(pres)
                    if not within_capacity(nxt, capacity):
                        bound_pruned = True
                        continue
                    if nxt.content_hash in on_path:
                        continue
                    if env_template.config.mask_noops and nxt.content_hash == pres.content_hash:
                        continue
                    generated += 1
                    if generated > self.config.max_generated:
                        budget_hit = True
                        stack.clear()
                        break
                    stack.append((nxt, (*path, action_id), on_path | {nxt.content_hash}))
            if budget_hit:
                break

        reason = "generated_budget_exhausted" if budget_hit else "depth_exhausted"
        return self._result(
            initial,
            best,
            best_path,
            expanded,
            generated,
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
            peak_frontier_size=len(path) + 1,
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
