from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

from ac_zero.agents.base import SolverResult
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.algebra.word import FreeGroupWord
from ac_zero.certificates.certificate import build_certificate
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.environment.env import ACEnvironment
from ac_zero.moves.catalog import ActionCatalog
from ac_zero.moves.primitive import ConjugateRelatorMove, InvertRelatorMove, PrimitiveMove


@dataclass(frozen=True, slots=True)
class BidirectionalConfig:
    """Budgets for bidirectional breadth-first search."""

    max_expansions: int = 50_000
    max_generated: int = 400_000
    write_certificate: bool = True


class BidirectionalSearch:
    """Meet-in-the-middle BFS between the start and the goal orbit.

    A forward frontier grows from the initial presentation by catalog moves while
    a backward frontier grows from every goal state by *predecessor* moves (the
    exact inverse operation of each catalog move, labelled with the forward move
    that re-derives the child). Layers are expanded alternately, smaller frontier
    first, and a state seen by both sides splices a forward path
    ``initial -> m -> goal``. Nicholson's rule (stop once the summed frontier
    depths reach the best meeting length) makes the spliced path a fewest-moves
    certificate, provably optimal unless the length cap pruned a branch or a
    budget was exhausted first. The win over one-directional BFS is reach: the
    two shallow frontiers together cover roughly ``b^(d/2)`` states instead of
    ``b^d``, so deeper trivializations land inside the same node budget.
    """

    def __init__(self, config: BidirectionalConfig | None = None) -> None:
        """Create a bidirectional searcher with explicit node budgets."""
        self.config = config or BidirectionalConfig()

    def solve(
        self,
        initial: BalancedPresentation,
        *,
        env_template: ACEnvironment,
        certificate_path: str | Path | None = None,
        experiment_id: str = "bidirectional",
        seed: int = 0,
    ) -> SolverResult:
        """Search both directions for a shortest move sequence to the goal."""
        catalog = ActionCatalog(initial.rank)
        cfg = env_template.config
        cap = cfg.total_length_cap
        max_moves = cfg.max_moves
        mask_noops = cfg.mask_noops

        seen_f: dict[str, tuple[int, ...]] = {initial.content_hash: ()}
        seen_b: dict[str, tuple[int, ...]] = {}
        frontier_b: list[tuple[BalancedPresentation, tuple[int, ...]]] = []
        for goal in _goal_states(initial, cfg.goal_mode):
            if goal.content_hash not in seen_b:
                seen_b[goal.content_hash] = ()
                frontier_b.append((goal, ()))
        frontier_f: list[tuple[BalancedPresentation, tuple[int, ...]]] = [(initial, ())]

        best_total: int | None = 0 if initial.content_hash in seen_b else None
        best_path: tuple[int, ...] = ()
        best_len_state: BalancedPresentation = initial
        best_len_path: tuple[int, ...] = ()
        expanded = generated = 0
        peak = len(frontier_f) + len(frontier_b)
        cap_pruned = budget_hit = False
        reason = "frontier_exhausted"
        df = db = 0

        while frontier_f and frontier_b:
            if best_total is not None and df + db >= best_total:
                break
            can_f, can_b = df < max_moves, db < max_moves
            if not can_f and not can_b:
                break
            forward = can_f and (not can_b or len(frontier_f) <= len(frontier_b))
            layer = frontier_f if forward else frontier_b
            batch = self._expand(
                layer,
                seen_f if forward else seen_b,
                seen_b if forward else seen_f,
                backward=not forward,
                catalog=catalog,
                cap=cap,
                mask_noops=mask_noops,
                max_moves=max_moves,
                generated_so_far=generated,
            )
            expanded += batch.expanded
            generated += batch.generated
            cap_pruned = cap_pruned or batch.cap_pruned
            for total, full in batch.meetings:
                if best_total is None or total < best_total:
                    best_total, best_path = total, full
            if forward:
                frontier_f = batch.next_layer
                df += 1
                for pres, path in batch.next_layer:
                    if pres.total_length < best_len_state.total_length:
                        best_len_state, best_len_path = pres, path
            else:
                frontier_b = batch.next_layer
                db += 1
            peak = max(peak, len(frontier_f) + len(frontier_b))
            if batch.budget_hit:
                budget_hit, reason = True, "generated_budget_exhausted"
                break
            if expanded > self.config.max_expansions:
                budget_hit, reason = True, "expansion_budget_exhausted"
                break

        if best_total is not None:
            final = _apply_path(initial, best_path, catalog)
            proved = not cap_pruned and not budget_hit
            return self._result(
                initial,
                final,
                best_path,
                expanded,
                generated,
                peak,
                "goal",
                True,
                certificate_path,
                cfg.goal_mode,
                experiment_id,
                seed,
                proved,
            )
        return self._result(
            initial,
            best_len_state,
            best_len_path,
            expanded,
            generated,
            peak,
            reason,
            False,
            certificate_path,
            cfg.goal_mode,
            experiment_id,
            seed,
            False,
        )

    def _expand(
        self,
        layer: list[tuple[BalancedPresentation, tuple[int, ...]]],
        seen_self: dict[str, tuple[int, ...]],
        seen_other: dict[str, tuple[int, ...]],
        *,
        backward: bool,
        catalog: ActionCatalog,
        cap: int,
        mask_noops: bool,
        max_moves: int,
        generated_so_far: int,
    ) -> _Batch:
        """Expand one full BFS layer, splicing any state the other side reached."""
        batch = _Batch()
        for pres, path in layer:
            batch.expanded += 1
            for action_id, move in enumerate(catalog.moves):
                nxt = _predecessor(pres, move) if backward else move.apply(pres)
                if nxt.total_length > cap:
                    batch.cap_pruned = True
                    continue
                if mask_noops and nxt.content_hash == pres.content_hash:
                    continue
                key = nxt.content_hash
                if key in seen_self:
                    continue
                new_path = (action_id, *path) if backward else (*path, action_id)
                other = seen_other.get(key)
                if other is not None:
                    total = len(new_path) + len(other)
                    if total <= max_moves:
                        full = (*other, *new_path) if backward else (*new_path, *other)
                        batch.meetings.append((total, full))
                seen_self[key] = new_path
                batch.generated += 1
                if generated_so_far + batch.generated > self.config.max_generated:
                    batch.budget_hit = True
                    return batch
                batch.next_layer.append((nxt, new_path))
        return batch

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


@dataclass(slots=True)
class _Batch:
    """Mutable accumulator for one expanded layer."""

    next_layer: list[tuple[BalancedPresentation, tuple[int, ...]]] = None  # type: ignore[assignment]
    meetings: list[tuple[int, tuple[int, ...]]] = None  # type: ignore[assignment]
    expanded: int = 0
    generated: int = 0
    cap_pruned: bool = False
    budget_hit: bool = False

    def __post_init__(self) -> None:
        self.next_layer = []
        self.meetings = []


def _predecessor(state: BalancedPresentation, move: PrimitiveMove) -> BalancedPresentation:
    """Return the unique state ``p`` with ``move.apply(p) == state``.

    Every catalog move is invertible on presentations: inversion (AC2) is an
    involution, conjugation (AC3) inverts by flipping the generator sign, and
    right-multiplication (AC1) inverts by right-multiplying with the source's
    inverse. Free reduction is confluent, so re-applying ``move`` to the returned
    state reproduces ``state`` exactly.
    """
    relators = state.relators
    if isinstance(move, InvertRelatorMove):
        return state.replace_relator(move.target, relators[move.target].inverse())
    if isinstance(move, ConjugateRelatorMove):
        return state.replace_relator(
            move.target, relators[move.target].conjugate_by_letter(-move.generator)
        )
    undone = relators[move.target] * relators[move.source].inverse()
    return state.replace_relator(move.target, undone)


def _goal_states(initial: BalancedPresentation, goal_mode: str) -> tuple[BalancedPresentation, ...]:
    """Enumerate every accepting state to seed the backward frontier.

    ``exact_standard`` has one goal; ``signed_permuted_basis`` is the orbit of
    ``2^n n!`` signed permutations of the basis. Goals reuse the initial
    generator names so their content hashes line up with forward states.
    """
    rank = initial.rank
    names = initial.generator_names
    if goal_mode == "exact_standard":
        relators = tuple(FreeGroupWord((gen,), rank) for gen in range(1, rank + 1))
        return (BalancedPresentation(rank, relators, names),)
    if goal_mode == "signed_permuted_basis":
        states: list[BalancedPresentation] = []
        for perm in permutations(range(1, rank + 1)):
            for signs in product((1, -1), repeat=rank):
                relators = tuple(
                    FreeGroupWord((s * g,), rank) for s, g in zip(signs, perm, strict=True)
                )
                states.append(BalancedPresentation(rank, relators, names))
        return tuple(states)
    raise ValueError(f"unknown goal mode {goal_mode!r}")


def _apply_path(
    initial: BalancedPresentation, path: tuple[int, ...], catalog: ActionCatalog
) -> BalancedPresentation:
    """Replay a forward action-ID path from the initial presentation."""
    pres = initial
    for action_id in path:
        pres = catalog.move(action_id).apply(pres)
    return pres
