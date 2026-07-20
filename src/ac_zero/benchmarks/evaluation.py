"""Score a checkpoint against a benchmark catalog.

Two passes, cheapest first:

1. **Scan** -- length-ordered greedy best-first search on every entry, under a
   small node budget. It costs little per presentation, so the whole catalog is
   covered early and the run always has a complete picture, however little time
   it ends up getting.
2. **Deep** -- model-guided PUCT on the entries the scan did not solve, in the
   catalog's smallest-first order. This is where a trained model earns its
   keep, and where the remaining budget goes.

Both passes stop at the wall-clock cap on an entry boundary, so a truncated run
reports exactly what it scored rather than losing the tail. Which entries went
unattempted is recorded, so a solve rate is never quietly computed over a subset.

A solved entry's full move ``path`` is recorded, which is the replayable witness
for the solve; this module does not itself emit or verify certificates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ac_zero.agents.base import SolverResult
from ac_zero.agents.greedy import GreedyBestFirstConfig, GreedyBestFirstSearch
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.benchmarks.catalog import BenchmarkCatalog
from ac_zero.benchmarks.config import DEEP_AGENT, SCAN_AGENT, BenchmarkConfig
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.base import PolicyValueModel
from ac_zero.search.puct import PUCTMCTS, PUCTConfig

Logger = Callable[[str], None]


@dataclass(slots=True)
class EntryResult:
    """One presentation's outcome under whichever pass last touched it."""

    presentation_id: str
    family: str
    total_length: int
    solved: bool
    agent: str
    moves: int
    path: tuple[int, ...]
    best_reduction: int
    termination_reason: str
    seconds: float
    expanded_nodes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "presentation_id": self.presentation_id,
            "family": self.family,
            "total_length": self.total_length,
            "solved": self.solved,
            "agent": self.agent,
            "moves": self.moves,
            "path": list(self.path),
            "best_reduction": self.best_reduction,
            "termination_reason": self.termination_reason,
            "seconds": round(self.seconds, 3),
            "expanded_nodes": self.expanded_nodes,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """Every entry's outcome plus the counts that summarize the run."""

    catalog_name: str
    checkpoint_name: str
    results: list[EntryResult] = field(default_factory=list)
    attempted: int = 0
    seconds: float = 0.0
    deep_pass_ran: bool = False
    stopped_early: bool = False
    # Entries the encoder could not represent, so never asked. Distinct from
    # unsolved, and excluded from `attempted`.
    out_of_capacity: int = 0

    @property
    def solved(self) -> list[EntryResult]:
        return [r for r in self.results if r.solved]

    @property
    def solve_rate(self) -> float:
        """Share of *attempted* entries solved (0.0 when nothing was attempted)."""
        return len(self.solved) / self.attempted if self.attempted else 0.0

    def counts_by_family(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for result in self.results:
            bucket = counts.setdefault(result.family, {"attempted": 0, "solved": 0})
            bucket["attempted"] += 1
            bucket["solved"] += int(result.solved)
        return counts


class BenchmarkEvaluator:
    """Run a catalog against one checkpoint under a configured budget."""

    def __init__(
        self,
        catalog: BenchmarkCatalog,
        config: BenchmarkConfig,
        model: PolicyValueModel | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.model = model
        self.encoder = StateEncoder(config.max_relator_tokens)

    def _env(self, presentation: BalancedPresentation, max_moves: int) -> ACEnvironment:
        return ACEnvironment(
            presentation,
            ACEnvironmentConfig(
                max_moves=max_moves,
                goal_mode=self.config.goal_mode,
                moveset=self.config.moveset,
            ),
            encoder=self.encoder,
        )

    def _scan(self, presentation: BalancedPresentation) -> SolverResult:
        """Classical length-ordered search, no model involved."""
        search = GreedyBestFirstSearch(
            GreedyBestFirstConfig(
                max_expansions=self.config.scan_expansions,
                max_generated=self.config.scan_generated,
            )
        )
        return search.solve(
            presentation,
            env_template=self._env(presentation, self.config.max_moves),
            experiment_id="benchmark-scan",
        )

    def _deep(self, presentation: BalancedPresentation) -> SolverResult:
        """Follow the model's PUCT visit counts until the goal or the horizon."""
        if self.model is None:
            raise RuntimeError("the deep pass needs a model")
        env = self._env(presentation, self.config.deep_moves)
        mcts = PUCTMCTS(
            self.model, self.encoder, PUCTConfig(simulations=self.config.deep_simulations)
        )
        path: list[int] = []
        terminated = False
        while len(path) < env.config.max_moves:
            if not any(env.legal_action_mask()):
                break
            action = mcts.select_action(env)
            path.append(action)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return SolverResult(
            best_state=env.state.presentation,
            best_reduction=presentation.total_length - env.state.presentation.total_length,
            path=tuple(path),
            expanded_nodes=len(path) * self.config.deep_simulations,
            generated_nodes=len(path) * self.config.deep_simulations,
            peak_frontier_size=1,
            termination_reason="goal" if terminated else "horizon",
            success=terminated,
        )

    def _record(
        self,
        presentation: BalancedPresentation,
        result: SolverResult,
        agent: str,
        seconds: float,
    ) -> EntryResult:
        return EntryResult(
            presentation_id=str(presentation.presentation_id),
            family=str(presentation.provenance.get("family", "unknown")),
            total_length=presentation.total_length,
            solved=result.success,
            agent=agent,
            moves=len(result.path),
            path=result.path,
            best_reduction=result.best_reduction,
            termination_reason=result.termination_reason,
            seconds=seconds,
            expanded_nodes=result.expanded_nodes,
        )

    def _within_capacity(self, *, log: Logger) -> list[BalancedPresentation]:
        """Drop entries no model of this encoder capacity could even represent.

        The encoder's capacity is a hard contract -- it raises rather than
        truncating, since a clipped relator is a different presentation -- so an
        over-long entry would abort the run partway. Skipping it is right, but it
        must be *reported* rather than counted as unsolved: the model was never
        asked. Normally the count is zero, because the catalog bound and the
        encoder capacity are set to the same number.
        """
        capacity = self.config.max_relator_tokens
        keep, dropped = [], 0
        for presentation in self.catalog.entries:
            if max(len(relator) for relator in presentation.relators) > capacity:
                dropped += 1
                continue
            keep.append(presentation)
        if dropped:
            log(
                f"[benchmark] skipping {dropped} entries longer than the encoder capacity "
                f"({capacity} tokens); they are reported as out-of-capacity, not unsolved"
            )
        return keep

    def run(self, *, log: Logger = print) -> BenchmarkReport:
        """Scan every entry, then deepen the unsolved ones until the budget ends."""
        report = BenchmarkReport(self.catalog.name, self.config.checkpoint_name)
        started = time.monotonic()
        deadline = self.config.deadline_seconds

        def out_of_time() -> bool:
            return deadline is not None and time.monotonic() - started >= deadline

        scorable = self._within_capacity(log=log)
        report.out_of_capacity = len(self.catalog.entries) - len(scorable)

        # presentation_id -> its slot in report.results, so the deep pass can
        # overwrite an entry in place without scanning the list for it.
        slot_of: dict[str, int] = {}
        for index, presentation in enumerate(scorable):
            if out_of_time():
                report.stopped_early = True
                log(f"[benchmark] budget spent after {index}/{len(scorable)} scanned")
                break
            entry_started = time.monotonic()
            result = self._scan(presentation)
            entry = self._record(presentation, result, SCAN_AGENT, time.monotonic() - entry_started)
            slot_of[entry.presentation_id] = len(report.results)
            report.results.append(entry)

        report.attempted = len(report.results)
        log(
            f"[benchmark] scan: {len(report.solved)}/{report.attempted} solved "
            f"({report.solve_rate:.1%})"
        )

        # Only what the scan actually reached: entries it never got to are
        # unattempted, not unsolved, and the deep pass does not invent coverage.
        unsolved = [
            presentation
            for presentation in scorable
            if str(presentation.presentation_id) in slot_of
            and not report.results[slot_of[str(presentation.presentation_id)]].solved
        ]
        if self.model is None:
            log("[benchmark] no checkpoint model; skipping the deep pass")
            report.seconds = time.monotonic() - started
            return report

        report.deep_pass_ran = True
        gained = 0
        for index, presentation in enumerate(unsolved):
            if out_of_time():
                report.stopped_early = True
                log(f"[benchmark] budget spent after {index}/{len(unsolved)} deep searches")
                break
            entry_started = time.monotonic()
            result = self._deep(presentation)
            if not result.success:
                continue
            # Only an improvement replaces the scan's record: a deep search that
            # also fails says nothing the cheaper pass did not already say.
            gained += 1
            entry = self._record(presentation, result, DEEP_AGENT, time.monotonic() - entry_started)
            report.results[slot_of[entry.presentation_id]] = entry

        report.seconds = time.monotonic() - started
        log(
            f"[benchmark] deep pass solved {gained} more; "
            f"{len(report.solved)}/{report.attempted} total ({report.solve_rate:.1%})"
        )
        return report


def load_checkpoint_model(path: str | Path) -> PolicyValueModel:
    """Rebuild the trained model from a checkpoint bundle file (``best.json``)."""
    from ac_zero.models.registry import model_from_json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return model_from_json(data.get("model_state", data))
