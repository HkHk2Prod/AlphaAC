from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.moves.catalog import ActionCatalog


@dataclass(frozen=True, slots=True)
class DescentConfig:
    """Budgets for one length-descent search.

    ``total_length_cap`` bounds the presentations the search may pass through --
    a descent often has to lengthen a presentation before it can shorten it, and
    this caps how far up those excursions may go. ``max_depth`` bounds the move
    sequence length, and ``max_expansions`` bounds the presentations expanded, so
    a search over a near-minimal presentation cannot run away.
    """

    total_length_cap: int = 48
    max_depth: int = 32
    max_expansions: int = 20_000


@dataclass(frozen=True, slots=True)
class DescentResult:
    """Fewest AC moves that strictly shorten a presentation.

    ``distance`` is the minimal number of catalog moves reaching a presentation
    whose total length is strictly smaller, or ``None`` when no such sequence was
    found. ``proven`` records whether the answer is exact: a found ``distance`` is
    the true minimum unless a strictly shallower branch was pruned by the length
    cap (which could hide an even shorter descent); a ``None`` distance is proven
    only when the whole reachable region under the cap was explored within budget
    -- i.e. the presentation sits at a local length minimum with no descent at all.
    """

    distance: int | None
    proven: bool
    expanded: int


def descent_distance(
    presentation: BalancedPresentation,
    catalog: ActionCatalog,
    config: DescentConfig,
) -> DescentResult:
    """Breadth-first search for the fewest moves that reduce the total length by >=1.

    States are explored in nondecreasing move count with content-hash dedup, so
    the first neighbour found with a smaller total length is reached by a
    fewest-moves sequence. Neighbours above the length cap are pruned (recording
    the shallowest prune depth so the result can report whether the minimum is
    proven), and no-op moves are skipped.
    """
    start_length = presentation.total_length
    frontier: deque[tuple[int, BalancedPresentation]] = deque([(0, presentation)])
    seen = {presentation.content_hash}
    expanded = 0
    shallowest_prune: int | None = None
    truncated = False
    while frontier:
        depth, pres = frontier.popleft()
        if depth >= config.max_depth:
            truncated = True
            continue
        if expanded >= config.max_expansions:
            truncated = True
            break
        expanded += 1
        for move in catalog.moves:
            nxt = move.apply(pres)
            if nxt.content_hash == pres.content_hash:
                continue
            if nxt.total_length < start_length:
                # First BFS hit is the fewest-move descent; it is provably minimal
                # unless a shallower branch was cap-pruned and could hide a shorter
                # one (a prune at parent depth < this depth reaches below at <= here).
                proven = shallowest_prune is None or shallowest_prune >= depth
                return DescentResult(depth + 1, proven, expanded)
            if nxt.total_length > config.total_length_cap:
                if shallowest_prune is None:
                    shallowest_prune = depth
                continue
            if nxt.content_hash in seen:
                continue
            seen.add(nxt.content_hash)
            frontier.append((depth + 1, nxt))
    # No descent found. It is proven absent only when the search settled the whole
    # reachable region: nothing was cut short by a budget or by the length cap.
    return DescentResult(None, not truncated and shallowest_prune is None, expanded)
