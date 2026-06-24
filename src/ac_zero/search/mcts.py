from __future__ import annotations

from dataclasses import dataclass

from ac_zero.environment.env import ACEnvironment


@dataclass(frozen=True, slots=True)
class MCTSStats:
    """Search accounting returned by a root MCTS call."""

    visit_counts: tuple[int, ...]
    expanded_nodes: int
    model_evaluations: int


class UniformMCTS:
    """Deterministic uniform-prior baseline that spreads visits over legal actions.

    This is the model-free reference; `search.puct.PUCTMCTS` is the real
    model-guided PUCT search used for training targets and the `puct` agent.
    """

    def __init__(self, simulations: int = 16) -> None:
        """Create a uniform-prior search with a fixed simulation count."""
        self.simulations = simulations

    def search(self, env: ACEnvironment) -> MCTSStats:
        """Assign visits round-robin across legal actions for smoke testing.

        This is intentionally not a full PUCT implementation yet; it supplies
        deterministic visit counts and budget accounting for CLI and tests.
        """

        mask = env.legal_action_mask()
        counts = [0 for _ in mask]
        legal = [i for i, ok in enumerate(mask) if ok]
        if not legal or self.simulations <= 0:
            return MCTSStats(tuple(counts), 0, 0)
        for sim in range(self.simulations):
            action = legal[sim % len(legal)]
            counts[action] += 1
        return MCTSStats(tuple(counts), len(legal), 0)

    def select_action(self, env: ACEnvironment) -> int:
        """Return the most-visited root action with deterministic tie-breaking."""
        stats = self.search(env)
        if not any(stats.visit_counts):
            raise RuntimeError("no legal actions")
        return max(range(len(stats.visit_counts)), key=lambda i: (stats.visit_counts[i], -i))
