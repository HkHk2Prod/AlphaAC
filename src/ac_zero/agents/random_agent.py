from __future__ import annotations

import random
from dataclasses import dataclass

from ac_zero.agents.base import Agent


@dataclass(slots=True)
class RandomLegalActionAgent(Agent):
    """Uniformly sample from the currently legal action IDs."""

    rng: random.Random

    def select_action(self, mask: tuple[bool, ...]) -> int:
        """Return a random legal action using the injected deterministic RNG."""
        legal = [i for i, ok in enumerate(mask) if ok]
        if not legal:
            raise RuntimeError("no legal actions")
        return self.rng.choice(legal)
