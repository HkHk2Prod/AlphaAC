from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class ReplayBuffer(Generic[T]):
    """Bounded in-memory replay buffer with deterministic sampling support."""

    capacity: int
    _items: deque[T] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate capacity and allocate the bounded deque."""
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items = deque(maxlen=self.capacity)

    def add(self, item: T) -> None:
        """Append one item, evicting the oldest item if the buffer is full."""
        self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        """Append a sequence of items in order."""
        for item in items:
            self.add(item)

    def sample(self, batch_size: int, rng: random.Random) -> list[T]:
        """Sample up to `batch_size` unique items using the provided RNG."""
        return rng.sample(list(self._items), min(batch_size, len(self._items)))

    def __len__(self) -> int:
        """Return the number of currently stored items."""
        return len(self._items)
