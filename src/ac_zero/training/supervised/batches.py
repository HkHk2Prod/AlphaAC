"""Turn a labelled dataset split into minibatches of encoded states and move targets.

One example is one group, presented to the model exactly as self-play would present
it at the *start* of an episode: the group is the problem, its horizon is the ``3L+6``
its own distance earns it, and nothing has been played yet. The targets come from the
supervised sidecar's per-move distance deltas:

* **Policy** -- ``softmax(-delta / temperature)`` over the moves whose neighbour has a
  known distance, and zero elsewhere. The zeros are *not* masked out of the model's
  softmax, so mass placed on a move the dataset knows nothing about is penalized like
  mass placed on a bad one.
* **Value** -- ``2 * gamma**distance - 1``: the tanh-bounded value head's rendering of
  "how far from the trivial group is this", matching the discounting the RL backends
  give a path of that length, so a pretrained critic warm-starts them usefully.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ac_zero.datasets.instance_store import InstanceStore
from ac_zero.datasets.supervised_store import DELTA_UNKNOWN, SupervisedStore
from ac_zero.encoding.padded import PaddedEncoding, StateEncoder
from ac_zero.environment.state import ACSearchState
from ac_zero.training.pipeline.pipeline_episodes import moves_for_distance


@dataclass(frozen=True, slots=True)
class LabelledBatch:
    """One minibatch: encoded states, their move targets, and the raw deltas to score."""

    encodings: list[PaddedEncoding]
    policy_targets: NDArray[np.float32]  # (batch, actions)
    # The two navigation value-head targets an optimal descent of length `d` implies:
    # `success` = gamma**d (the discounted-success value of a state d steps from the
    # origin) and `progress` = B~, the shaping return-to-go of that descent normalized
    # by d. Pretraining these seeds the exact heads the navigation RL runs read.
    success_targets: NDArray[np.float32]  # (batch,)
    progress_targets: NDArray[np.float32]  # (batch,)
    # The per-move distance deltas the targets were built from, kept so evaluation can
    # ask the question the task is really about: what does the move the model picked do
    # to the distance to the origin?
    deltas: NDArray[np.int16]  # (batch, actions)

    @property
    def size(self) -> int:
        return len(self.encodings)


def _head_targets(
    distances: NDArray[np.float64], gamma: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The (success, progress) head targets an optimal descent of each length implies.

    These match the return-to-go a d-step optimal descent produces under the RL
    collectors' convention (the first transition discounted by ``gamma**0``), so the
    pretrained heads seed exactly what the runs regress:

    * ``success = gamma**(d-1)`` -- the goal reward lands on the d-th (last)
      transition, discounted from the start by ``gamma**(d-1)``; ``1`` at the origin.
    * ``progress = B~ = (1 - gamma**d) / ((1 - gamma) * d)`` -- the discounted sum of
      one clipped +1 per descent step, normalized by ``d``; ``0`` at the origin.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        success = np.where(distances > 0, gamma ** (distances - 1.0), 1.0)
        shaping = (1.0 - gamma**distances) / ((1.0 - gamma) * distances)
    progress = np.where(distances > 0, shaping, 0.0)
    return success.astype(np.float32), progress.astype(np.float32)


def policy_targets(deltas: NDArray[np.int16], temperature: float) -> NDArray[np.float32]:
    """Softmax the negated distance deltas of the known moves; zero the unknown ones.

    Rows are handled together: the unknown entries are pushed to ``-inf`` before the
    softmax, which lands them at exactly zero probability without a per-row gather.
    """
    known = deltas != DELTA_UNKNOWN
    scores = np.where(known, -deltas.astype(np.float64) / temperature, -np.inf)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.where(known, np.exp(scores), 0.0)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


class SupervisedBatches:
    """Draws labelled minibatches from the splits of one dataset."""

    def __init__(
        self,
        instances: InstanceStore,
        labels: SupervisedStore,
        encoder: StateEncoder,
        *,
        temperature: float,
        gamma: float,
        catalog_version: str,
    ) -> None:
        self._instances = instances
        self._labels = labels
        self._encoder = encoder
        self._temperature = temperature
        self._gamma = gamma
        self._catalog_version = catalog_version
        self._splits = {name: labels.trainable(name) for name in ("train", "val", "test")}
        for name, rows in self._splits.items():
            if not rows.size:
                raise ValueError(
                    f"the {name!r} split has no labelled group: every group in it is "
                    "unexpanded, unannotated, or the origin itself"
                )

    def size(self, split: str) -> int:
        """How many labelled groups the split holds."""
        return int(self._splits[split].size)

    def sample(self, split: str, batch_size: int, rng: random.Random) -> LabelledBatch:
        """Draw ``batch_size`` groups from ``split``, with replacement."""
        rows = self._splits[split]
        drawn = [int(rows[rng.randrange(rows.size)]) for _ in range(batch_size)]
        return self.rows(drawn)

    def epoch(self, split: str, batch_size: int) -> list[LabelledBatch]:
        """Cut the whole split into consecutive batches, for a deterministic sweep.

        Used to score the held-out test split once at the end of a run: every labelled
        group is seen exactly once, so the number does not depend on a sampling seed.
        """
        rows = self._splits[split]
        return [
            self.rows([int(row) for row in rows[start : start + batch_size]])
            for start in range(0, rows.size, batch_size)
        ]

    def rows(self, indices: list[int]) -> LabelledBatch:
        """Assemble the batch for these group rows."""
        deltas = self._labels.deltas[indices]
        distances = self._labels.distances[indices].astype(np.float64)
        success, progress = _head_targets(distances, self._gamma)
        return LabelledBatch(
            encodings=[self._encode(index) for index in indices],
            policy_targets=policy_targets(deltas, self._temperature),
            success_targets=success,
            progress_targets=progress,
            deltas=deltas,
        )

    def _encode(self, index: int) -> PaddedEncoding:
        """Encode one group as the start state of the episode it would seed."""
        presentation = self._instances.presentation(index)
        length = presentation.total_length
        state = ACSearchState(
            presentation=presentation,
            initial_length=length,
            best_length=length,
            moves_used=0,
            moves_remaining=moves_for_distance(int(self._labels.distances[index])),
            catalog_version=self._catalog_version,
        )
        return self._encoder.encode(state)
