"""The supervised optimizer: fit the policy to the dataset's known descent directions.

Unlike the RL backends -- which rebuild a plain SGD optimizer per minibatch because
each one sees freshly self-played data -- this is ordinary supervised learning over a
fixed dataset, so it keeps one Adam optimizer for the whole run and lets its moment
estimates accumulate.

Evaluation asks the question the task is actually posed in: take the move the model
ranks first and look up what it does to the distance to the origin. ``descent_accuracy``
is how often that move is on a shortest path (delta == -1); ``mean_delta`` is the
average distance change it causes. Every trainable group has at least one descent move
by construction, so a perfect model scores 1.0 and -1.0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from ac_zero.datasets.supervised_store import DELTA_UNKNOWN
from ac_zero.models.trainable import TrainablePolicyValueModel
from ac_zero.training.supervised.batches import LabelledBatch, SupervisedBatches


@dataclass(frozen=True, slots=True)
class SupervisedLoss:
    """Mean losses over one minibatch."""

    policy_loss: float
    value_loss: float
    total_loss: float


@dataclass(frozen=True, slots=True)
class SupervisedMetrics:
    """How a model scores on one split.

    ``descent_accuracy`` is the share of groups whose top-ranked move steps strictly
    closer to the trivial group; ``mean_delta`` the mean distance change that move
    causes; ``unknown_rate`` the share where it leaves the annotated region entirely
    (a move the dataset cannot score, and so cannot credit).
    """

    policy_loss: float
    value_loss: float
    descent_accuracy: float
    mean_delta: float
    unknown_rate: float
    groups: int

    def as_metrics(self, prefix: str) -> dict[str, float | int | bool | str]:
        """Flatten into `<prefix>_<name>` rows for the event log and plots."""
        return {
            f"{prefix}_policy_loss": round(self.policy_loss, 6),
            f"{prefix}_value_loss": round(self.value_loss, 6),
            f"{prefix}_descent_accuracy": round(self.descent_accuracy, 4),
            f"{prefix}_mean_delta": round(self.mean_delta, 4),
            f"{prefix}_unknown_rate": round(self.unknown_rate, 4),
            f"{prefix}_groups": self.groups,
        }


class SupervisedTrainer:
    """Trains one policy/value model against a labelled dataset split."""

    def __init__(
        self,
        model: TrainablePolicyValueModel,
        batches: SupervisedBatches,
        *,
        actions: int,
        learning_rate: float,
        value_loss_weight: float,
        grad_clip: float,
        warmup_steps: int = 0,
    ) -> None:
        self._model = model
        self._batches = batches
        self._actions = actions
        self._value_weight = value_loss_weight
        self._grad_clip = grad_clip
        self._learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._updates = 0
        self._optimizer: torch.optim.Optimizer | None = None

    def _losses(
        self,
        batch: LabelledBatch,
        logits: torch.Tensor,
        success: torch.Tensor,
        progress: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-example cross-entropy against the move target, and value squared error.

        The softmax spans *every* action, not just the labelled ones: an unlabelled
        move carries zero target mass, so probability the model puts there is
        probability taken away from the moves that descend. The value error pretrains
        the two navigation heads the RL runs read -- ``success`` toward ``gamma**d``
        and ``progress`` toward the descent's normalized shaping-return B~ -- summed
        with equal weight, matching :meth:`TrainablePolicyValueModel._value_error`.
        """
        device = self._model.device
        targets = torch.from_numpy(batch.policy_targets).to(device)
        success_targets = torch.from_numpy(batch.success_targets).to(device)
        progress_targets = torch.from_numpy(batch.progress_targets).to(device)
        policy = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
        value = (success - success_targets) ** 2 + (progress - progress_targets) ** 2
        return policy, value

    def step(self, split: str, batch_size: int, rng: random.Random) -> SupervisedLoss:
        """Draw one minibatch and apply a single Adam update."""
        batch = self._batches.sample(split, batch_size, rng)
        logits, _, success, progress = self._model.forward(
            self._model.encode(batch.encodings), self._actions
        )
        policy, value = self._losses(batch, logits, success, progress)
        loss = (policy + self._value_weight * value).mean()

        optimizer = self._ensure_optimizer()
        self._updates += 1
        self._apply_warmup(optimizer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        if self._grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
        optimizer.step()
        return SupervisedLoss(
            policy_loss=float(policy.mean().item()),
            value_loss=float(value.mean().item()),
            total_loss=float(loss.item()),
        )

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        """Bind Adam to the network, which only exists once a batch has built it."""
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        return self._optimizer

    def _apply_warmup(self, optimizer: torch.optim.Optimizer) -> None:
        """Scale the learning rate linearly from 0 to its target across the warmup.

        ``self._updates`` is the 1-based index of the update about to be applied, so the
        first step already carries a small nonzero rate and step ``warmup_steps`` reaches
        the full rate; every later step holds it there.
        """
        if self._warmup_steps <= 0:
            return
        scale = min(1.0, self._updates / self._warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = self._learning_rate * scale

    def evaluate(self, batches: list[LabelledBatch]) -> SupervisedMetrics:
        """Score the model on already-drawn batches, without touching its weights."""
        if not batches:
            raise ValueError("cannot evaluate on no batches")
        policy_loss = value_loss = 0.0
        descents = deltas = unknown = 0.0
        total = 0
        for batch in batches:
            with torch.no_grad():
                encoded = self._model.encode(batch.encodings)
                logits, _, success, progress = self._model.forward(encoded, self._actions)
                policy, value = self._losses(batch, logits, success, progress)
                chosen = logits.argmax(dim=1).cpu().numpy()
            picked = np.take_along_axis(batch.deltas, chosen[:, None], axis=1).ravel()
            known = picked != DELTA_UNKNOWN
            policy_loss += float(policy.sum().item())
            value_loss += float(value.sum().item())
            descents += float(np.count_nonzero(picked == -1))
            deltas += float(picked[known].sum())
            unknown += float(np.count_nonzero(~known))
            total += batch.size
        # `mean_delta` averages over the moves the dataset can score; a model that only
        # ever picks unlabelled moves has no delta to report, so it reads as 0 and
        # `unknown_rate` is the number that tells you so.
        scored = total - unknown
        return SupervisedMetrics(
            policy_loss=policy_loss / total,
            value_loss=value_loss / total,
            descent_accuracy=descents / total,
            mean_delta=deltas / scored if scored else 0.0,
            unknown_rate=unknown / total,
            groups=total,
        )

    def sample_batches(
        self, split: str, count: int, batch_size: int, seed: int
    ) -> list[LabelledBatch]:
        """Draw a fixed set of validation batches, identical on every epoch.

        Scoring each epoch against the *same* groups means a change in the metric is a
        change in the model, not a change in the sample.
        """
        rng = random.Random(seed)
        return [self._batches.sample(split, batch_size, rng) for _ in range(count)]
