from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.base import PolicyValueOutput
from ac_zero.models.batch import EncodedBatch, encode_batch
from ac_zero.models.torch_utils import select_device
from ac_zero.training.ppo.losses import PolicyValueLoss, PPOBatchStats

# A replay example is duck-typed; only these attributes are read during training.
TrainingExample = Any

# Bumped when the serialized network shape changes incompatibly. v2 split the single
# scalar value head into the navigation `success`/`progress` heads, so a v1 checkpoint
# cannot load into a v2 network and carries different value semantics besides.
_MODEL_FORMAT_VERSION = 2


# Range of the `progress` head's `tanh`. The normalized shaping-return-to-go B~ is
# bounded below by roughly -(3 + 6/L0) -- the horizon is 3L+6 and each off-descent
# step contributes -1 before dividing by L0 -- and above by ~1. A scale of 4 covers
# the realistic mass (starts cluster at L0=8, floor ~-3.25); the rare tiny-L0 episode
# saturates at the extreme, where the exact value barely matters.
PROGRESS_VALUE_SCALE = 4.0


class _PolicyValueNet(nn.Module):
    """An architecture trunk with a linear policy head and three value heads.

    ``value`` is the legacy tanh scalar critic the non-navigation reward modes read.
    ``success`` (a sigmoid probability) and ``progress`` (a scaled tanh) are the
    navigation reward's ``alpha``-invariant decomposition; a run trains the two that
    match its reward mode and leaves the third untouched. The navigation heads are
    added after ``value_head`` so the trunk/policy/value parameters draw from the
    seeded RNG in the same order as before -- a non-navigation model is numerically
    unchanged by their presence.
    """

    def __init__(self, trunk: nn.Module, feature_dim: int, action_count: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.policy_head = nn.Linear(feature_dim, action_count)
        self.value_head = nn.Linear(feature_dim, 1)
        self.success_head = nn.Linear(feature_dim, 1)
        self.progress_head = nn.Linear(feature_dim, 1)

    def forward(
        self, batch: EncodedBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map a batch to ``(batch, actions)`` logits and three ``(batch,)`` value heads."""
        features = self.trunk(batch)
        logits: torch.Tensor = self.policy_head(features)
        value = torch.tanh(self.value_head(features)).reshape(-1)
        success = torch.sigmoid(self.success_head(features)).reshape(-1)
        progress = PROGRESS_VALUE_SCALE * torch.tanh(self.progress_head(features)).reshape(-1)
        return logits, value, success, progress


class TrainablePolicyValueModel(ABC):
    """Shared training machinery for the policy/value architectures.

    Subclasses contribute an architecture-specific trunk ``nn.Module`` that maps an
    :class:`EncodedBatch` to a ``(batch, feature_dim)`` feature tensor; this base
    attaches linear policy and (tanh-bounded) value heads and trains every parameter
    with PyTorch autograd. The network is built lazily on first use so the action-head
    width and any encoding-dependent dimensions (vocabulary size, token slots) come
    from real inputs.

    Everything runs on one batched path: search evaluates a single state as a batch of
    one (:meth:`apply`), while the replay, PPO, and supervised optimizers push whole
    minibatches through :meth:`forward` in a single kernel launch each -- which is what
    makes a GPU worth using at all.
    """

    architecture: str = "trainable"

    def __init__(self, *, seed: int = 0, device: str = "cpu", **hyperparameters: int) -> None:
        self.seed = seed
        self.device = select_device(device)
        self._hp: dict[str, int] = dict(hyperparameters)
        self._net: _PolicyValueNet | None = None
        self._feature_dim = 0
        self._action_count = 0
        self._pending_state: dict[str, Any] | None = None

    # -- subclass contract -------------------------------------------------
    @abstractmethod
    def _build_trunk(self, batch: EncodedBatch) -> tuple[nn.Module, int]:
        """Create the trunk module and return its output feature dimension."""

    # -- build -------------------------------------------------------------
    def ensure_built(self, batch: EncodedBatch, action_count: int) -> None:
        """Build the network from the first real batch, if it does not exist yet."""
        if self._net is not None:
            if action_count != self._action_count:
                raise ValueError("action_count changed after the model was built")
            return
        # Seed the global RNG so the lazily created layers initialize
        # reproducibly from the model seed.
        torch.manual_seed(self.seed)
        trunk, feature_dim = self._build_trunk(batch)
        self._net = _PolicyValueNet(trunk, feature_dim, action_count)
        self._feature_dim = feature_dim
        self._action_count = action_count
        if self._pending_state is not None:
            self._net.load_state_dict(
                {name: torch.tensor(value) for name, value in self._pending_state.items()}
            )
            self._pending_state = None
        self._net.to(self.device)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters, or 0 before the first batch builds the network."""
        if self._net is None:
            return 0
        return sum(p.numel() for p in self._net.parameters() if p.requires_grad)

    def parameters(self) -> list[torch.nn.Parameter]:
        """The built network's parameters, for a caller-owned optimizer."""
        if self._net is None:
            raise RuntimeError("the model has no parameters until it is built on a batch")
        return list(self._net.parameters())

    # -- inference and training -------------------------------------------
    def encode(self, encodings: list[PaddedEncoding]) -> EncodedBatch:
        """Stack encoded states into a batch on this model's device."""
        return encode_batch(encodings, self.device)

    def forward(
        self, batch: EncodedBatch, action_count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable ``(logits, value, success, progress)`` for a batch, building if needed."""
        self.ensure_built(batch, action_count)
        assert self._net is not None
        logits, value, success, progress = self._net(batch)
        return logits, value, success, progress

    def apply(self, encoding: PaddedEncoding, action_count: int) -> PolicyValueOutput:
        """Predict policy logits and the three value heads for one encoded state."""
        batch = self.encode([encoding])
        self.ensure_built(batch, action_count)
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            logits, value, success, progress = self._net(batch)
        return PolicyValueOutput(
            logits[0].cpu().numpy().astype(np.float64),
            float(value[0].item()),
            float(success[0].item()),
            float(progress[0].item()),
        )

    def train_batch(
        self,
        batch: list[TrainingExample],
        *,
        learning_rate: float,
        value_loss_weight: float,
        grad_clip: float,
        reward_mode: str,
    ) -> PolicyValueLoss:
        """Apply one averaged gradient step and return mean pre-update losses."""
        if not batch:
            raise ValueError("batch must not be empty")
        encoded = self.encode([example.encoding for example in batch])
        legal = self._legal(batch)
        self.ensure_built(encoded, legal.shape[1])
        assert self._net is not None
        self._net.train()
        # Plain SGD (no momentum) applies one averaged gradient step: p -= lr * grad.
        optimizer = torch.optim.SGD(self._net.parameters(), lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)

        logits, value, success, progress = self._net(encoded)
        targets = self._stack(batch, "policy_target", legal.shape[1])
        log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
        # Illegal actions carry no target mass, so zeroing their (-inf) log-probs
        # keeps them out of the sum instead of turning it into a NaN.
        policy = -(targets * log_probs.nan_to_num(neginf=0.0)).sum(dim=1)
        value_err = self._value_error(batch, value, success, progress, reward_mode, "value_target")
        (policy + value_loss_weight * value_err).mean().backward()  # type: ignore[no-untyped-call]
        self._clip_gradients(grad_clip)
        optimizer.step()

        return PolicyValueLoss(
            policy_loss=float(policy.mean().item()),
            value_loss=float(value_err.mean().item()),
            total_loss=float((policy + value_loss_weight * value_err).mean().item()),
        )

    def _value_error(
        self,
        batch: list[TrainingExample],
        value: torch.Tensor,
        success: torch.Tensor,
        progress: torch.Tensor,
        reward_mode: str,
        scalar_field: str,
    ) -> torch.Tensor:
        """Per-example value squared error against the heads the reward mode trains.

        Navigation regresses the two ``alpha``-invariant heads against their own
        return-to-go targets; every other mode regresses the legacy scalar critic
        against ``scalar_field`` (the replay's ``value_target``, PPO's
        ``return_target``) and leaves the navigation heads untouched. The two
        navigation errors are summed with equal weight -- ``success`` lands in
        ``[0, 1]`` and ``progress`` in roughly ``[-4, 1]``, so the progress head
        carries the larger share of the loss, which matches its larger role while a
        scratch run has no successes for the success head to fit.
        """
        if reward_mode == "navigation":
            success_targets = self._column(batch, "success_target")
            progress_targets = self._column(batch, "progress_target")
            return (success - success_targets) ** 2 + (progress - progress_targets) ** 2
        return (value - self._column(batch, scalar_field)) ** 2

    def ppo_update(
        self,
        batch: list[TrainingExample],
        *,
        learning_rate: float,
        clip_ratio: float,
        value_weight: float,
        entropy_weight: float,
        grad_clip: float,
        reward_mode: str,
    ) -> PPOBatchStats:
        """Apply one clipped-surrogate PPO gradient step over a minibatch.

        Each example carries the log-probability and advantage recorded when the
        action was sampled, plus the value-head return targets. The loss is the
        standard PPO objective: a clipped policy-ratio surrogate, a value
        regression (against the heads this reward mode trains, see
        :meth:`_value_error`), and an entropy bonus, averaged over the minibatch.
        """
        if not batch:
            raise ValueError("batch must not be empty")
        encoded = self.encode([example.encoding for example in batch])
        legal = self._legal(batch)
        self.ensure_built(encoded, legal.shape[1])
        assert self._net is not None
        self._net.train()
        optimizer = torch.optim.SGD(self._net.parameters(), lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)

        logits, value, success, progress = self._net(encoded)
        actions = torch.tensor(
            [int(example.action) for example in batch], dtype=torch.long, device=self.device
        )
        old_log_prob = self._column(batch, "old_log_prob")
        advantage = self._column(batch, "advantage")

        log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
        log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(log_prob - old_log_prob)
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surrogate = -torch.minimum(ratio * advantage, clipped * advantage)
        value_loss = self._value_error(
            batch, value, success, progress, reward_mode, "return_target"
        )
        finite = log_probs.nan_to_num(neginf=0.0)
        entropy = -(finite.exp() * finite).sum(dim=1)
        loss = surrogate + value_weight * value_loss - entropy_weight * entropy
        loss.mean().backward()  # type: ignore[no-untyped-call]
        self._clip_gradients(grad_clip)
        optimizer.step()

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
            approx_kl = (old_log_prob - log_prob).mean()
        return PPOBatchStats(
            policy_loss=float(surrogate.mean().item()),
            value_loss=float(value_loss.mean().item()),
            entropy=float(entropy.mean().item()),
            total_loss=float(loss.mean().item()),
            clip_fraction=float(clip_fraction.item()),
            approx_kl=float(approx_kl.item()),
        )

    def _clip_gradients(self, grad_clip: float) -> None:
        """Clip the accumulated gradient norm in place; ``0`` disables clipping.

        The RL losses ran unclipped while the supervised one clipped, so a single
        pathological minibatch -- an advantage spike after a reward-scale change,
        say -- could take the whole policy out in one step.
        """
        if grad_clip > 0.0:
            assert self._net is not None
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), grad_clip)

    # -- batch assembly ----------------------------------------------------
    def _legal(self, batch: list[TrainingExample]) -> torch.Tensor:
        """Stack the per-example legal-action masks into a ``(batch, actions)`` tensor."""
        mask = np.asarray([example.legal_mask for example in batch], dtype=np.bool_)
        return torch.from_numpy(mask).to(self.device)

    def _stack(self, batch: list[TrainingExample], field: str, actions: int) -> torch.Tensor:
        rows = np.asarray([getattr(example, field) for example in batch], dtype=np.float32)
        if rows.shape[1] != actions:
            raise ValueError(f"{field} width {rows.shape[1]} does not match {actions} actions")
        return torch.from_numpy(rows).to(self.device)

    def _column(self, batch: list[TrainingExample], field: str) -> torch.Tensor:
        column = np.asarray([float(getattr(example, field)) for example in batch], dtype=np.float32)
        return torch.from_numpy(column).to(self.device)

    # -- serialization -----------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        state = self._net.state_dict() if self._net is not None else {}
        parameters = {name: value.detach().cpu().numpy().tolist() for name, value in state.items()}
        return {
            "architecture": self.architecture,
            "format_version": _MODEL_FORMAT_VERSION,
            "hyperparameters": {"seed": self.seed, **self._hp},
            "built": self._net is not None,
            "feature_dim": self._feature_dim,
            "action_count": self._action_count,
            "parameters": parameters,
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Stage parameters written by :meth:`to_json` for the next lazy build.

        The trunk shape depends on the first batch (vocabulary size, token slots), so
        the weights are applied inside :meth:`ensure_built` once the network exists.

        A checkpoint from before the value head was split into ``success``/``progress``
        is rejected here with a clear message rather than left to fail deep in a
        ``state_dict`` load on the missing head weights: its value semantics differ,
        so it must be re-pretrained rather than loaded.
        """
        if not data.get("built", False):
            return
        version = int(data.get("format_version", 1))
        if version != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint model format v{version} predates the success/progress value "
                f"heads (v{_MODEL_FORMAT_VERSION}); its single scalar value head is "
                "incompatible -- re-run supervised pretraining to produce a v"
                f"{_MODEL_FORMAT_VERSION} checkpoint."
            )
        self._feature_dim = int(data["feature_dim"])
        self._action_count = int(data["action_count"])
        self._pending_state = data["parameters"]
