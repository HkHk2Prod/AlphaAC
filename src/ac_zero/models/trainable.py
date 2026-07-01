from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.base import PolicyValueOutput
from ac_zero.training.losses import PolicyValueLoss, PPOBatchStats, policy_value_loss

# A replay example is duck-typed; only these attributes are read during training.
TrainingExample = Any


class _PolicyValueNet(nn.Module):
    """An architecture trunk with shared linear policy and (tanh) value heads."""

    def __init__(self, trunk: nn.Module, feature_dim: int, action_count: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.policy_head = nn.Linear(feature_dim, action_count)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, encoding: PaddedEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(encoding)
        logits = self.policy_head(features)[0]
        value = torch.tanh(self.value_head(features)).reshape(())
        return logits, value


class TrainablePolicyValueModel(ABC):
    """Shared training machinery for the CPU policy/value architectures.

    Subclasses contribute an architecture-specific trunk ``nn.Module`` that maps an
    encoded state to a ``(1, feature_dim)`` feature tensor; this base attaches linear
    policy and (tanh-bounded) value heads and trains every parameter with PyTorch
    autograd. The network is built lazily on first use so the action-head width and
    any encoding-dependent dimensions (e.g. vocabulary size) come from real inputs.
    """

    architecture: str = "trainable"

    def __init__(self, *, seed: int = 0, **hyperparameters: int) -> None:
        self.seed = seed
        self._hp: dict[str, int] = dict(hyperparameters)
        self._net: _PolicyValueNet | None = None
        self._feature_dim = 0
        self._action_count = 0
        self._pending_state: dict[str, Any] | None = None

    # -- subclass contract -------------------------------------------------
    @abstractmethod
    def _build_trunk(self, encoding: PaddedEncoding) -> tuple[nn.Module, int]:
        """Create the trunk module and return its output feature dimension."""

    # -- build -------------------------------------------------------------
    def _ensure_built(self, encoding: PaddedEncoding, action_count: int) -> None:
        if self._net is not None:
            if action_count != self._action_count:
                raise ValueError("action_count changed after the model was built")
            return
        # Seed the global RNG so the lazily created layers initialize
        # reproducibly from the model seed.
        torch.manual_seed(self.seed)
        trunk, feature_dim = self._build_trunk(encoding)
        self._net = _PolicyValueNet(trunk, feature_dim, action_count)
        self._feature_dim = feature_dim
        self._action_count = action_count
        if self._pending_state is not None:
            self._net.load_state_dict(
                {name: torch.tensor(value) for name, value in self._pending_state.items()}
            )
            self._pending_state = None

    # -- inference and training -------------------------------------------
    def apply(self, encoding: PaddedEncoding, action_count: int) -> PolicyValueOutput:
        """Predict policy logits and a bounded value for one encoded state."""
        self._ensure_built(encoding, action_count)
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            logits, value = self._net(encoding)
        return PolicyValueOutput(logits.numpy().astype(np.float64), float(value))

    def train_batch(
        self,
        batch: list[TrainingExample],
        *,
        learning_rate: float,
        value_loss_weight: float,
    ) -> PolicyValueLoss:
        """Apply one averaged gradient step and return mean pre-update losses."""
        if not batch:
            raise ValueError("batch must not be empty")
        self._ensure_built(batch[0].encoding, len(batch[0].legal_mask))
        assert self._net is not None
        self._net.train()
        # Plain SGD (no momentum) applies one averaged gradient step: p -= lr * grad.
        optimizer = torch.optim.SGD(self._net.parameters(), lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)

        policy_loss = 0.0
        value_loss = 0.0
        example_losses: list[torch.Tensor] = []
        for example in batch:
            logits, value = self._net(example.encoding)
            reported = policy_value_loss(
                logits.detach().numpy(),
                float(value.detach()),
                example.policy_target,
                example.value_target,
                example.legal_mask,
                value_weight=value_loss_weight,
            )
            policy_loss += reported.policy_loss
            value_loss += reported.value_loss
            example_losses.append(_example_loss(logits, value, example, value_loss_weight))

        torch.stack(example_losses).mean().backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        n = float(len(batch))
        return PolicyValueLoss(
            policy_loss=policy_loss / n,
            value_loss=value_loss / n,
            total_loss=(policy_loss + value_loss_weight * value_loss) / n,
        )

    def ppo_update(
        self,
        batch: list[TrainingExample],
        *,
        learning_rate: float,
        clip_ratio: float,
        value_weight: float,
        entropy_weight: float,
    ) -> PPOBatchStats:
        """Apply one clipped-surrogate PPO gradient step over a minibatch.

        Each example carries the log-probability and advantage recorded when the
        action was sampled, plus a return target for the value head. The loss is
        the standard PPO objective: a clipped policy-ratio surrogate, a value
        regression, and an entropy bonus, averaged over the minibatch.
        """
        if not batch:
            raise ValueError("batch must not be empty")
        self._ensure_built(batch[0].encoding, len(batch[0].legal_mask))
        assert self._net is not None
        self._net.train()
        optimizer = torch.optim.SGD(self._net.parameters(), lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)

        losses: list[torch.Tensor] = []
        policy_sum = value_sum = entropy_sum = clip_sum = kl_sum = 0.0
        for example in batch:
            logits, value = self._net(example.encoding)
            legal = torch.tensor(example.legal_mask, dtype=torch.bool)
            log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=0)
            log_prob = log_probs[example.action]
            ratio = torch.exp(log_prob - example.old_log_prob)
            advantage = float(example.advantage)
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            surrogate = -torch.minimum(ratio * advantage, clipped * advantage)
            value_loss = (value - float(example.return_target)) ** 2
            entropy = -(log_probs[legal].exp() * log_probs[legal]).sum()
            losses.append(surrogate + value_weight * value_loss - entropy_weight * entropy)

            policy_sum += float(surrogate.detach())
            value_sum += float(value_loss.detach())
            entropy_sum += float(entropy.detach())
            clip_sum += float(abs(float(ratio.detach()) - 1.0) > clip_ratio)
            kl_sum += float(example.old_log_prob - float(log_prob.detach()))

        torch.stack(losses).mean().backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        n = float(len(batch))
        return PPOBatchStats(
            policy_loss=policy_sum / n,
            value_loss=value_sum / n,
            entropy=entropy_sum / n,
            total_loss=(policy_sum + value_weight * value_sum - entropy_weight * entropy_sum) / n,
            clip_fraction=clip_sum / n,
            approx_kl=kl_sum / n,
        )

    # -- serialization -----------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        state = self._net.state_dict() if self._net is not None else {}
        parameters = {name: value.detach().numpy().tolist() for name, value in state.items()}
        return {
            "architecture": self.architecture,
            "hyperparameters": {"seed": self.seed, **self._hp},
            "built": self._net is not None,
            "feature_dim": self._feature_dim,
            "action_count": self._action_count,
            "parameters": parameters,
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Stage parameters written by :meth:`to_json` for the next lazy build.

        The trunk shape depends on the first encoding (vocabulary size), so the
        weights are applied inside :meth:`_ensure_built` once the network exists.
        """
        if not data.get("built", False):
            return
        self._feature_dim = int(data["feature_dim"])
        self._action_count = int(data["action_count"])
        self._pending_state = data["parameters"]


def _example_loss(
    logits: torch.Tensor,
    value: torch.Tensor,
    example: TrainingExample,
    value_weight: float,
) -> torch.Tensor:
    """Masked softmax cross-entropy plus weighted value mean-squared error."""
    legal = torch.tensor(example.legal_mask, dtype=torch.bool)
    target = torch.from_numpy(np.ascontiguousarray(example.policy_target, dtype=np.float32))
    log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=0)
    policy_loss = -(target[legal] * log_probs[legal]).sum()
    value_loss = (value - float(example.value_target)) ** 2
    return policy_loss + value_weight * value_loss
