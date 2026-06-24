from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ac_zero.encoding.padded import PaddedEncoding
from ac_zero.models.autograd import Node
from ac_zero.models.base import PolicyValueOutput
from ac_zero.training.losses import PolicyValueLoss, masked_softmax, policy_value_loss

# A replay example is duck-typed; only these attributes are read during training.
TrainingExample = Any


class TrainablePolicyValueModel(ABC):
    """Shared training machinery for the CPU policy/value architectures.

    Subclasses contribute an architecture-specific *trunk* that maps an encoded
    state to a feature vector; this base attaches linear policy and (tanh-bounded)
    value heads and trains every parameter by exact reverse-mode gradient descent.
    Parameters are built lazily on first use so the action-head width and any
    encoding-dependent dimensions are taken from real inputs.
    """

    architecture: str = "trainable"

    def __init__(self, *, seed: int = 0, **hyperparameters: int) -> None:
        self.seed = seed
        self._hp: dict[str, int] = dict(hyperparameters)
        self._params: dict[str, Node] = {}
        self._feature_dim = 0
        self._action_count = 0
        self._built = False

    # -- subclass contract -------------------------------------------------
    @abstractmethod
    def _build_trunk(self, rng: np.random.Generator, encoding: PaddedEncoding) -> int:
        """Create trunk parameters via :meth:`_param` and return the feature dim."""

    @abstractmethod
    def _forward_trunk(self, encoding: PaddedEncoding) -> Node:
        """Compute a ``(1, feature_dim)`` feature node from the current parameters."""

    # -- parameter helpers -------------------------------------------------
    def _param(self, name: str, array: NDArray[np.float64]) -> Node:
        node = Node(array.astype(np.float64), requires_grad=True)
        self._params[name] = node
        return node

    def _ensure_built(self, encoding: PaddedEncoding, action_count: int) -> None:
        if self._built:
            if action_count != self._action_count:
                raise ValueError("action_count changed after the model was built")
            return
        rng = np.random.default_rng(self.seed)
        self._feature_dim = self._build_trunk(rng, encoding)
        feature_dim = self._feature_dim
        self._param("policy_w", rng.normal(0.0, 0.01, (feature_dim, action_count)))
        self._param("policy_b", np.zeros((1, action_count)))
        self._param("value_w", rng.normal(0.0, 0.01, (feature_dim, 1)))
        self._param("value_b", np.zeros((1, 1)))
        self._action_count = action_count
        self._built = True

    def _heads(self, trunk: Node) -> tuple[Node, Node]:
        logits = trunk @ self._params["policy_w"] + self._params["policy_b"]
        value = (trunk @ self._params["value_w"] + self._params["value_b"]).tanh()
        return logits, value

    # -- inference and training -------------------------------------------
    def apply(self, encoding: PaddedEncoding, action_count: int) -> PolicyValueOutput:
        """Predict policy logits and a bounded value for one encoded state."""
        self._ensure_built(encoding, action_count)
        trunk = self._forward_trunk(encoding)
        logits, value = self._heads(trunk)
        return PolicyValueOutput(logits.data[0].astype(np.float64), float(value.data[0, 0]))

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
        for node in self._params.values():
            node.grad = np.zeros_like(node.data)

        policy_loss = 0.0
        value_loss = 0.0
        for example in batch:
            trunk = self._forward_trunk(example.encoding)
            logits, value = self._heads(trunk)
            reported = policy_value_loss(
                logits.data[0],
                float(value.data[0, 0]),
                example.policy_target,
                example.value_target,
                example.legal_mask,
                value_weight=value_loss_weight,
            )
            policy_loss += reported.policy_loss
            value_loss += reported.value_loss
            loss_node = _masked_cross_entropy(logits, example.policy_target, example.legal_mask)
            diff = value - Node(float(example.value_target))
            loss_node = loss_node + (diff * diff) * value_loss_weight
            loss_node.backward()

        scale = learning_rate / len(batch)
        for node in self._params.values():
            node.data -= scale * node.grad
        n = float(len(batch))
        return PolicyValueLoss(
            policy_loss=policy_loss / n,
            value_loss=value_loss / n,
            total_loss=(policy_loss + value_loss_weight * value_loss) / n,
        )

    # -- serialization -----------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hyperparameters": {"seed": self.seed, **self._hp},
            "built": self._built,
            "feature_dim": self._feature_dim,
            "action_count": self._action_count,
            "parameters": {name: node.data.tolist() for name, node in self._params.items()},
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore parameters previously written by :meth:`to_json`."""
        if not data.get("built", False):
            return
        self._feature_dim = int(data["feature_dim"])
        self._action_count = int(data["action_count"])
        self._params = {
            name: Node(np.asarray(value, dtype=np.float64), requires_grad=True)
            for name, value in data["parameters"].items()
        }
        self._built = True


def _masked_cross_entropy(
    logits: Node,
    target: NDArray[np.float64],
    legal_mask: tuple[bool, ...],
) -> Node:
    """Masked softmax cross-entropy with the standard ``probs - target`` backward."""
    probs = masked_softmax(logits.data[0], legal_mask)
    loss_value = 0.0
    for prob, weight in zip(probs, target, strict=True):
        if weight > 0.0:
            loss_value -= float(weight) * float(np.log(max(prob, 1e-12)))
    out = Node(np.array([[loss_value]], dtype=np.float64), (logits,))

    def _backward() -> None:
        logits.grad += out.grad[0, 0] * (probs - target)[np.newaxis, :]

    out._backward = _backward
    return out
