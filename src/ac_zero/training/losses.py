from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def return_to_go(rewards: list[float]) -> list[float]:
    """Compute undiscounted return-to-go targets for one trajectory."""
    total = 0.0
    out = [0.0 for _ in rewards]
    for idx in range(len(rewards) - 1, -1, -1):
        total += rewards[idx]
        out[idx] = total
    return out


@dataclass(frozen=True, slots=True)
class PolicyValueLoss:
    """Scalar policy/value loss components for one or more replay examples."""

    policy_loss: float
    value_loss: float
    total_loss: float


def visit_count_policy(
    visit_counts: tuple[int, ...],
    legal_mask: tuple[bool, ...],
) -> NDArray[np.float64]:
    """Convert root visit counts into a normalized policy target over legal actions."""
    if len(visit_counts) != len(legal_mask):
        raise ValueError("visit_counts and legal_mask must have the same length")
    policy = np.zeros(len(visit_counts), dtype=np.float64)
    legal = [idx for idx, ok in enumerate(legal_mask) if ok]
    if not legal:
        return policy
    total_visits = sum(max(0, visit_counts[idx]) for idx in legal)
    if total_visits <= 0:
        uniform = 1.0 / len(legal)
        for idx in legal:
            policy[idx] = uniform
        return policy
    for idx in legal:
        policy[idx] = max(0, visit_counts[idx]) / total_visits
    return policy


def masked_softmax(
    logits: NDArray[np.float64],
    legal_mask: tuple[bool, ...],
) -> NDArray[np.float64]:
    """Compute a stable softmax with exactly zero probability on illegal actions."""
    if logits.shape != (len(legal_mask),):
        raise ValueError("logits shape must match legal_mask length")
    legal = np.asarray(legal_mask, dtype=np.bool_)
    probs = np.zeros_like(logits, dtype=np.float64)
    if not bool(legal.any()):
        return probs
    legal_logits = logits[legal]
    shifted = legal_logits - float(np.max(legal_logits))
    exp = np.exp(shifted)
    probs[legal] = exp / float(np.sum(exp))
    return probs


def policy_value_loss(
    logits: NDArray[np.float64],
    value: float,
    policy_target: NDArray[np.float64],
    value_target: float,
    legal_mask: tuple[bool, ...],
    *,
    value_weight: float = 1.0,
) -> PolicyValueLoss:
    """Compute masked cross-entropy plus weighted value mean-squared error."""
    probs = masked_softmax(logits, legal_mask)
    if probs.shape != policy_target.shape:
        raise ValueError("policy_target shape must match logits shape")
    policy_loss = 0.0
    for prob, target in zip(probs, policy_target, strict=True):
        if target > 0.0:
            policy_loss -= float(target) * math.log(max(float(prob), 1e-12))
    value_loss = (float(value) - value_target) ** 2
    return PolicyValueLoss(
        policy_loss=policy_loss,
        value_loss=value_loss,
        total_loss=policy_loss + value_weight * value_loss,
    )
