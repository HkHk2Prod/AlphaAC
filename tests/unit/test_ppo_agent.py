from __future__ import annotations

import random

import numpy as np
import pytest

from ac_zero.agents.ppo import PPOAgent
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.base import PolicyValueOutput
from ac_zero.models.registry import create_trainable_model


class _FixedModel:
    """A stand-in policy-value model that returns preset logits for every state."""

    def __init__(self, logits: np.ndarray, value: float = 0.0) -> None:
        self._logits = np.asarray(logits, dtype=np.float64)
        self._value = value

    def apply(self, encoding: object, action_count: int) -> PolicyValueOutput:
        del encoding, action_count
        return PolicyValueOutput(self._logits.copy(), self._value)


class _NoLegalEnv:
    """Minimal env whose current state has no legal actions."""

    state = None

    def legal_action_mask(self) -> tuple[bool, ...]:
        return (False, False, False)


def _env() -> ACEnvironment:
    pres = BalancedPresentation.from_letters(2, [[1, 2], [2]])
    return ACEnvironment(pres, ACEnvironmentConfig(max_moves=6))


def test_greedy_agent_ignores_illegal_actions_and_takes_the_best_legal_one() -> None:
    env = _env()
    mask = env.legal_action_mask()
    legal = [i for i, ok in enumerate(mask) if ok]
    illegal = [i for i, ok in enumerate(mask) if not ok]
    assert legal and illegal  # the fixture masks its no-op moves

    logits = np.zeros(len(mask))
    logits[illegal[0]] = 100.0  # a masked action must never be chosen
    winner = legal[len(legal) // 2]
    logits[winner] = 10.0

    agent = PPOAgent(_FixedModel(logits), StateEncoder())
    assert agent.select_action(env) == winner


def test_sampling_agent_only_ever_returns_legal_actions() -> None:
    env = _env()
    mask = env.legal_action_mask()
    model = create_trainable_model("residual_mlp", seed=0)
    agent = PPOAgent(model, StateEncoder(), random.Random(0))
    drawn = {agent.select_action(env) for _ in range(50)}
    assert drawn  # something was sampled
    assert all(mask[action] for action in drawn)


def test_agent_raises_when_no_action_is_legal() -> None:
    agent = PPOAgent(_FixedModel(np.zeros(3)))
    with pytest.raises(RuntimeError, match="no legal actions"):
        agent.select_action(_NoLegalEnv())  # type: ignore[arg-type]
