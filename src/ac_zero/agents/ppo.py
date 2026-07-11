from __future__ import annotations

import random

from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment
from ac_zero.models.base import PolicyValueModel
from ac_zero.training.ppo.losses import masked_softmax, sample_from_policy


class PPOAgent:
    """Act on an AC environment with a policy-value model's masked policy.

    With no RNG the agent decodes greedily (the most probable legal action, ties
    broken by lowest action id) for reproducible rollouts and certificates; given
    an RNG it samples from the masked policy, matching how PPO explores during
    training. Illegal actions always receive exactly zero probability.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        encoder: StateEncoder | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Bind the agent to a model, optional encoder, and optional sampling RNG."""
        self.model = model
        self.encoder = encoder or StateEncoder()
        self.rng = rng

    def select_action(self, env: ACEnvironment) -> int:
        """Choose one legal action for the environment's current state."""
        mask = env.legal_action_mask()
        if not any(mask):
            raise RuntimeError("no legal actions")
        output = self.model.apply(self.encoder.encode(env.state), len(mask))
        probs = masked_softmax(output.logits, mask)
        if self.rng is not None:
            return sample_from_policy(probs, self.rng)
        return max(range(len(probs)), key=lambda i: (probs[i], -i))
