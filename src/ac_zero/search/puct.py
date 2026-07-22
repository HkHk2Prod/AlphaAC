from __future__ import annotations

import math
from dataclasses import dataclass

from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, NavigationRewardState
from ac_zero.environment.state import ACSearchState
from ac_zero.models.base import PolicyValueModel
from ac_zero.search.mcts import MCTSStats
from ac_zero.training.ppo.losses import masked_softmax


@dataclass(frozen=True, slots=True)
class PUCTConfig:
    """Hyperparameters for single-player PUCT search."""

    simulations: int = 64
    c_puct: float = 1.5


class _Node:
    """Mutable per-state search statistics."""

    __slots__ = ("legal", "priors", "terminal", "value", "values", "visits")

    def __init__(
        self, priors: list[float], legal: list[bool], terminal: bool, value: float
    ) -> None:
        size = len(priors)
        self.priors = priors
        self.visits = [0] * size
        self.values = [0.0] * size
        self.legal = legal
        self.terminal = terminal
        self.value = value


class PUCTMCTS:
    """Single-player PUCT search guided by a policy/value model.

    Selection follows the AlphaZero PUCT rule using model priors and mean action
    values; leaves are expanded with the model and their value, combined with the
    normalized length-reduction rewards collected along the path, is backed up.
    The search reuses the environment by saving and restoring its root state, so
    it produces visit-count policy targets without disturbing the caller.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        encoder: StateEncoder | None = None,
        config: PUCTConfig | None = None,
    ) -> None:
        """Bind the search to a model and optional encoder/config."""
        self.model = model
        self.encoder = encoder or StateEncoder()
        self.config = config or PUCTConfig()
        self.model_evaluations = 0

    def search(self, env: ACEnvironment) -> MCTSStats:
        """Run PUCT simulations from the environment's current state."""
        root = env.state
        # The navigation reward keeps per-episode state (visited set, distance
        # anchor) outside `env.state`, so restoring the state alone would leave the
        # caller's episode -- and every simulation after the first -- scored against
        # moves this search only imagined.
        root_reward = env.navigation_reward_state()
        action_count = len(env.catalog)
        nodes: dict[tuple[object, ...], _Node] = {}
        self.model_evaluations = 0
        reward_scale = env.reward_scale
        self._expand(env, root, nodes)
        for _ in range(self.config.simulations):
            self._simulate(env, root, root_reward, nodes, reward_scale)
        env.state = root
        env.restore_navigation_reward_state(root_reward)
        root_node = nodes[root.key]
        counts = tuple(root_node.visits)
        if not any(counts):
            return MCTSStats((0,) * action_count, len(nodes), self.model_evaluations)
        return MCTSStats(counts, len(nodes), self.model_evaluations)

    def select_action(self, env: ACEnvironment) -> int:
        """Return the most-visited root action with deterministic tie-breaking."""
        stats = self.search(env)
        if not any(stats.visit_counts):
            raise RuntimeError("no legal actions")
        return max(range(len(stats.visit_counts)), key=lambda i: (stats.visit_counts[i], -i))

    def _simulate(
        self,
        env: ACEnvironment,
        root: ACSearchState,
        root_reward: NavigationRewardState | None,
        nodes: dict[tuple[object, ...], _Node],
        reward_scale: float,
    ) -> None:
        # Each simulation is scored as a continuation of the *real* episode, so it
        # rewinds the navigation reward to the root alongside the Markov state.
        env.state = root
        env.restore_navigation_reward_state(root_reward)
        path: list[tuple[tuple[object, ...], int, float]] = []
        state = root
        while True:
            node = nodes[state.key]
            if node.terminal:
                break
            action = self._select_action(node)
            if action is None:
                break
            prev_key = state.key
            _, reward, terminated, truncated, _ = env.step(action)
            state = env.state
            path.append((prev_key, action, reward * reward_scale))
            if state.key not in nodes:
                self._expand(env, state, nodes, terminated or truncated, terminated)
                break
            if terminated or truncated:
                break
        leaf_value = nodes[state.key].value
        self._backup(nodes, path, leaf_value)

    def _select_action(self, node: _Node) -> int | None:
        total = sum(node.visits)
        sqrt_total = math.sqrt(total) if total > 0 else 0.0
        best_score = -math.inf
        best_action: int | None = None
        for action, legal in enumerate(node.legal):
            if not legal:
                continue
            visits = node.visits[action]
            mean = node.values[action] / visits if visits else 0.0
            exploration = self.config.c_puct * node.priors[action] * sqrt_total / (1 + visits)
            score = mean + exploration
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _expand(
        self,
        env: ACEnvironment,
        state: ACSearchState,
        nodes: dict[tuple[object, ...], _Node],
        terminal: bool = False,
        reached_goal: bool = False,
    ) -> None:
        mask = env.legal_action_mask(state)
        encoding = self.encoder.encode(state)
        output = self.model.apply(encoding, len(mask))
        self.model_evaluations += 1
        priors = masked_softmax(output.logits, mask).tolist()
        # A goal leaf has no future reward: its destination bonus is paid on the
        # transition into it, which is already in the backed-up path rewards, so its
        # leaf value is zero. Every other leaf takes the model's value, reconstructed
        # from its heads at this episode's alpha and start distance (see
        # `ACEnvironment.leaf_value`).
        value = 0.0 if reached_goal else env.leaf_value(output)
        nodes[state.key] = _Node(priors, list(mask), terminal or not any(mask), value)

    def _backup(
        self,
        nodes: dict[tuple[object, ...], _Node],
        path: list[tuple[tuple[object, ...], int, float]],
        leaf_value: float,
    ) -> None:
        suffix = leaf_value
        for key, action, reward in reversed(path):
            suffix += reward
            node = nodes[key]
            node.visits[action] += 1
            node.values[action] += suffix
