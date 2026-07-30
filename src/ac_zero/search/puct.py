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


class _MinMaxStats:
    """Running bounds on the action values seen this search, for normalization.

    The PUCT score adds a mean action value to ``c_puct * prior * sqrt(N)/(1+n)``,
    which only balances when the two live on comparable scales. They do not here:
    the navigation reward is trained raw (``reward_scale`` is 1.0), so a mean action
    value spans tens -- a destination bonus of ``L0`` against a shaping sum that
    reaches -30 -- while the prior term is bounded by ``c_puct``. Left unnormalized
    the prior is numerically negligible and selection collapses to greedy on
    one-sample value estimates, which is how a search can pick a descending move
    less often than the policy prior it was handed.

    Normalizing the mean into ``[0, 1]`` against the values actually seen restores
    the balance the constant assumes, and makes ``c_puct`` mean the same thing
    across reward modes and reward scales. This is the MuZero treatment.
    """

    __slots__ = ("maximum", "minimum")

    def __init__(self) -> None:
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        """Map ``value`` into ``[0, 1]``; identity until two distinct values are seen."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


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
        bounds = _MinMaxStats()
        self._expand(env, root, nodes)
        for _ in range(self.config.simulations):
            self._simulate(env, root, root_reward, nodes, reward_scale, bounds)
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
        bounds: _MinMaxStats,
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
            action = self._select_action(node, bounds)
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
        self._backup(nodes, path, leaf_value, bounds)

    def _select_action(self, node: _Node, bounds: _MinMaxStats) -> int | None:
        """Pick the child maximizing normalized mean value plus the prior bonus.

        The mean is normalized against the values seen so far (see
        :class:`_MinMaxStats`) so it is commensurate with the prior term.

        An unvisited child takes its parent's value ("first play urgency = parent
        value"), not a constant. A constant cannot be right in both reward modes:
        against navigation's raw negative rewards a bare ``0.0`` is wildly
        optimistic, while against the small positive rewards of the length-reduction
        modes -- once the values are normalized to ``[0, 1]`` -- the same ``0.0``
        reads as *worst seen* and the search stops exploring. Measured on a greedy
        rollout, that constant alone moved the solve rate from 85% to 56%. The
        parent's value carries the scale with it and needs no such choice.

        The visit total is offset by one inside the square root so the prior term is
        non-zero on the very first simulation. With a bare ``sqrt(total)`` every
        score is 0 before any visit and selection falls to index order, so the first
        simulation of every search discarded the policy entirely.
        """
        total = sum(node.visits)
        sqrt_total = math.sqrt(1 + total)
        parent_value = bounds.normalize(node.value)
        best_score = -math.inf
        best_action: int | None = None
        for action, legal in enumerate(node.legal):
            if not legal:
                continue
            visits = node.visits[action]
            value = bounds.normalize(node.values[action] / visits) if visits else parent_value
            exploration = self.config.c_puct * node.priors[action] * sqrt_total / (1 + visits)
            score = value + exploration
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
        bounds: _MinMaxStats,
    ) -> None:
        suffix = leaf_value
        for key, action, reward in reversed(path):
            suffix += reward
            node = nodes[key]
            node.visits[action] += 1
            node.values[action] += suffix
            # The bounds track the *mean* action values selection compares, so they
            # are updated with the same quantity `_select_action` normalizes.
            bounds.update(node.values[action] / node.visits[action])
