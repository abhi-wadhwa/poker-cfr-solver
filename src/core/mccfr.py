"""Monte Carlo Counterfactual Regret Minimization (MCCFR).

Implements External Sampling MCCFR from:
    Lanctot, M., Waugh, K., Zinkevich, M., & Bowling, M. (2009).
    "Monte Carlo Sampling for Regret Minimization in Extensive Games."
    Advances in Neural Information Processing Systems (NeurIPS).

    Brown, N., & Sandholm, T. (2019).
    "Superhuman AI for multiplayer poker." Science, 365(6456).

External sampling: for the non-traverser, we sample a single action
according to the current strategy instead of exploring all actions.
This dramatically reduces per-iteration cost while preserving convergence
guarantees (in expectation).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import structlog

from src.core.regret_matching import RegretMatchedStrategy

if TYPE_CHECKING:
    from src.games.game_base import ExtensiveFormGame

logger = structlog.get_logger()


class MCCFR:
    """Monte Carlo CFR with external sampling.

    Attributes:
        game: The game to solve.
        strategy_map: Mapping from information set key to RegretMatchedStrategy.
        iteration: Current iteration count.
        rng: Random number generator for sampling.
    """

    def __init__(self, game: ExtensiveFormGame, seed: int | None = None) -> None:
        self.game = game
        self.strategy_map: dict[str, RegretMatchedStrategy] = {}
        self.iteration: int = 0
        self.rng = random.Random(seed)

    def _get_strategy(self, info_set_key: str, num_actions: int) -> RegretMatchedStrategy:
        """Get or create the strategy for an information set."""
        if info_set_key not in self.strategy_map:
            self.strategy_map[info_set_key] = RegretMatchedStrategy(num_actions)
        return self.strategy_map[info_set_key]

    def train(self, num_iterations: int) -> list[float]:
        """Run External Sampling MCCFR.

        Each iteration does one traversal for each player as the traverser.

        Args:
            num_iterations: Number of iterations (each iteration = 2 traversals).

        Returns:
            List of exploitability values sampled during training.
        """
        from src.core.exploitability import compute_exploitability

        exploitability_history: list[float] = []

        for i in range(num_iterations):
            self.iteration += 1
            for traverser in range(self.game.num_players):
                initial_state = self.game.initial_state()
                self._external_sampling(initial_state, traverser)

            # Sample exploitability periodically
            if (self.iteration % max(1, num_iterations // 100)) == 0 or i == num_iterations - 1:
                expl = compute_exploitability(self.game, self.average_strategy())
                exploitability_history.append(expl)

        return exploitability_history

    def _external_sampling(
        self,
        state: object,
        traverser: int,
    ) -> float:
        """External sampling MCCFR traversal.

        For the traverser: explore all actions (like vanilla CFR).
        For the opponent: sample one action from the current strategy.
        For chance nodes: sample one outcome.

        Args:
            state: Current game state.
            traverser: The player whose regrets we are updating.

        Returns:
            The sampled counterfactual value for the traverser.
        """
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traverser)

        if self.game.is_chance(state):
            # Sample one chance outcome
            outcomes = self.game.chance_outcomes(state)
            actions, probs = zip(*outcomes, strict=True)
            probs_array = np.array(probs, dtype=np.float64)
            probs_array /= probs_array.sum()  # normalize
            idx = self._sample_action(probs_array)
            next_state = self.game.apply_action(state, actions[idx])
            return self._external_sampling(next_state, traverser)

        current_player = self.game.current_player(state)
        info_set_key = self.game.information_set_key(state)
        actions = self.game.actions(state)
        num_actions = len(actions)

        node = self._get_strategy(info_set_key, num_actions)
        strategy = node.current_strategy()

        if current_player != traverser:
            # External sampling: sample one action for the opponent
            node.cumulative_strategy += strategy  # unweighted for MCCFR
            sampled_idx = self._sample_action(strategy)
            next_state = self.game.apply_action(state, actions[sampled_idx])
            return self._external_sampling(next_state, traverser)

        # Traverser: explore all actions
        action_values = np.zeros(num_actions, dtype=np.float64)
        for i, action in enumerate(actions):
            next_state = self.game.apply_action(state, action)
            action_values[i] = self._external_sampling(next_state, traverser)

        # Compute node value
        node_value = np.dot(strategy, action_values)

        # Update regrets (no reach probability weighting in external sampling)
        for i in range(num_actions):
            node.cumulative_regret[i] += action_values[i] - node_value

        # Update cumulative strategy
        node.cumulative_strategy += strategy

        return node_value

    def _sample_action(self, probs: np.ndarray) -> int:
        """Sample an action index from a probability distribution.

        Args:
            probs: Probability distribution over actions.

        Returns:
            Sampled action index.
        """
        r = self.rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    def average_strategy(self) -> dict[str, np.ndarray]:
        """Return the average strategy profile."""
        result: dict[str, np.ndarray] = {}
        for key, node in self.strategy_map.items():
            result[key] = node.average_strategy()
        return result

    def current_strategy(self) -> dict[str, np.ndarray]:
        """Return the current (regret-matched) strategy profile."""
        result: dict[str, np.ndarray] = {}
        for key, node in self.strategy_map.items():
            result[key] = node.current_strategy()
        return result
