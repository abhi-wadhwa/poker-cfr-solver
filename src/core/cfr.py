"""Vanilla Counterfactual Regret Minimization (CFR).

Implements the full-width CFR algorithm from:
    Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007).
    "Regret Minimization in Games with Incomplete Information."
    Advances in Neural Information Processing Systems (NeurIPS).

The algorithm traverses the entire game tree on each iteration, computing
counterfactual values for every information set and updating cumulative regrets.
The average strategy profile converges to a Nash equilibrium at rate O(1/sqrt(T)).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from src.core.regret_matching import RegretMatchedStrategy

if TYPE_CHECKING:
    from src.games.game_base import ExtensiveFormGame

logger = structlog.get_logger()


class VanillaCFR:
    """Vanilla CFR solver for two-player zero-sum extensive-form games.

    Attributes:
        game: The game to solve.
        strategy_map: Mapping from information set key to RegretMatchedStrategy.
        iteration: Current iteration count.
    """

    def __init__(self, game: ExtensiveFormGame) -> None:
        self.game = game
        self.strategy_map: dict[str, RegretMatchedStrategy] = {}
        self.iteration: int = 0

    def _get_strategy(self, info_set_key: str, num_actions: int) -> RegretMatchedStrategy:
        """Get or create the strategy for an information set."""
        if info_set_key not in self.strategy_map:
            self.strategy_map[info_set_key] = RegretMatchedStrategy(num_actions)
        return self.strategy_map[info_set_key]

    def train(self, num_iterations: int) -> list[float]:
        """Run CFR for the specified number of iterations.

        Args:
            num_iterations: Number of full game-tree traversals.

        Returns:
            List of exploitability values sampled during training.
        """
        from src.core.exploitability import compute_exploitability

        exploitability_history: list[float] = []

        for i in range(num_iterations):
            self.iteration += 1
            for traverser in range(self.game.num_players):
                initial_state = self.game.initial_state()
                reach_probs = np.ones(self.game.num_players, dtype=np.float64)
                self._cfr(initial_state, traverser, reach_probs)

            # Sample exploitability periodically
            if (self.iteration % max(1, num_iterations // 100)) == 0 or i == num_iterations - 1:
                expl = compute_exploitability(self.game, self.average_strategy())
                exploitability_history.append(expl)

        return exploitability_history

    def _cfr(
        self,
        state: object,
        traverser: int,
        reach_probs: np.ndarray,
    ) -> float:
        """Recursive CFR traversal.

        Args:
            state: Current game state.
            traverser: The player whose regrets we are updating (0 or 1).
            reach_probs: Reach probabilities for each player to this state.

        Returns:
            The counterfactual value of this state for the traverser.
        """
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traverser)

        if self.game.is_chance(state):
            value = 0.0
            for action, prob in self.game.chance_outcomes(state):
                next_state = self.game.apply_action(state, action)
                value += prob * self._cfr(next_state, traverser, reach_probs)
            return value

        current_player = self.game.current_player(state)
        info_set_key = self.game.information_set_key(state)
        actions = self.game.actions(state)
        num_actions = len(actions)

        node = self._get_strategy(info_set_key, num_actions)
        strategy = node.current_strategy()

        # Accumulate reach-weighted strategy for average computation
        node.update_cumulative_strategy(reach_probs[current_player])

        # Compute counterfactual value for each action
        action_values = np.zeros(num_actions, dtype=np.float64)
        node_value = 0.0

        for i, action in enumerate(actions):
            next_state = self.game.apply_action(state, action)
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[i]
            action_values[i] = self._cfr(next_state, traverser, new_reach)
            node_value += strategy[i] * action_values[i]

        # Update regrets for the traverser
        if current_player == traverser:
            opponent = 1 - traverser
            counterfactual_reach = reach_probs[opponent]
            for i in range(num_actions):
                regret = action_values[i] - node_value
                node.cumulative_regret[i] += counterfactual_reach * regret

        return node_value

    def average_strategy(self) -> dict[str, np.ndarray]:
        """Return the average strategy profile across all information sets.

        Returns:
            Mapping from information set key to probability distribution.
        """
        result: dict[str, np.ndarray] = {}
        for key, node in self.strategy_map.items():
            result[key] = node.average_strategy()
        return result

    def current_strategy(self) -> dict[str, np.ndarray]:
        """Return the current (regret-matched) strategy profile.

        Returns:
            Mapping from information set key to probability distribution.
        """
        result: dict[str, np.ndarray] = {}
        for key, node in self.strategy_map.items():
            result[key] = node.current_strategy()
        return result
