"""CFR+ (CFR Plus) algorithm implementation.

Implements CFR+ from:
    Tammelin, O. (2014). "Solving Large Imperfect Information Games Using CFR+."
    arXiv:1407.5042.

    Bowling, M., Burch, N., Johanson, M., & Tammelin, O. (2015).
    "Heads-up Limit Hold'em Poker is Solved." Science, 347(6218).

Key differences from vanilla CFR:
    1. Regret floor: cumulative regrets are clamped to be non-negative after each update.
    2. Linear averaging: the average strategy uses iteration-weighted contributions
       (strategy at iteration t gets weight t), which improves convergence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from src.core.regret_matching import RegretMatchedStrategy

if TYPE_CHECKING:
    from src.games.game_base import ExtensiveFormGame

logger = structlog.get_logger()


class CFRPlusStrategy(RegretMatchedStrategy):
    """Strategy node for CFR+ with regret clamping."""

    def clamp_regrets(self) -> None:
        """Clamp all cumulative regrets to be non-negative (regret floor at zero)."""
        np.maximum(self.cumulative_regret, 0.0, out=self.cumulative_regret)


class CFRPlus:
    """CFR+ solver for two-player zero-sum extensive-form games.

    Attributes:
        game: The game to solve.
        strategy_map: Mapping from information set key to CFRPlusStrategy.
        iteration: Current iteration count.
    """

    def __init__(self, game: ExtensiveFormGame) -> None:
        self.game = game
        self.strategy_map: dict[str, CFRPlusStrategy] = {}
        self.iteration: int = 0

    def _get_strategy(self, info_set_key: str, num_actions: int) -> CFRPlusStrategy:
        """Get or create the strategy for an information set."""
        if info_set_key not in self.strategy_map:
            self.strategy_map[info_set_key] = CFRPlusStrategy(num_actions)
        return self.strategy_map[info_set_key]

    def train(self, num_iterations: int) -> list[float]:
        """Run CFR+ for the specified number of iterations.

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
                self._cfr_plus(initial_state, traverser, reach_probs)

            # Clamp regrets after each iteration (CFR+ key step)
            for node in self.strategy_map.values():
                node.clamp_regrets()

            # Sample exploitability periodically
            if (self.iteration % max(1, num_iterations // 100)) == 0 or i == num_iterations - 1:
                expl = compute_exploitability(self.game, self.average_strategy())
                exploitability_history.append(expl)

        return exploitability_history

    def _cfr_plus(
        self,
        state: object,
        traverser: int,
        reach_probs: np.ndarray,
    ) -> float:
        """Recursive CFR+ traversal.

        Args:
            state: Current game state.
            traverser: The player whose regrets we are updating.
            reach_probs: Reach probabilities for each player.

        Returns:
            The counterfactual value of this state for the traverser.
        """
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, traverser)

        if self.game.is_chance(state):
            value = 0.0
            for action, prob in self.game.chance_outcomes(state):
                next_state = self.game.apply_action(state, action)
                value += prob * self._cfr_plus(next_state, traverser, reach_probs)
            return value

        current_player = self.game.current_player(state)
        info_set_key = self.game.information_set_key(state)
        actions = self.game.actions(state)
        num_actions = len(actions)

        node = self._get_strategy(info_set_key, num_actions)
        strategy = node.current_strategy()

        # CFR+ uses linear averaging: weight = iteration number
        node.cumulative_strategy += self.iteration * reach_probs[current_player] * strategy

        # Compute counterfactual value for each action
        action_values = np.zeros(num_actions, dtype=np.float64)
        node_value = 0.0

        for i, action in enumerate(actions):
            next_state = self.game.apply_action(state, action)
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[i]
            action_values[i] = self._cfr_plus(next_state, traverser, new_reach)
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
        """Return the average strategy profile.

        Note: CFR+ accumulates with linear weighting (iteration * reach * strategy),
        so the average naturally gives more weight to later iterations.
        """
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
