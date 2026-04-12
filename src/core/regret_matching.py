"""Regret matching strategy representation for information sets.

Implements the regret-matching algorithm (Hart & Mas-Colell, 2000) that converts
cumulative regrets into a mixed strategy. Each information set maintains its own
cumulative regret vector and cumulative strategy profile.
"""

from __future__ import annotations

import numpy as np


class RegretMatchedStrategy:
    """Strategy for a single information set using regret matching.

    Attributes:
        num_actions: Number of available actions at this information set.
        cumulative_regret: Running sum of counterfactual regrets per action.
        cumulative_strategy: Running sum of reach-probability-weighted strategies
            (used for computing the average strategy).
    """

    __slots__ = ("num_actions", "cumulative_regret", "cumulative_strategy")

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.cumulative_regret = np.zeros(num_actions, dtype=np.float64)
        self.cumulative_strategy = np.zeros(num_actions, dtype=np.float64)

    def current_strategy(self) -> np.ndarray:
        """Compute the current strategy via regret matching.

        strategy[a] = max(R[a], 0) / sum_b max(R[b], 0)

        If all regrets are non-positive, returns the uniform strategy.

        Returns:
            Probability distribution over actions (sums to 1).
        """
        positive_regret = np.maximum(self.cumulative_regret, 0.0)
        total = positive_regret.sum()
        if total > 0:
            return positive_regret / total
        return np.full(self.num_actions, 1.0 / self.num_actions)

    def average_strategy(self) -> np.ndarray:
        """Compute the average strategy over all iterations.

        The average strategy converges to a Nash equilibrium component.

        Returns:
            Probability distribution over actions (sums to 1).
        """
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return np.full(self.num_actions, 1.0 / self.num_actions)

    def update_cumulative_strategy(self, reach_probability: float) -> None:
        """Accumulate the current strategy weighted by reach probability.

        Args:
            reach_probability: The probability of reaching this information set
                under the current player's strategy.
        """
        strategy = self.current_strategy()
        self.cumulative_strategy += reach_probability * strategy

    def __repr__(self) -> str:
        avg = self.average_strategy()
        return (
            f"RegretMatchedStrategy(n={self.num_actions}, "
            f"avg={np.array2string(avg, precision=3)})"
        )
