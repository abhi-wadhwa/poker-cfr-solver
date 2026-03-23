"""Tests for CFR algorithm implementations.

Verifies that Vanilla CFR, CFR+, and MCCFR all converge to approximately
correct strategies on Kuhn Poker within a reasonable number of iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cfr import VanillaCFR
from src.core.cfr_plus import CFRPlus
from src.core.mccfr import MCCFR
from src.games.kuhn_poker import KuhnPoker


@pytest.fixture
def kuhn_game():
    return KuhnPoker()


class TestVanillaCFR:
    """Tests for the Vanilla CFR solver."""

    def test_train_returns_exploitability(self, kuhn_game):
        solver = VanillaCFR(kuhn_game)
        history = solver.train(100)
        assert len(history) > 0
        assert all(isinstance(v, float) for v in history)

    def test_converges_on_kuhn(self, kuhn_game):
        """CFR should converge to near-zero exploitability on Kuhn Poker."""
        solver = VanillaCFR(kuhn_game)
        history = solver.train(10000)
        final_expl = history[-1]
        assert final_expl < 0.01, f"Exploitability {final_expl} too high after 10k iterations"

    def test_strategy_map_populated(self, kuhn_game):
        """After training, strategy map should contain all Kuhn info sets."""
        solver = VanillaCFR(kuhn_game)
        solver.train(100)
        # Kuhn Poker has 12 information sets (6 for each player)
        assert len(solver.strategy_map) == 12

    def test_average_strategy_valid_distributions(self, kuhn_game):
        """Average strategy must be valid probability distributions."""
        solver = VanillaCFR(kuhn_game)
        solver.train(1000)
        avg = solver.average_strategy()
        for key, probs in avg.items():
            assert np.all(probs >= 0), f"Negative probability at {key}: {probs}"
            assert abs(probs.sum() - 1.0) < 1e-10, f"Probs don't sum to 1 at {key}: {probs}"

    def test_iteration_counter(self, kuhn_game):
        solver = VanillaCFR(kuhn_game)
        solver.train(500)
        assert solver.iteration == 500


class TestCFRPlus:
    """Tests for the CFR+ solver."""

    def test_converges_on_kuhn(self, kuhn_game):
        """CFR+ should converge faster than vanilla CFR."""
        solver = CFRPlus(kuhn_game)
        history = solver.train(5000)
        final_expl = history[-1]
        assert final_expl < 0.01, f"Exploitability {final_expl} too high after 5k iterations"

    def test_regrets_non_negative(self, kuhn_game):
        """CFR+ clamps regrets to be non-negative."""
        solver = CFRPlus(kuhn_game)
        solver.train(100)
        for node in solver.strategy_map.values():
            assert np.all(
                node.cumulative_regret >= 0
            ), f"Negative regret found: {node.cumulative_regret}"

    def test_average_strategy_valid(self, kuhn_game):
        solver = CFRPlus(kuhn_game)
        solver.train(1000)
        avg = solver.average_strategy()
        for _key, probs in avg.items():
            assert np.all(probs >= 0)
            assert abs(probs.sum() - 1.0) < 1e-10


class TestMCCFR:
    """Tests for the Monte Carlo CFR solver."""

    def test_converges_on_kuhn(self, kuhn_game):
        """MCCFR should converge on Kuhn Poker (may need more iterations)."""
        solver = MCCFR(kuhn_game, seed=42)
        history = solver.train(20000)
        final_expl = history[-1]
        # MCCFR is noisier, allow slightly higher tolerance
        assert final_expl < 0.05, f"Exploitability {final_expl} too high after 20k iterations"

    def test_deterministic_with_seed(self, kuhn_game):
        """Same seed should produce same results."""
        solver1 = MCCFR(kuhn_game, seed=123)
        solver1.train(500)
        strat1 = solver1.average_strategy()

        solver2 = MCCFR(kuhn_game, seed=123)
        solver2.train(500)
        strat2 = solver2.average_strategy()

        for key in strat1:
            np.testing.assert_array_almost_equal(strat1[key], strat2[key])

    def test_average_strategy_valid(self, kuhn_game):
        solver = MCCFR(kuhn_game, seed=42)
        solver.train(1000)
        avg = solver.average_strategy()
        for _key, probs in avg.items():
            assert np.all(probs >= 0)
            assert abs(probs.sum() - 1.0) < 1e-10
