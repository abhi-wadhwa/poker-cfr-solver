"""Tests for Kuhn Poker game implementation.

Verifies the game tree structure, terminal payoffs, information sets,
and that CFR converges to the known Nash equilibrium.
"""

from __future__ import annotations

import pytest

from src.core.cfr import VanillaCFR
from src.games.kuhn_poker import (
    BET,
    JACK,
    KING,
    PASS,
    QUEEN,
    KuhnPoker,
    KuhnState,
)


@pytest.fixture
def game():
    return KuhnPoker()


class TestKuhnGameTree:
    """Test the game tree structure of Kuhn Poker."""

    def test_initial_state_is_chance(self, game):
        state = game.initial_state()
        assert game.is_chance(state)
        assert not game.is_terminal(state)

    def test_chance_outcomes(self, game):
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        assert len(outcomes) == 6  # P(3,2) = 6
        total_prob = sum(p for _, p in outcomes)
        assert abs(total_prob - 1.0) < 1e-10

    def test_all_deals_present(self, game):
        state = game.initial_state()
        outcomes = game.chance_outcomes(state)
        deals = {deal for deal, _ in outcomes}
        expected = {
            (JACK, QUEEN), (JACK, KING),
            (QUEEN, JACK), (QUEEN, KING),
            (KING, JACK), (KING, QUEEN),
        }
        assert deals == expected

    def test_terminal_pp(self, game):
        """Pass-Pass: showdown, higher card wins ante."""
        # K vs J: King wins
        state = KuhnState(cards=(KING, JACK), history=(PASS, PASS))
        assert game.is_terminal(state)
        assert game.terminal_utility(state, 0) == 1.0
        assert game.terminal_utility(state, 1) == -1.0

    def test_terminal_bp(self, game):
        """Bet-Pass: bettor wins (fold)."""
        state = KuhnState(cards=(JACK, KING), history=(BET, PASS))
        assert game.is_terminal(state)
        # Player 0 wins despite having lower card (opponent folded)
        assert game.terminal_utility(state, 0) == 1.0
        assert game.terminal_utility(state, 1) == -1.0

    def test_terminal_bb(self, game):
        """Bet-Bet: showdown, higher card wins 2 chips."""
        state = KuhnState(cards=(QUEEN, KING), history=(BET, BET))
        assert game.is_terminal(state)
        assert game.terminal_utility(state, 0) == -2.0  # Queen loses to King
        assert game.terminal_utility(state, 1) == 2.0

    def test_terminal_pbp(self, game):
        """Pass-Bet-Pass: player 0 folds."""
        state = KuhnState(cards=(JACK, QUEEN), history=(PASS, BET, PASS))
        assert game.is_terminal(state)
        assert game.terminal_utility(state, 0) == -1.0
        assert game.terminal_utility(state, 1) == 1.0

    def test_terminal_pbb(self, game):
        """Pass-Bet-Bet: showdown for 2 chips."""
        state = KuhnState(cards=(KING, JACK), history=(PASS, BET, BET))
        assert game.is_terminal(state)
        assert game.terminal_utility(state, 0) == 2.0  # King beats Jack
        assert game.terminal_utility(state, 1) == -2.0

    def test_non_terminal_states(self, game):
        """Intermediate states should not be terminal."""
        state = KuhnState(cards=(JACK, QUEEN), history=())
        assert not game.is_terminal(state)

        state = KuhnState(cards=(JACK, QUEEN), history=(PASS,))
        assert not game.is_terminal(state)

        state = KuhnState(cards=(JACK, QUEEN), history=(PASS, BET))
        assert not game.is_terminal(state)


class TestKuhnInformationSets:
    """Test information set computation."""

    def test_info_set_includes_card(self, game):
        state = KuhnState(cards=(JACK, QUEEN), history=())
        key = game.information_set_key(state)
        assert key.startswith("J:")

    def test_info_set_includes_history(self, game):
        state = KuhnState(cards=(QUEEN, KING), history=(PASS,))
        key = game.information_set_key(state)
        # Player 1 sees their card (King) and the history
        assert key == "K:p"

    def test_different_cards_different_info_sets(self, game):
        s1 = KuhnState(cards=(JACK, QUEEN), history=())
        s2 = KuhnState(cards=(KING, QUEEN), history=())
        assert game.information_set_key(s1) != game.information_set_key(s2)

    def test_same_observable_same_info_set(self, game):
        """Player should have same info set regardless of opponent's hidden card."""
        s1 = KuhnState(cards=(JACK, QUEEN), history=())
        s2 = KuhnState(cards=(JACK, KING), history=())
        assert game.information_set_key(s1) == game.information_set_key(s2)


class TestKuhnNashEquilibrium:
    """Verify that CFR converges to the known Kuhn Poker Nash equilibrium.

    Known Nash equilibrium properties:
    - Player 0 with Jack: should rarely bet (bluff with prob alpha in [0, 1/3])
    - Player 0 with King: should always bet (or if checked, always call)
    - Player 0 with Queen: should mostly check
    - Player 1 with King facing bet: should always call
    - Player 1 with Jack facing bet: should always fold
    """

    def test_nash_equilibrium_convergence(self):
        game = KuhnPoker()
        solver = VanillaCFR(game)
        solver.train(10000)
        strategy = solver.average_strategy()

        # Player 0 with King at root: should bet frequently
        k_strat = strategy["K:"]
        assert k_strat[1] > 0.5, f"King should bet frequently, got bet={k_strat[1]:.4f}"

        # Player 1 with King facing bet: should always call
        k_b_strat = strategy["K:b"]
        assert k_b_strat[1] > 0.95, f"King facing bet should call, got call={k_b_strat[1]:.4f}"

        # Player 1 with Jack facing bet: should always fold
        j_b_strat = strategy["J:b"]
        assert j_b_strat[0] > 0.95, f"Jack facing bet should fold, got fold={j_b_strat[0]:.4f}"

        # Player 0 with Jack: bluff probability should be at most ~1/3
        j_strat = strategy["J:"]
        assert j_strat[1] < 0.4, f"Jack should rarely bluff, got bet={j_strat[1]:.4f}"

    def test_game_value_approximately_correct(self):
        """The game value of Kuhn Poker is -1/18 for player 0."""
        from src.core.exploitability import compute_exploitability

        game = KuhnPoker()
        solver = VanillaCFR(game)
        solver.train(10000)
        strategy = solver.average_strategy()

        # At Nash equilibrium, player 0's expected value should be close to -1/18
        # The exploitability should be near zero
        expl = compute_exploitability(game, strategy)
        assert expl < 0.01
