"""Kuhn Poker implementation.

Kuhn Poker (Kuhn, 1950) is a simplified poker game used as a benchmark for
game-theoretic algorithms:

    - 3 cards: Jack (J=0), Queen (Q=1), King (K=2)
    - 2 players, each antes 1 chip
    - Each player is dealt one card
    - Player 0 acts first: Pass (p) or Bet (b)
    - Depending on the action, player 1 can Pass or Bet
    - If both pass, higher card wins the pot (1 chip each)
    - If one bets and the other passes (folds), the bettor wins
    - If both bet, higher card wins the pot (2 chips each)

The game has a known Nash equilibrium (analytically solvable), making it
ideal for verifying CFR implementations.

Known Nash equilibrium for Player 0:
    - J: bet with probability alpha in [0, 1/3], always pass after opponent bet
    - Q: always pass, call with probability 1/3
    - K: always bet (or pass then call), always call/raise

With alpha = 0 (one of the equilibria):
    P0: J always checks, Q always checks, K always bets
    P0 facing bet: J always folds, Q calls 1/3, K always calls
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from src.games.game_base import ExtensiveFormGame

# Cards
JACK = 0
QUEEN = 1
KING = 2
CARD_NAMES = {JACK: "J", QUEEN: "Q", KING: "K"}

# Actions
PASS = 0  # Check / Fold
BET = 1   # Bet / Call

ACTION_NAMES = ["pass", "bet"]


class KuhnState:
    """State representation for Kuhn Poker.

    Attributes:
        cards: Tuple of cards dealt (cards[0] = player 0's card, cards[1] = player 1's card).
        history: Tuple of actions taken so far.
    """

    __slots__ = ("cards", "history")

    def __init__(self, cards: tuple[int, int] | None, history: tuple[int, ...] = ()) -> None:
        self.cards = cards
        self.history = history

    def __repr__(self) -> str:
        if self.cards is None:
            return "KuhnState(dealing)"
        card_str = f"{CARD_NAMES[self.cards[0]]}{CARD_NAMES[self.cards[1]]}"
        hist_str = "".join("p" if a == PASS else "b" for a in self.history)
        return f"KuhnState({card_str}, {hist_str})"


class KuhnPoker(ExtensiveFormGame):
    """Kuhn Poker: a 3-card, 2-player poker game.

    The game tree structure:
        - Root: chance node deals cards (6 permutations of 3 cards, each prob 1/6)
        - Player 0 acts: pass or bet
        - Player 1 acts: pass or bet
        - Possible additional action depending on prior choices

    Terminal sequences:
        pp  -> showdown (1 chip pot per player)
        pbp -> player 1 wins (player 0 folds to bet)
        pbb -> showdown (2 chip pot per player)
        bp  -> player 0 wins (player 1 folds to bet)
        bb  -> showdown (2 chip pot per player)
    """

    @property
    def num_players(self) -> int:
        return 2

    def initial_state(self) -> KuhnState:
        """Return the initial chance node (cards not yet dealt)."""
        return KuhnState(cards=None)

    def is_terminal(self, state: KuhnState) -> bool:
        """A state is terminal when the betting round is over."""
        h = state.history
        if len(h) < 2:
            return False
        # Terminal sequences: pp, bp, bb, pbp, pbb
        if h == (PASS, PASS):
            return True  # pp -> showdown
        if h == (BET, PASS):
            return True  # bp -> fold
        if h == (BET, BET):
            return True  # bb -> showdown
        return len(h) == 3  # pbp or pbb

    def terminal_utility(self, state: KuhnState, player: int) -> float:
        """Compute the utility for a player at a terminal state.

        Each player antes 1 chip. Bets add 1 more chip.
        """
        h = state.history
        cards = state.cards

        # Determine pot size and winner
        if h == (PASS, PASS):
            # Both check, showdown for ante (1 chip each)
            winner = 0 if cards[0] > cards[1] else 1
            payoff = 1.0
        elif h == (BET, PASS):
            # Player 0 bets, player 1 folds
            winner = 0
            payoff = 1.0
        elif h == (BET, BET):
            # Both bet, showdown for 2 chips each
            winner = 0 if cards[0] > cards[1] else 1
            payoff = 2.0
        elif h == (PASS, BET, PASS):
            # Player 0 checks, player 1 bets, player 0 folds
            winner = 1
            payoff = 1.0
        elif h == (PASS, BET, BET):
            # Player 0 checks, player 1 bets, player 0 calls → showdown
            winner = 0 if cards[0] > cards[1] else 1
            payoff = 2.0
        else:
            raise ValueError(f"Unexpected terminal history: {h}")

        if player == winner:
            return payoff
        return -payoff

    def is_chance(self, state: KuhnState) -> bool:
        """The initial state (no cards dealt) is the chance node."""
        return state.cards is None

    def chance_outcomes(self, state: KuhnState) -> list[tuple[Any, float]]:
        """Return all possible card deals with equal probability.

        There are P(3,2) = 6 ways to deal 2 cards from {J, Q, K}.
        """
        deals = list(permutations([JACK, QUEEN, KING], 2))
        prob = 1.0 / len(deals)
        return [(deal, prob) for deal in deals]

    def current_player(self, state: KuhnState) -> int:
        """Determine which player acts based on the history length.

        In Kuhn Poker:
        - History length 0: Player 0
        - History length 1: Player 1
        - History length 2: Player 0 (only if history is "pb")
        """
        return len(state.history) % 2

    def actions(self, state: KuhnState) -> list[int]:
        """Both actions are always available: Pass (0) or Bet (1)."""
        return [PASS, BET]

    def apply_action(self, state: KuhnState, action: Any) -> KuhnState:
        """Apply an action and return a new state.

        If this is a chance node (cards not yet dealt), action is a card deal
        tuple (card0, card1). Otherwise action is PASS or BET.
        """
        if state.cards is None:
            # Chance node: action is the card deal
            return KuhnState(cards=action, history=())
        return KuhnState(cards=state.cards, history=state.history + (action,))

    def information_set_key(self, state: KuhnState) -> str:
        """Create the information set key.

        A player knows their own card and the history of actions,
        but NOT the opponent's card.
        """
        player = self.current_player(state)
        card_name = CARD_NAMES[state.cards[player]]
        hist_str = "".join("p" if a == PASS else "b" for a in state.history)
        return f"{card_name}:{hist_str}"

    def action_names(self) -> list[str]:
        return ACTION_NAMES

    @staticmethod
    def card_name(card: int) -> str:
        """Return the name of a card."""
        return CARD_NAMES[card]
