"""Leduc Hold'em implementation.

Leduc Hold'em (Southey et al., 2005) is a simplified poker game with two rounds:

    - Deck: 6 cards — two suits of three ranks: J, Q, K (JJ, QQ, KK)
    - 2 players, each antes 1 chip
    - Round 1: Each player is dealt one private card
      - Up to 2 raises allowed per round, bet size = 2
    - A community card is dealt face-up
    - Round 2: Same betting structure, bet size = 4
    - Showdown: Pair with community card beats high card; else higher card wins

This game is larger than Kuhn Poker (~936 information sets) and provides
a more meaningful benchmark for CFR convergence.
"""

from __future__ import annotations

from typing import Any

from src.games.game_base import ExtensiveFormGame

# Ranks
JACK = 0
QUEEN = 1
KING = 2

RANK_NAMES = {JACK: "J", QUEEN: "Q", KING: "K"}

# Full deck: two suits of each rank
DECK = [JACK, JACK, QUEEN, QUEEN, KING, KING]

# Actions
FOLD = 0
CHECK_CALL = 1
RAISE = 2

ACTION_NAMES_LEDUC = ["fold", "check/call", "raise"]

# Bet sizes per round
ROUND_1_BET = 2
ROUND_2_BET = 4

# Max raises per round
MAX_RAISES_PER_ROUND = 2


class LeducState:
    """State representation for Leduc Hold'em.

    Attributes:
        cards: Tuple of (player0_card, player1_card, community_card or None).
        round_num: Current round (0 = preflop, 1 = postflop).
        history: List of action tuples per round, e.g. [(a1, a2, ...), (a3, ...)].
        pot: List of chips committed by each player [p0, p1].
        folded: Player who folded (-1 if none).
    """

    __slots__ = ("cards", "round_num", "history", "pot", "folded", "_deal_community")

    def __init__(
        self,
        cards: tuple[int, int, int | None] | None = None,
        round_num: int = 0,
        history: list[tuple[int, ...]] | None = None,
        pot: list[int] | None = None,
        folded: int = -1,
        deal_community: bool = False,
    ) -> None:
        self.cards = cards
        self.round_num = round_num
        self.history = history if history is not None else [()]
        self.pot = pot if pot is not None else [1, 1]  # antes
        self.folded = folded
        self._deal_community = deal_community

    def copy(self) -> LeducState:
        return LeducState(
            cards=self.cards,
            round_num=self.round_num,
            history=[tuple(h) for h in self.history],
            pot=list(self.pot),
            folded=self.folded,
            deal_community=self._deal_community,
        )


class LeducHoldem(ExtensiveFormGame):
    """Leduc Hold'em: a 6-card, 2-player, 2-round poker game."""

    @property
    def num_players(self) -> int:
        return 2

    def initial_state(self) -> LeducState:
        """Return initial chance node (cards not dealt)."""
        return LeducState(cards=None)

    def is_terminal(self, state: LeducState) -> bool:
        if state.cards is None:
            return False
        if state._deal_community:
            return False
        if state.folded >= 0:
            return True
        return self._is_showdown(state)

    def _is_showdown(self, state: LeducState) -> bool:
        """Check if the current round's betting is complete and it's showdown time."""
        return state.round_num == 1 and self._round_complete(state)

    def _round_complete(self, state: LeducState) -> bool:
        """Check if the current betting round is complete."""
        h = state.history[state.round_num] if state.round_num < len(state.history) else ()
        if len(h) < 2:
            return False

        # Round is complete when both players have acted and actions are "closed"
        # Check-Check, Bet-Call, Bet-Raise-Call, Check-Bet-Call, etc.

        # If last two actions don't include a raise, round is over
        # (both checked, or last action was a call)
        if len(h) >= 2:
            if h[-1] == CHECK_CALL:
                # Call after raise, or check after check
                if len(h) == 2 and h[0] == CHECK_CALL:
                    return True  # check-check
                if any(a == RAISE for a in h):
                    return True  # ...raise-call
                if len(h) >= 2:
                    return True  # multiple checks

            if h[-1] == FOLD:
                return True  # fold ends the hand

        return False

    def terminal_utility(self, state: LeducState, player: int) -> float:
        """Compute utility at a terminal state."""
        if state.folded >= 0:
            # Someone folded
            if state.folded == player:
                return -float(state.pot[player])
            else:
                return float(state.pot[1 - player])

        # Showdown
        winner = self._determine_winner(state)
        if winner == -1:
            return 0.0  # tie (shouldn't happen with distinct cards)
        if winner == player:
            return float(state.pot[1 - player])
        return -float(state.pot[player])

    def _determine_winner(self, state: LeducState) -> int:
        """Determine the winner at showdown.

        Pair with community card beats high card.
        If both have pairs or neither has, higher card wins.
        """
        p0_card = state.cards[0]
        p1_card = state.cards[1]
        community = state.cards[2]

        p0_pair = p0_card == community
        p1_pair = p1_card == community

        if p0_pair and not p1_pair:
            return 0
        if p1_pair and not p0_pair:
            return 1
        # Both pair or neither — higher card wins
        if p0_card > p1_card:
            return 0
        if p1_card > p0_card:
            return 1
        return -1  # tie

    def is_chance(self, state: LeducState) -> bool:
        if state.cards is None:
            return True
        return bool(state._deal_community)

    def chance_outcomes(self, state: LeducState) -> list[tuple[Any, float]]:
        if state.cards is None:
            # Deal two private cards from the 6-card deck
            outcomes: list[tuple[Any, float]] = []
            total = 0
            for i in range(len(DECK)):
                for j in range(len(DECK)):
                    if i == j:
                        continue
                    deal = (DECK[i], DECK[j])
                    remaining_indices = tuple(
                        k for k in range(len(DECK)) if k != i and k != j
                    )
                    key = (deal, remaining_indices)
                    # We track by (i, j) pair to get correct probabilities
                    total += 1

            # Generate unique (card_pair, remaining_deck) combinations
            # with correct probabilities
            deal_counts: dict[tuple[int, int, tuple[int, ...]], int] = {}
            for i in range(len(DECK)):
                for j in range(len(DECK)):
                    if i == j:
                        continue
                    remaining = tuple(sorted(DECK[k] for k in range(len(DECK)) if k != i and k != j))
                    key = (DECK[i], DECK[j], remaining)
                    deal_counts[key] = deal_counts.get(key, 0) + 1

            prob_each = 1.0 / total  # = 1/30
            for (c0, c1, remaining), count in deal_counts.items():
                outcomes.append(((c0, c1, remaining), prob_each * count))
            return outcomes

        if state._deal_community:
            # Deal community card from remaining deck
            remaining = state.cards[2]  # stored as remaining deck tuple
            card_counts: dict[int, int] = {}
            for card in remaining:
                card_counts[card] = card_counts.get(card, 0) + 1
            total_remaining = len(remaining)
            outcomes = []
            for card, count in card_counts.items():
                outcomes.append((card, count / total_remaining))
            return outcomes

        return []

    def current_player(self, state: LeducState) -> int:
        """Determine current player from the action history of the current round."""
        h = state.history[state.round_num] if state.round_num < len(state.history) else ()
        return len(h) % 2

    def actions(self, state: LeducState) -> list[int]:
        """Return available actions at the current state."""
        h = state.history[state.round_num] if state.round_num < len(state.history) else ()
        num_raises = sum(1 for a in h if a == RAISE)

        available = []

        if len(h) == 0:
            # First to act: can check or raise (no fold since nothing to call)
            available = [CHECK_CALL, RAISE]
        elif h[-1] == RAISE:
            # Facing a raise: can fold, call, or re-raise (if raises remain)
            available = [FOLD, CHECK_CALL]
            if num_raises < MAX_RAISES_PER_ROUND:
                available.append(RAISE)
        else:
            # After a check: can check or raise
            available = [CHECK_CALL, RAISE]

        return available

    def apply_action(self, state: LeducState, action: Any) -> LeducState:
        """Apply an action and return the new state."""
        if state.cards is None:
            # Chance node: deal cards
            c0, c1, remaining = action
            new_state = state.copy()
            new_state.cards = (c0, c1, remaining)
            return new_state

        if state._deal_community:
            # Deal community card
            new_state = state.copy()
            p0, p1, _remaining = state.cards
            new_state.cards = (p0, p1, action)
            new_state._deal_community = False
            new_state.round_num = 1
            new_state.history.append(())
            return new_state

        # Player action
        new_state = state.copy()
        current_round = state.round_num
        current_player = self.current_player(state)
        h = list(new_state.history[current_round])
        h.append(action)
        new_state.history[current_round] = tuple(h)

        if action == FOLD:
            new_state.folded = current_player
            return new_state

        if action == RAISE:
            bet_size = ROUND_1_BET if current_round == 0 else ROUND_2_BET
            # First, match the opponent's bet (if any), then raise
            opponent = 1 - current_player
            call_amount = max(0, new_state.pot[opponent] - new_state.pot[current_player])
            new_state.pot[current_player] += call_amount + bet_size
            return new_state

        if action == CHECK_CALL:
            # Match opponent's bet
            opponent = 1 - current_player
            call_amount = max(0, new_state.pot[opponent] - new_state.pot[current_player])
            new_state.pot[current_player] += call_amount

            # Check if round is complete
            if self._action_round_complete(tuple(h)) and current_round == 0:
                # Move to community card deal
                new_state._deal_community = True
                # If round 1, will be terminal (handled by is_terminal)
            return new_state

        return new_state

    def _action_round_complete(self, h: tuple[int, ...]) -> bool:
        """Check if a round is complete given its action history."""
        if len(h) < 2:
            return False
        if h[-1] == FOLD:
            return True
        if h[-1] == CHECK_CALL:
            if len(h) == 2 and h[0] == CHECK_CALL:
                return True  # check-check
            if any(a == RAISE for a in h):
                return True  # raise-...-call
            return True
        return False

    def information_set_key(self, state: LeducState) -> str:
        """Create the information set key.

        Player knows: their own card, community card (if dealt), action history.
        """
        player = self.current_player(state)
        card = state.cards[player]
        card_name = RANK_NAMES[card]

        community = state.cards[2] if state.cards[2] is not None and not isinstance(state.cards[2], tuple) else None
        community_str = RANK_NAMES[community] if community is not None else ""

        # Build action history string
        history_parts = []
        for _round_idx, h in enumerate(state.history):
            round_actions = "".join(
                "f" if a == FOLD else ("c" if a == CHECK_CALL else "r")
                for a in h
            )
            history_parts.append(round_actions)
        hist_str = "|".join(history_parts)

        return f"{card_name}:{community_str}:{hist_str}"

    def action_names(self) -> list[str]:
        return ACTION_NAMES_LEDUC
