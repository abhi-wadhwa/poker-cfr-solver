"""Abstract base class for extensive-form games.

Defines the interface that CFR solvers use to interact with game implementations.
Each game must represent states, information sets, actions, terminal payoffs,
and chance nodes through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExtensiveFormGame(ABC):
    """Abstract interface for two-player extensive-form games.

    The game is represented as a tree where each node is either:
    - A chance node (nature deals cards)
    - A player decision node (player chooses an action)
    - A terminal node (game is over, payoffs are determined)

    States are opaque objects created by the game. The solver only interacts
    with them through this interface.
    """

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players in the game (always 2 for our solvers)."""

    @abstractmethod
    def initial_state(self) -> Any:
        """Create and return the initial game state (root of the game tree).

        Returns:
            The initial state, typically a chance node for card dealing.
        """

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if a state is terminal (game over).

        Args:
            state: The game state to check.

        Returns:
            True if the game is over at this state.
        """

    @abstractmethod
    def terminal_utility(self, state: Any, player: int) -> float:
        """Get the utility (payoff) for a player at a terminal state.

        Args:
            state: A terminal game state.
            player: The player whose utility to return (0 or 1).

        Returns:
            The payoff for the specified player.
        """

    @abstractmethod
    def is_chance(self, state: Any) -> bool:
        """Check if a state is a chance (nature) node.

        Args:
            state: The game state to check.

        Returns:
            True if nature acts at this state (e.g., dealing cards).
        """

    @abstractmethod
    def chance_outcomes(self, state: Any) -> list[tuple[Any, float]]:
        """Get the possible outcomes at a chance node.

        Args:
            state: A chance node state.

        Returns:
            List of (action, probability) pairs.
        """

    @abstractmethod
    def current_player(self, state: Any) -> int:
        """Get the player who acts at this state.

        Args:
            state: A non-terminal, non-chance state.

        Returns:
            Player index (0 or 1).
        """

    @abstractmethod
    def actions(self, state: Any) -> list[Any]:
        """Get the available actions at a state.

        Args:
            state: A non-terminal, non-chance state.

        Returns:
            List of available actions.
        """

    @abstractmethod
    def apply_action(self, state: Any, action: Any) -> Any:
        """Apply an action to a state and return the resulting state.

        Must not modify the original state (return a new state object).

        Args:
            state: The current game state.
            action: The action to apply.

        Returns:
            The new game state after the action.
        """

    @abstractmethod
    def information_set_key(self, state: Any) -> str:
        """Get the information set key for the current player at this state.

        Two states belong to the same information set if the current player
        cannot distinguish between them (same observable history).

        Args:
            state: A non-terminal, non-chance state.

        Returns:
            A string key uniquely identifying the information set.
        """

    @abstractmethod
    def action_names(self) -> list[str]:
        """Return human-readable names for the actions.

        Returns:
            List of action name strings.
        """
