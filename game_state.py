from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Any, Optional

# Define a generic type for actions, specific to each game implementation
Action = TypeVar('Action')

class GameState(ABC, Generic[Action]):
    """Abstract Base Class for a game state usable with MCTS.

    Defines the essential methods required by the MCTS algorithm
    to explore the game tree and evaluate states. Player 1 is
    assumed to be represented by 1 and Player 2 by -1.
    """

    # Use a class variable for the game title
    # Subclasses MUST override this.
    game_title: str = "Abstract Game (Please Override)"

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Return the player whose turn it is (1 or -1)."""
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[Action]:
        """Return a list of all valid actions for the current player."""
        pass

    @abstractmethod
    def move(self, action: Action) -> 'GameState[Action]':
        """Apply the given action and return the resulting game state.

        This method should return a *new* game state instance
        and not modify the current state in place.
        """
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Return True if the game has ended, False otherwise."""
        pass

    @abstractmethod
    def game_result(self) -> int:
        """Return the outcome of the game from Player 1's perspective.

        Returns:
            1: Player 1 (X) won.
           -1: Player 2 (O) won.
            0: Draw or game is unfinished.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the board state."""
        pass

    # Hash and equality are crucial for MCTS performance (e.g., detecting cycles, transposition tables)
    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash value for the current state."""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if this state is equal to another state."""
        pass

    # --- Optional methods for enhanced CLI ---

    def get_action_prompt(self) -> str:
        """Return a string describing how the user should input their move."""
        # Default implementation if not overridden
        return "Enter your move:"

    def parse_action(self, input_str: str) -> Action:
        """Parse a string input from the user into a valid Action.

        Raises:
            ValueError: If the input string cannot be parsed into a valid Action format.
        """
        # Default implementation - subclasses should override this
        raise NotImplementedError("Action parsing not implemented for this game state.")

    def action_to_string(self, action: Action) -> str:
        """Convert an action into a human-readable string (e.g., for AI moves)."""
        # Default implementation
        return str(action)