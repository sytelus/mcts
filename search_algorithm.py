from abc import ABC, abstractmethod
from typing import Tuple, Optional

from game_state import GameState

class SearchAlgorithm(ABC):
    """Abstract Base Class for AI search algorithms.

    Defines the interface for finding the best action from a given game state.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the algorithm. kwargs can be used for algorithm-specific settings."""
        pass

    @abstractmethod
    def next_action(self, state: GameState) -> Tuple:
        """Given a game state, return the next action determined by the algorithm.

        Args:
            state: The current GameState.

        Returns:
            The next action (as a Tuple) found.

        Raises:
            RuntimeError: If no legal actions are available or search fails.
            ValueError: If called on a terminal state.
        """
        pass