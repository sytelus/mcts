import math
import sys
from typing import Tuple, Optional, Dict

from games.game_state import GameState
from .search_algorithm import SearchAlgorithm

class MinimaxSearch(SearchAlgorithm): # Renamed class
    """Minimax search algorithm.

    Explores the game tree to a specified depth or until terminal states
    are reached, choosing the move with the best guaranteed outcome.
    Uses alpha-beta pruning implicitly through standard minimax recursion.
    """

    def __init__(self, max_depth: Optional[int] = None, **kwargs) -> None:
        """Initialize MinimaxSearch with an optional maximum search depth."""
        super().__init__(**kwargs)
        self.max_depth = max_depth if max_depth is not None else float('inf')
        if self.max_depth <= 0:
            raise ValueError("Max depth must be positive or None.")
        # Simple caching for minimax results (transposition table)
        self._cache: Dict[GameState, float] = {}

    def next_action(self, state: GameState) -> Tuple:
        """Find the next action using minimax search."""
        if state.is_game_over():
            raise ValueError("Cannot find best action for a terminal state.")

        available_actions = state.available_actions()
        if not available_actions:
            raise RuntimeError("No available actions from the current state.")

        # Clear cache for each new top-level search
        self._cache = {}
        best_action = None
        # Initialize based on player: Max player wants +inf, Min player wants -inf
        best_score = -math.inf if state.current_player == 1 else math.inf

        # Evaluate each possible action
        for action in available_actions:
            next_state = state.move(action)
            score = self._minimax(next_state, 0)

            # Update best score and action based on the current player
            if state.current_player == 1: # Maximizing player (X)
                if score > best_score:
                    best_score = score
                    best_action = action
            else: # Minimizing player (O)
                if score < best_score:
                    best_score = score
                    best_action = action

        if best_action is None:
             # Fallback if no move improves the initial best_score (e.g., all moves lead to loss)
             print("Warning: MinimaxSearch returning first available action (no optimal found).", file=sys.stderr)
             return available_actions[0]

        return best_action

    def _minimax(self, state: GameState, depth: int) -> float:
        """Recursive minimax helper function with caching and depth limit.

        Returns the score of the state from the perspective of Player 1 (X).
        """
        # Check cache first
        if state in self._cache:
            return self._cache[state]

        # Check for terminal state or depth limit
        if state.is_game_over() or depth >= self.max_depth:
            result = float(state.game_result())
            self._cache[state] = result # Cache terminal/depth-limited result
            return result

        available_actions = state.available_actions()
        if not available_actions:
            # No legal moves from non-terminal state? Treat as draw.
            self._cache[state] = 0.0
            return 0.0

        # Recursive step
        if state.current_player == 1: # Maximizing player (X)
            max_score = -math.inf
            for action in available_actions:
                score = self._minimax(state.move(action), depth + 1)
                max_score = max(max_score, score)
            self._cache[state] = max_score # Cache result
            return max_score
        else: # Minimizing player (O)
            min_score = math.inf
            for action in available_actions:
                score = self._minimax(state.move(action), depth + 1)
                min_score = min(min_score, score)
            self._cache[state] = min_score # Cache result
            return min_score