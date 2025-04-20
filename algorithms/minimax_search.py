import math
import sys
from typing import Tuple, Optional, Dict

from games.game_state import GameState
from .search_algorithm import SearchAlgorithm

class MinimaxSearch(SearchAlgorithm): # Renamed class
    """Minimax search algorithm with Alpha-Beta Pruning.

    Explores the game tree to a specified depth or until terminal states
    are reached, choosing the move with the best guaranteed outcome
    while pruning branches that cannot influence the final decision.
    """

    def __init__(self, max_depth: Optional[int] = None, **kwargs) -> None:
        """Initialize MinimaxSearch with an optional maximum search depth."""
        super().__init__(**kwargs)
        self.max_depth = max_depth if max_depth is not None else float('inf')
        if self.max_depth <= 0:
            raise ValueError("Max depth must be positive or None.")
        # Simple caching for minimax results (transposition table)
        # Note: Alpha-beta requires storing bounds, not just exact values for cache hits
        # For simplicity, this cache stores exact scores, which might limit pruning effectiveness
        # if a cached state is encountered with looser bounds.
        self._cache: Dict[GameState, float] = {}

    def next_action(self, state: GameState) -> Tuple:
        """Find the next action using minimax search with alpha-beta pruning."""
        if state.is_game_over():
            raise ValueError("Cannot find best action for a terminal state.")

        available_actions = state.available_actions()
        if not available_actions:
            raise RuntimeError("No available actions from the current state.")

        # Clear cache for each new top-level search
        self._cache = {}
        best_action = None
        # Initialize alpha and beta
        alpha = -math.inf
        beta = math.inf

        # Determine best score initialization based on player
        best_score = -math.inf if state.current_player == 1 else math.inf

        # Evaluate each possible action
        for action in available_actions:
            next_state = state.move(action)
            # Pass alpha and beta to the recursive call
            score = self._minimax(next_state, 0, alpha, beta)

            # Update best score and action based on the current player
            if state.current_player == 1: # Maximizing player (X)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score) # Update alpha
                # No prune check here as we need to evaluate all top-level actions
            else: # Minimizing player (O)
                if score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score) # Update beta
                # No prune check here

            # Note: Pruning happens *within* the _minimax calls initiated here.
            # The alpha/beta updates here refine the bounds for *subsequent* top-level action evaluations.

        if best_action is None:
             # Fallback if no move improves the initial best_score (e.g., all moves lead to loss)
             print("Warning: MinimaxSearch returning first available action (no optimal found).", file=sys.stderr)
             return available_actions[0]

        return best_action

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        """Recursive minimax helper function with alpha-beta pruning, caching, and depth limit.

        Args:
            state: The current game state.
            depth: The current depth in the search tree.
            alpha: The best value that the maximizing player currently can guarantee at this level or above.
            beta: The best value that the minimizing player currently can guarantee at this level or above.

        Returns:
            The score of the state from the perspective of Player 1 (X).
        """
        # Check cache first (Basic caching - see note in __init__)
        if state in self._cache:
            # Potential issue: cached value might be from different alpha/beta bounds.
            # A full transposition table would store bounds (upper, lower, exact).
            # For simplicity, we return the cached value directly.
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

        # Recursive step with alpha-beta pruning
        if state.current_player == 1: # Maximizing player (X)
            max_score = -math.inf
            for action in available_actions:
                score = self._minimax(state.move(action), depth + 1, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, max_score) # Update alpha *before* the check
                # The prune condition compares the best score the maximizer can guarantee (alpha)
                # against the best score the minimizer can guarantee (beta).
                if alpha >= beta: # Pruning condition (Corrected)
                    break # Beta cutoff
            self._cache[state] = max_score # Cache result
            return max_score
        else: # Minimizing player (O)
            min_score = math.inf
            for action in available_actions:
                score = self._minimax(state.move(action), depth + 1, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, min_score) # Update beta *before* the check
                # The prune condition compares the best score the minimizer can guarantee (beta)
                # against the best score the maximizer can guarantee (alpha).
                if beta <= alpha: # Pruning condition (Corrected)
                    break # Alpha cutoff
            self._cache[state] = min_score # Cache result
            return min_score