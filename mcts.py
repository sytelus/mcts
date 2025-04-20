import math
import random
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np

from game_state import GameState
from search_algorithm import SearchAlgorithm # Import the base class

class MonteCarloTreeSearchNode:
    """A node in the (game) search tree for Monte Carlo Tree Search.

    Works with GameState where Action is Tuple.
    """

    def __init__(
        self,
        state: GameState,
        parent: Optional['MonteCarloTreeSearchNode'] = None,
        parent_action: Optional[Tuple] = None,
    ) -> None:
        self.state: GameState = state
        self.parent: Optional['MonteCarloTreeSearchNode'] = parent
        self.parent_action: Optional[Tuple] = parent_action
        self.children: List['MonteCarloTreeSearchNode'] = []
        self._number_of_visits: int = 0
        self._results: defaultdict[int, int] = defaultdict(int)

        # Initialize results for player 1 and player -1
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions: Optional[List[Tuple]] = None # Lazy initialization

    def get_untried_actions(self) -> List[Tuple]:
        """Lazily fetch and store untried actions."""
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    # ------------------------------------------------------------------
    # Basic statistics helpers
    # ------------------------------------------------------------------

    def q(self) -> int:
        """Return *net* wins = wins - losses (from Player 1's perspective)."""
        wins = self._results[1]
        losses = self._results[-1]
        # Adjust result based on whose turn it was *leading* to this state
        # If the parent state's player was -1 (O), we want O's wins - losses
        # MCTS traditionally maximizes the score for the player whose turn it is at the node.
        # However, the UCB formula expects the score from the perspective of the parent node's player.
        # Let's stick to the tutorial's convention: q is always from player 1's perspective.
        return wins - losses

    def n(self) -> int:
        """Return the total number of visits to this node."""
        return self._number_of_visits

    # ------------------------------------------------------------------
    # Tree expansion / traversal helpers
    # ------------------------------------------------------------------

    def expand(self) -> 'MonteCarloTreeSearchNode':
        """Expand the tree by adding a new child node for an untried action."""
        untried = self.get_untried_actions()
        action: Tuple = untried.pop()
        next_state = self.state.move(action)
        # Create the child node using the same class
        child_node = self.__class__(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        """Check if the node represents a terminal game state."""
        return self.state.is_game_over()

    def rollout(self) -> int:
        """Simulate a random playout from the current state to a terminal state."""
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves: List[Tuple] = current_rollout_state.get_legal_actions()
            action: Tuple = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result() # Result is from player 1's perspective

    def rollout_policy(self, possible_moves: Sequence[Tuple]) -> Tuple:
        """Select a move during the rollout phase (default: random)."""
        return random.choice(possible_moves)

    def backpropagate(self, result: int) -> None:
        """Update visit counts and results up the tree from this node."""
        self._number_of_visits += 1
        # Result is always from player 1's perspective (1, -1, or 0)
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self) -> bool:
        """Check if all legal actions from this state have been explored."""
        return len(self.get_untried_actions()) == 0

    def best_child(self, c_param: float = math.sqrt(2)) -> 'MonteCarloTreeSearchNode':
        """Select the child node with the highest UCB1 score.

        Uses the UCB1 formula: q/n + c * sqrt(2 * log(N) / n),
        where q = net wins, n = visits for child, N = visits for parent.
        """
        log_N = math.log(self.n())
        best_score = -float('inf')
        best_node: Optional['MonteCarloTreeSearchNode'] = None

        for child in self.children:
            # The q value should be from the perspective of the *parent* node's player.
            # If the current node represents player -1's turn, we should flip the q value.
            # The stored q() is always from player 1's perspective.
            # The parent node's player is self.state.current_player
            q_perspective = child.q() * self.state.current_player # Flip score if parent is O

            ucb_score = (q_perspective / child.n()) + c_param * math.sqrt(2 * log_N / child.n())

            if ucb_score > best_score:
                best_score = ucb_score
                best_node = child

        if best_node is None:
             # Should not happen if node is fully expanded and not terminal
             raise ValueError("No best child found for non-terminal, fully expanded node.")
        # Return type is already NodeType, should be okay
        return best_node

    def _tree_policy(self) -> 'MonteCarloTreeSearchNode':
        """Select or expand a node according to the tree policy (UCB1)."""
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulations_number: int = 200) -> Tuple:
        """Run MCTS simulations and return the best action found.

        Selects the action leading to the child node with the highest visit count.
        """
        if self.is_terminal_node():
             raise ValueError("best_action called on a terminal node")

        for _ in range(simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        if not self.children:
             # May happen if simulations_number is 0 or state has no moves
             raise RuntimeError("No children found after simulations. Cannot determine best action.")

        # After simulations, choose child with highest visit count (exploitation)
        best = max(self.children, key=lambda node: node.n())

        if best.parent_action is None:
            # This should be impossible if `best` is a child node
            raise RuntimeError("Best child node has no parent action.")

        return best.parent_action

# New wrapper class implementing the SearchAlgorithm interface
class MCTSAlgorithm(SearchAlgorithm):
    """Wraps the MonteCarloTreeSearchNode's search logic."""

    def __init__(self, simulations_per_move: int = 200, **kwargs) -> None:
        """Initialize MCTS with the number of simulations per move."""
        super().__init__(**kwargs)
        if simulations_per_move <= 0:
             raise ValueError("Simulations per move must be positive.")
        self.simulations_per_move = simulations_per_move

    def next_action(self, state: GameState) -> Tuple:
        """Uses the MonteCarloTreeSearchNode's best_action method."""
        # Create the root node
        root = MonteCarloTreeSearchNode(state=state)
        # Call the existing best_action method on the root node
        # This preserves the exact MCTS logic from the user's context
        return root.best_action(simulations_number=self.simulations_per_move)