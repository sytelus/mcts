import math
import random
import sys
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np

# Updated imports for new structure
from games.game_state import GameState # Import from games package
from .search_algorithm import SearchAlgorithm # Relative import

class MonteCarloTreeSearchNode:
    """A node in the (game) search tree for Monte Carlo Tree Search.

    Works with GameState where Action is Tuple.
    """
    def __init__(self, state: GameState, parent: Optional['MonteCarloTreeSearchNode'] = None, parent_action: Optional[Tuple] = None) -> None:
        self.state: GameState = state
        self.parent: Optional['MonteCarloTreeSearchNode'] = parent
        self.parent_action: Optional[Tuple] = parent_action
        self.children: List['MonteCarloTreeSearchNode'] = []
        self._number_of_visits: int = 0
        self._results: defaultdict[int, int] = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions: Optional[List[Tuple]] = None

    def get_untried_actions(self) -> List[Tuple]:
        # cache the available actions, pop them one by one as the evaluation progresses
        if self._untried_actions is None:
            self._untried_actions = self.state.available_actions()
        return self._untried_actions

    def q(self) -> int:
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses

    def n(self) -> int:
        return self._number_of_visits

    def expand(self) -> 'MonteCarloTreeSearchNode':
        untried = self.get_untried_actions()
        action: Tuple = untried.pop()
        next_state = self.state.move(action)
        child_node = self.__class__(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    def rollout(self) -> int:
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves: List[Tuple] = current_rollout_state.available_actions()
            if not possible_moves:
                return 0
            action: Tuple = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def rollout_policy(self, possible_moves: Sequence[Tuple]) -> Tuple:
        return random.choice(possible_moves)

    def backpropagate(self, result: int) -> None:
        self._number_of_visits += 1
        # The result is always from Player 1's perspective (1, -1, 0)
        # We need to record it based on which player *won* or *lost*.
        # MCTS typically stores wins/losses for the player whose turn it *was* at the parent.
        # The backpropagation needs the result relative to the player who *made the move* into this node.
        # Since self.parent.state.current_player made the move, we check if the result matches that player.
        # However, the current _results defaultdict directly uses 1 and -1 keys. Let's keep that structure.
        # A positive q() favors player 1, negative favors player -1.
        # result=1 means player 1 won, result=-1 means player -1 won.
        # We simply add the result to the corresponding key.
        if result != 0: # Don't record draws in the win/loss counts
            self._results[result] += 1

        if self.parent:
            # Backpropagate the same result (always from P1's perspective)
            self.parent.backpropagate(result)

    def is_fully_expanded(self) -> bool:
        return len(self.get_untried_actions()) == 0

    def best_child(self, c_param: float = math.sqrt(2)) -> 'MonteCarloTreeSearchNode':
        """Selects the child node with the highest UCB1 score."""
        if self.n() == 0:
            # This can happen if the node was just expanded and hasn't been visited via backpropagation yet.
            # In standard MCTS, the parent node's visit count (N) should be available.
            # If the root node (parent=None) has n=0, it means no simulations ran, which is handled in best_action.
            # Let's assume if self.n() is 0, log_N should conceptually be 0, but avoid math domain error.
            # A simple approach is to return the first child if n=0, though this is debatable.
            # A better approach might be to ensure backpropagation happens at least once before selection.
            # For now, raise error as it indicates an unusual state or potential issue in the MCTS loop.
            raise ValueError("best_child called on node with zero visits. Ensure backpropagation occurs.")

        log_N = math.log(self.n()) # Total visits to the parent node (self)
        best_score = -float('inf')
        best_node: Optional['MonteCarloTreeSearchNode'] = None

        for child in self.children:
            child_n = child.n() # Visits to the child node
            if child_n == 0:
                # Prioritize exploring unvisited children
                ucb_score = float('inf')
            else:
                # Exploitation term: Average reward from the child node's perspective.
                # child.q() = wins(P1) - losses(P1) = wins(P1) - wins(P-1)
                # We need the score from the perspective of the *current* node's player (self.state.current_player).
                # If self is P1 (maximizer), use child.q() directly.
                # If self is P-1 (minimizer), use -child.q().
                # This is equivalent to child.q() * self.state.current_player.
                exploitation_score = child.q() * self.state.current_player / child_n

                # Exploration term (Standard UCB1)
                exploration_score = c_param * math.sqrt(log_N / child_n)

                ucb_score = exploitation_score + exploration_score

            if ucb_score > best_score:
                best_score = ucb_score
                best_node = child

        if best_node is None:
            # This should not happen if the node has children and n() > 0.
            raise ValueError("No children found or error in UCB calculation.")

        return best_node

    def _tree_policy(self) -> 'MonteCarloTreeSearchNode':
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                if not current_node.children:
                    return current_node
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulations_number: int = 200) -> Tuple:
        if self.is_terminal_node():
             raise ValueError("best_action called on a terminal node")
        available_actions = self.state.available_actions()
        if not available_actions:
             raise RuntimeError("No available actions from the root state.")
        if simulations_number <= 0:
             print("Warning: MCTS best_action called with 0 simulations, returning random action.", file=sys.stderr)
             return random.choice(available_actions)

        # Perform MCTS simulations
        for _ in range(simulations_number):
            v = self._tree_policy() # Selection & Expansion
            reward = v.rollout()   # Simulation
            v.backpropagate(reward) # Backpropagation

        # After simulations, choose the best move based on visit count
        if not self.children:
             print("Warning: MCTS root has no children after simulations, returning first available action.", file=sys.stderr)
             # This can happen if the only actions lead immediately to terminal states
             # and simulations_number is very low, or if root itself is terminal but wasn't caught.
             return available_actions[0]

        # Select the child node corresponding to the most visited action
        # note that we use robust child criterion for the best node after simulations instead of UCB
        best = max(self.children, key=lambda node: node.n())
        if best.parent_action is None:
            # This should be impossible if the node is a child created by expand()
            raise RuntimeError("Best child node has no parent action.")
        return best.parent_action

class MCTSAlgorithm(SearchAlgorithm):
    """Wraps the MonteCarloTreeSearchNode's search logic."""
    def __init__(self, simulations_per_move: int = 200, **kwargs) -> None:
        super().__init__(**kwargs)
        if simulations_per_move <= 0:
             raise ValueError("Simulations per move must be positive.")
        self.simulations_per_move = simulations_per_move

    def next_action(self, state: GameState) -> Tuple:
        """Uses the MonteCarloTreeSearchNode's best_action method."""
        root = MonteCarloTreeSearchNode(state=state)
        return root.best_action(simulations_number=self.simulations_per_move)