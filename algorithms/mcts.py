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
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self) -> bool:
        return len(self.get_untried_actions()) == 0

    def best_child(self, c_param: float = math.sqrt(2)) -> 'MonteCarloTreeSearchNode':
        if self.n() == 0:
            raise ValueError("best_child called on node with zero visits.")
        log_N = math.log(self.n())
        best_score = -float('inf')
        best_node: Optional['MonteCarloTreeSearchNode'] = None
        for child in self.children:
            child_n = child.n()
            if child_n == 0:
                ucb_score = float('inf')
            else:
                q_perspective = child.q() * self.state.current_player
                ucb_score = (q_perspective / child_n) + c_param * math.sqrt(2 * log_N / child_n)
            if ucb_score > best_score:
                best_score = ucb_score
                best_node = child
        if best_node is None:
            raise ValueError("No children found to select best_child from.")
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
        for _ in range(simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        if not self.children:
             print("Warning: MCTS root has no children after simulations, returning first available action.", file=sys.stderr)
             return available_actions[0]
        best = max(self.children, key=lambda node: node.n())
        if best.parent_action is None:
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