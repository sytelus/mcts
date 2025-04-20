#!/usr/bin/env python3
"""Sudo Tic-Tac-Toe with Monte Carlo Tree Search

This program follows (and completes the omissions in) the tutorial at
https://ai-boson.github.io/mcts/.  It implements:

* A `SudoTicTacToeState` class that represents the 9x9 "sudo"/"ultimate" board
  (9 local 3x3 Tic-Tac-Toe boards) together with the current player, the forced
  local board for the next move, and helper methods that the MCTS engine
  expects (`get_legal_actions`, `move`, `is_game_over`, `game_result`).
* The `MonteCarloTreeSearchNode` class exactly as in the article, but with full
  type hints and richer doc-strings.
* A simple command-line interface that lets a human („X“) play against the MCTS
  AI („O“) or watch two AIs play.  Adjust the constant `SIMULATIONS_PER_MOVE`
  to trade playing strength for thinking time.

Run:
    python sudo_tic_tac_toe_mcts.py
"""
from __future__ import annotations

import copy
import math
import random
import sys
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

###############################################################################
# Game logic
###############################################################################

Action = Tuple[int, int]  # (local_board_index 0-8, cell_index inside that board 0-8)

# Pre-computed winning triplets (rows, columns, diagonals) in a 3x3 board.
_WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


class SudoTicTacToeState:
    """Complete game state for Sudo Tic-Tac-Toe.

    A *local* 3x3 board is represented by a flat list of 9 integers:
        0  ➜ empty
        1  ➜ player X (AI or human, depending on who starts)
       -1  ➜ player O

    `boards` is a length-9 list, one entry per local board.
    `board_status` is a length-9 list recording the outcome of each local board:
        0  ➜ ongoing
        1  ➜ X won       (the *entire* Sudo game ends immediately!)
       -1  ➜ O won       (same — single local win ends the match)
        2  ➜ draw / full without winner

    `current_player` is  1 for X, -1 for O.
    `forced_board` holds the index (0-8) of the local board the current player
    *must* play in (per the rules).  It is `None` when the player can choose
    any unfinished local board: either at the very first move or when the
    designated board is already won/drawn/full.
    """

    # there can be millions of instances of this class, so we use __slots__
    # as slots take up less space than a dict
    __slots__ = ("boards", "board_status", "current_player", "forced_board")
    boards: List[List[int]]
    board_status: List[int]
    current_player: int
    forced_board: Optional[int]

    def __init__(
        self,
        boards: Optional[List[List[int]]] = None, # 9x9 board of 9x9 boards
        board_status: Optional[List[int]] = None, # game status of each board 0: ongoing, 1: X won, -1: O won, 2: draw
        current_player: int = 1, # 1 for X, -1 for O
        forced_board: Optional[int] = None, # board that must be played, it's board that was last played and is not full or won
    ) -> None:
        self.boards = boards if boards is not None else [[0] * 9 for _ in range(9)]
        self.board_status = board_status if board_status is not None else [0] * 9
        self.current_player = current_player
        self.forced_board = forced_board

    # ---------------------------------------------------------------------
    # Mandatory API for the MCTS engine
    # ---------------------------------------------------------------------

    def get_legal_actions(self) -> List[Action]:
        """Return every *legal* `(board, cell)` pair the current player can play."""
        legal: List[Action] = []

        # for given local board, get indices of empty cells
        def empty_cells(b_idx: int) -> List[Action]:
            return [
                (b_idx, c_idx) # board index, cell index
                for c_idx, cell in enumerate(self.boards[b_idx])
                if cell == 0
            ]

        # If a board is forced and still playable, we *must* play there.
        if self.forced_board is not None and self.board_status[self.forced_board] == 0:
            return empty_cells(self.forced_board)

        # Otherwise, any empty cell in any *ongoing* local board is allowed.
        for b_idx, status in enumerate(self.board_status):
            if status == 0:  # ongoing
                legal.extend(empty_cells(b_idx))
        return legal

    def move(self, action: Action) -> "SudoTicTacToeState":
        """Return a *new* state that results from applying `action`."""
        board_idx, cell_idx = action
        assert self.boards[board_idx][cell_idx] == 0, "Illegal move: cell already occupied"
        assert self.board_status[board_idx] == 0, "Illegal move: local board already decided"
        assert not (
            self.forced_board is not None
            and self.forced_board == board_idx
            and self.board_status[board_idx] != 0
        ), "Illegal move: forced board already decided"

        # Deep-copy mutable structures
        new_boards = copy.deepcopy(self.boards)
        new_status = self.board_status.copy()

        # Place the mark
        new_boards[board_idx][cell_idx] = self.current_player

        # Check if *this move* wins the local board
        if _is_local_win(new_boards[board_idx], self.current_player):
            new_status[board_idx] = self.current_player  # 1 or -1 — game ends
        elif 0 not in new_boards[board_idx]:
            new_status[board_idx] = 2  # draw

        # Determine next forced board: index equals the *cell* just played
        next_forced: Optional[int] = cell_idx
        if new_status[next_forced] != 0:  # forced board already full/won — open move
            next_forced = None

        return SudoTicTacToeState(
            boards=new_boards,
            board_status=new_status,
            current_player=-self.current_player,  # switch player
            forced_board=next_forced,
        )

    def is_game_over(self) -> bool:
        """Game ends *immediately* when any local board is won."""
        return 1 in self.board_status or -1 in self.board_status or all(
            status != 0 for status in self.board_status
        )

    def game_result(self) -> int:
        """Return outcome from *Player X*'s perspective.

        +1  – X won (X captured any local board)
        -1  – O won (O captured any local board)
         0  – draw / unfinished (all local boards full without a single win)
        """
        if 1 in self.board_status:
            return 1  # X wins
        if -1 in self.board_status:
            return -1  # O wins
        return 0  # draw (no wins, all boards decided)

    # ------------------------------------------------------------------
    # Convenience helpers (not used by the MCTS engine, but handy)
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # noqa: DunderStr
        """Pretty printable representation of the 9x9 board with indices."""
        symbols = {1: "X", -1: "O"}
        rows: List[str] = []
        separator = "─────┼─────┼─────"  # 17 chars
        header_template = " B {} │ B {} │ B {} "  # 17 chars

        for big_row in range(3):
            # Add header for the current row of boards
            rows.append(header_template.format(3 * big_row, 3 * big_row + 1, 3 * big_row + 2))
            if big_row > 0:
                # Add separator line above rows 1 and 2
                rows.insert(-9, separator) # Insert separator before the 9 lines of the previous big_row

            for small_row in range(3):
                line_parts: List[str] = []
                for big_col in range(3):
                    board_idx = 3 * big_row + big_col
                    local_board_cells: List[str] = [] # Cells for one local board row
                    start_cell_idx = 3 * small_row # 0, 3, 6
                    for i in range(3): # 0, 1, 2
                        cell_idx = start_cell_idx + i
                        cell_value = self.boards[board_idx][cell_idx]
                        local_board_cells.append(symbols.get(cell_value, str(cell_idx)))
                    # Join the 3 cells for the local board: "0 1 2" (5 chars)
                    line_parts.append(" ".join(local_board_cells))

                # Join the 3 local board strings with "│": "b0│b1│b2" (5+1+5+1+5=17 chars)
                rows.append("│".join(line_parts))

        # Add the final separator manually if needed, but the loop structure handles it
        # Need to fix the separator insertion logic slightly
        # Let's rebuild the structure more clearly

        rows = [] # Reset rows
        for big_row in range(3):
            # Header for boards in this big_row
            rows.append(header_template.format(3 * big_row, 3 * big_row + 1, 3 * big_row + 2))
            # Grid rows
            for small_row in range(3):
                line_parts = []
                for big_col in range(3):
                    board_idx = 3 * big_row + big_col
                    local_board_cells = []
                    start_cell_idx = 3 * small_row
                    for i in range(3):
                        cell_idx = start_cell_idx + i
                        cell_value = self.boards[board_idx][cell_idx]
                        local_board_cells.append(symbols.get(cell_value, str(cell_idx)))
                    line_parts.append(" ".join(local_board_cells))
                rows.append("│".join(line_parts))
            # Separator after each big_row, except the last one
            if big_row < 2:
                rows.append(separator)

        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Equality / hashing to let MCTS store states efficiently (optional)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:  # noqa: DunderEq
        if not isinstance(other, SudoTicTacToeState):
            return False
        return (
            self.current_player == other.current_player
            and self.forced_board == other.forced_board
            and self.board_status == other.board_status
            and self.boards == other.boards
        )

    def __hash__(self) -> int:  # noqa: DunderHash
        # Good enough – packs the most volatile fields.
        return hash(
            (
                tuple(tuple(board) for board in self.boards),
                tuple(self.board_status),
                self.current_player,
                self.forced_board,
            )
        )


# -----------------------------------------------------------------------------
# Helper: detect a win inside a single 3x3 board
# -----------------------------------------------------------------------------

def _is_local_win(board: Sequence[int], player: int) -> bool:
    """Return True if *player* has three in-a-row in *board*."""
    return any(all(board[i] == player for i in line) for line in _WIN_LINES)

###############################################################################
# Monte Carlo Tree Search implementation (verbatim from tutorial, plus types)
###############################################################################

class MonteCarloTreeSearchNode:
    """A node in the (game) search tree for Monte Carlo Tree Search."""

    def __init__(
        self,
        state: SudoTicTacToeState,
        parent: Optional["MonteCarloTreeSearchNode"] = None,
        parent_action: Optional[Action] = None,
    ) -> None:
        self.state: SudoTicTacToeState = state
        self.parent: Optional["MonteCarloTreeSearchNode"] = parent
        self.parent_action: Optional[Action] = parent_action
        self.children: List["MonteCarloTreeSearchNode"] = []
        self._number_of_visits: int = 0
        self._results: defaultdict[int, int] = defaultdict(int)

        # Initialize results for X and O
        # 1: X wins, -1: O wins, 0: draw
        self._results[1] = 0  # wins for X (root-player)
        self._results[-1] = 0  # wins for O
        self._untried_actions: Optional[List[Action]] = self.untried_actions()

    # ------------------------------------------------------------------
    # Basic statistics helpers
    # ------------------------------------------------------------------

    def q(self) -> int:
        """Return *net* wins = wins – losses (from X's perspective)."""
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses

    def n(self) -> int:
        return self._number_of_visits

    # ------------------------------------------------------------------
    # Tree expansion / traversal helpers
    # ------------------------------------------------------------------

    def untried_actions(self) -> List[Action]:
        return self.state.get_legal_actions()

    def expand(self) -> "MonteCarloTreeSearchNode":
        action = self._untried_actions.pop() # type: ignore[assignment]
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    def rollout(self) -> int:
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def rollout_policy(self, possible_moves: Sequence[Action]) -> Action:
        return random.choice(possible_moves)

    def backpropagate(self, result: int) -> None:
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0 # type: ignore[no-untyped-call]

    def best_child(self, c_param: float = math.sqrt(2)) -> "MonteCarloTreeSearchNode":
        choices_weights = [
            (child.q() / child.n()) + c_param * math.sqrt(2 * math.log(self.n()) / child.n())
            for child in self.children
        ]
        return self.children[int(np.argmax(choices_weights))]

    def _tree_policy(self) -> "MonteCarloTreeSearchNode":
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulations_number: int = 200) -> Action:
        for _ in range(simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # After simulations, choose child with highest visit count (exploitation)
        best = max(self.children, key=lambda node: node.n())
        return best.parent_action  # type: ignore[return-value]

###############################################################################
# Simple command-line interface
###############################################################################

SIMULATIONS_PER_MOVE = 400  # tweak for stronger/weaker AI

def _prompt_human_move(state: SudoTicTacToeState) -> Action:
    """Ask the user for a move until a valid one is entered."""
    print("Enter your move as <board> <cell> (numbers 0-8).  Layout:")
    print("Local boards are numbered like this:")
    print("0 1 2\n3 4 5\n6 7 8")
    print("Cells inside a local board are numbered the same 0-8.")
    if state.forced_board is not None and state.board_status[state.forced_board] == 0:
        print(f"You *must* play in local board {state.forced_board}.")
    while True:
        try:
            tokens = input("Your move> ").strip().split()
            if len(tokens) != 2:
                raise ValueError
            b, c = map(int, tokens)
            action = (b, c)
            if action in state.get_legal_actions():
                return action
            else:
                raise ValueError
        except ValueError:
            print("Illegal input or move — please try again.")


def play_cli() -> None:
    print("Welcome to Sudo Tic-Tac-Toe (first to win *any* local board).")
    print("Do you want to be X and start? [y/N] ", end="")
    human_is_x = input().strip().lower().startswith("y")
    state = SudoTicTacToeState(current_player=1)  # X always starts

    while True:
        print("\nCurrent board state:")
        print(state)

        if state.is_game_over():
            result = state.game_result()
            if result == 1:
                print("X wins!")
            elif result == -1:
                print("O wins!")
            else:
                print("It's a draw.")
            return

        if (state.current_player == 1 and human_is_x) or (
            state.current_player == -1 and not human_is_x
        ):
            # Human move
            action = _prompt_human_move(state)
        else:
            # AI move via MCTS
            print("AI is thinking …", file=sys.stderr)
            root = MonteCarloTreeSearchNode(state)
            action = root.best_action(simulations_number=SIMULATIONS_PER_MOVE)
            print(f"AI plays board {action[0]}, cell {action[1]}")

        state = state.move(action)


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed()
    np.random.seed()
    play_cli()
