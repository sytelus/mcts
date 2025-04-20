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
* A simple command-line interface that lets a human (X) play against the MCTS
  AI (O) or watch two AIs play.  Adjust the constant `SIMULATIONS_PER_MOVE`
  to trade playing strength for thinking time.

Run:
    python sudo_tic_tac_toe.py # Updated filename
"""
from __future__ import annotations

import copy
import math
import random
import sys
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Local imports
from game_state import GameState
from mcts import MonteCarloTreeSearchNode

###############################################################################
# Game logic for Sudo Tic-Tac-Toe
###############################################################################

# Define the specific Action type for this game
SudoAction = Tuple[int, int]  # (local_board_index 0-8, cell_index inside that board 0-8)

# Pre-computed winning triplets (rows, columns, diagonals) in a 3x3 board.
_WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)

# Helper to check for a win in a local 3x3 board
def _is_local_win(board: Sequence[int], player: int) -> bool:
    """Return True if *player* has three in-a-row in *board*."""
    return any(all(board[i] == player for i in line) for line in _WIN_LINES)

class SudoTicTacToeState(GameState):
    """Complete game state for Sudo Tic-Tac-Toe, implementing GameState.

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

    `_current_player` is  1 for X, -1 for O.
    `forced_board` holds the index (0-8) of the local board the current player
    *must* play in (per the rules). It is `None` when the player can choose
    any unfinished local board.
    """
    __slots__ = ("boards", "board_status", "_current_player", "forced_board")

    # --- GameState required class variables ---
    game_title: str = "Sudo Tic-Tac-Toe"
    simulations_per_move: int = 400 # Keep higher value for complex Sudo

    # --- Instance variables ---
    boards: List[List[int]]
    board_status: List[int]
    _current_player: int
    forced_board: Optional[int]

    def __init__(
        self,
        boards: Optional[List[List[int]]] = None,
        board_status: Optional[List[int]] = None,
        current_player: int = 1,
        forced_board: Optional[int] = None,
    ) -> None:
        self.boards = boards if boards is not None else [[0] * 9 for _ in range(9)]
        self.board_status = board_status if board_status is not None else [0] * 9
        self._current_player = current_player # Renamed with underscore
        self.forced_board = forced_board

    # ---------------------------------------------------------------------
    # Implementation of GameState abstract methods
    # ---------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        """Return the player whose turn it is (1 or -1)."""
        return self._current_player

    def get_legal_actions(self) -> List[SudoAction]:
        """Return every *legal* `(board, cell)` pair the current player can play."""
        legal: List[SudoAction] = []

        def empty_cells(b_idx: int) -> List[SudoAction]:
            return [
                (b_idx, c_idx)
                for c_idx, cell in enumerate(self.boards[b_idx])
                if cell == 0
            ]

        if self.forced_board is not None and self.board_status[self.forced_board] == 0:
            return empty_cells(self.forced_board)

        for b_idx, status in enumerate(self.board_status):
            if status == 0:
                legal.extend(empty_cells(b_idx))
        return legal

    def move(self, action: SudoAction) -> GameState:
        """Return a *new* state that results from applying `action`."""
        board_idx, cell_idx = action

        if not (0 <= board_idx <= 8 and 0 <= cell_idx <= 8):
             raise ValueError("Action coordinates out of bounds")
        if self.boards[board_idx][cell_idx] != 0:
             raise ValueError("Illegal move: cell already occupied")
        if self.board_status[board_idx] != 0:
             raise ValueError("Illegal move: local board already decided")
        if self.forced_board is not None and self.forced_board != board_idx and self.board_status[self.forced_board] == 0:
             raise ValueError(f"Illegal move: must play in forced board {self.forced_board}")

        new_boards = copy.deepcopy(self.boards)
        new_status = self.board_status.copy()

        new_boards[board_idx][cell_idx] = self.current_player

        if _is_local_win(new_boards[board_idx], self.current_player):
            new_status[board_idx] = self.current_player
        elif 0 not in new_boards[board_idx]:
            new_status[board_idx] = 2 # draw

        next_forced: Optional[int] = cell_idx
        if new_status[next_forced] != 0:
            next_forced = None

        return SudoTicTacToeState(
            boards=new_boards,
            board_status=new_status,
            current_player=-self.current_player,
            forced_board=next_forced,
        )

    def is_game_over(self) -> bool:
        """Game ends *immediately* when any local board is won."""
        return 1 in self.board_status or -1 in self.board_status or all(
            status != 0 for status in self.board_status
        )

    def game_result(self) -> int:
        """Return outcome from *Player X*'s perspective."""
        if 1 in self.board_status: return 1
        if -1 in self.board_status: return -1
        if all(status != 0 for status in self.board_status): return 0 # Draw
        return 0 # Should not happen if is_game_over is true?
                 # Let's keep 0 for draw/unfinished as per GameState doc

    def __str__(self) -> str:
        """Pretty printable representation of the 9x9 board with indices."""
        symbols = {1: "X", -1: "O"}
        rows: List[str] = []
        separator = "─────┼─────┼─────"  # 17 chars
        header_template = " B {} │ B {} │ B {} "  # 17 chars

        for big_row in range(3):
            rows.append(header_template.format(3 * big_row, 3 * big_row + 1, 3 * big_row + 2))
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
            if big_row < 2:
                rows.append(separator)
        return "\n".join(rows)

    def __hash__(self) -> int:
        return hash((
            tuple(tuple(board) for board in self.boards),
            tuple(self.board_status),
            self.current_player,
            self.forced_board,
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SudoTicTacToeState):
            return False
        return (
            self.current_player == other.current_player
            and self.forced_board == other.forced_board
            and self.board_status == other.board_status
            and self.boards == other.boards
        )

    # --- Optional GameState CLI methods ---

    def get_action_prompt(self) -> str:
        """Provide specific instructions for Sudo Tic-Tac-Toe input."""
        prompt = "Enter your move as <board> <cell> (numbers 0-8)."
        if self.forced_board is not None and self.board_status[self.forced_board] == 0:
            prompt += f" You *must* play in local board {self.forced_board}."
        return prompt

    def parse_action(self, input_str: str) -> SudoAction:
        """Parse <board> <cell> input."""
        try:
            tokens = input_str.strip().split()
            if len(tokens) != 2:
                raise ValueError("Input must be two numbers.")
            b, c = map(int, tokens)
            action = (b, c)
            if not (0 <= b <= 8 and 0 <= c <= 8):
                 raise ValueError("Board and cell numbers must be between 0 and 8.")
            # Further validation (e.g., is cell empty, is board allowed) is done in get_legal_actions/move
            return action
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input format: {e}") from e

    def action_to_string(self, action: SudoAction) -> str:
        """Convert (board, cell) action to readable string."""
        return f"board {action[0]}, cell {action[1]}"

# --- CLI Code and __main__ block removed, moved to main.py ---