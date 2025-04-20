#!/usr/bin/env python3
"""Sudo Tic-Tac-Toe Game State implementation.

Based on the tutorial at https://ai-boson.github.io/mcts/.
"""
from __future__ import annotations

import copy
import math
import random
import sys
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np

# Local imports
from .game_state import GameState # Relative import
# Remove unused MCTS import from this file
# from mcts import MonteCarloTreeSearchNode

# ... (rest of the file is unchanged) ...
SudoAction = Tuple[int, int]
_WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)
def _is_local_win(board: Sequence[int], player: int) -> bool:
    return any(all(board[i] == player for i in line) for line in _WIN_LINES)

class SudoTicTacToeState(GameState):
    """Complete game state for Sudo Tic-Tac-Toe, implementing GameState.

    Action is Tuple[int, int].
    """
    __slots__ = ("boards", "board_status", "_current_player", "forced_board")

    game_title: str = "Sudo Tic-Tac-Toe"
    simulations_per_move: int = 400

    boards: List[List[int]]
    board_status: List[int]
    _current_player: int
    forced_board: Optional[int]

    def __init__(self, boards: Optional[List[List[int]]] = None, board_status: Optional[List[int]] = None, current_player: int = 1, forced_board: Optional[int] = None) -> None:
        self.boards = boards if boards is not None else [[0] * 9 for _ in range(9)]
        self.board_status = board_status if board_status is not None else [0] * 9
        self._current_player = current_player
        self.forced_board = forced_board

    @property
    def current_player(self) -> int:
        return self._current_player

    def get_legal_actions(self) -> List[SudoAction]:
        legal: List[SudoAction] = []
        def empty_cells(b_idx: int) -> List[SudoAction]:
            return [(b_idx, c_idx) for c_idx, cell in enumerate(self.boards[b_idx]) if cell == 0]
        if self.forced_board is not None and self.board_status[self.forced_board] == 0:
            return empty_cells(self.forced_board)
        for b_idx, status in enumerate(self.board_status):
            if status == 0:
                legal.extend(empty_cells(b_idx))
        return legal

    def move(self, action: SudoAction) -> GameState:
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
            new_status[board_idx] = 2
        next_forced: Optional[int] = cell_idx
        # Add temporary variable to satisfy mypy about potential None index
        check_board_idx = next_forced
        if check_board_idx is not None and new_status[check_board_idx] != 0:
            next_forced = None
        return SudoTicTacToeState(boards=new_boards, board_status=new_status, current_player=-self.current_player, forced_board=next_forced)

    def is_game_over(self) -> bool:
        return 1 in self.board_status or -1 in self.board_status or all(status != 0 for status in self.board_status)

    def game_result(self) -> int:
        if 1 in self.board_status: return 1
        if -1 in self.board_status: return -1
        if all(status != 0 for status in self.board_status): return 0
        return 0

    def __str__(self) -> str:
        symbols = {1: "X", -1: "O"}
        rows: List[str] = []
        separator = "─────┼─────┼─────"
        header_template = " B {} │ B {} │ B {} "
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
        return hash((tuple(tuple(board) for board in self.boards), tuple(self.board_status), self.current_player, self.forced_board))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SudoTicTacToeState):
            return False
        return (self.current_player == other.current_player and self.forced_board == other.forced_board and self.board_status == other.board_status and self.boards == other.boards)

    def get_action_prompt(self) -> str:
        prompt = "Enter your move as <board> <cell> (numbers 0-8)."
        if self.forced_board is not None and self.board_status[self.forced_board] == 0:
            prompt += f" You *must* play in local board {self.forced_board}."
        return prompt

    def parse_action(self, input_str: str) -> SudoAction:
        try:
            tokens = input_str.strip().split()
            if len(tokens) != 2:
                raise ValueError("Input must be two numbers.")
            b, c = map(int, tokens)
            action = (b, c)
            if not (0 <= b <= 8 and 0 <= c <= 8):
                 raise ValueError("Board and cell numbers must be between 0 and 8.")
            return action
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input format: {e}") from e

    def action_to_string(self, action: SudoAction) -> str:
        if not isinstance(action, tuple) or len(action) != 2:
            return f"unknown action format: {action}"
        return f"board {action[0]}, cell {action[1]}"

# --- CLI Code and __main__ block removed, moved to main.py ---