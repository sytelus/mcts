#!/usr/bin/env python3
"""Sudo Tic‑Tac‑Toe (a.k.a. *Ultimate Tic‑Tac‑Toe*) Game State.

Rules
-----
The game consists of nine **local** 3 × 3 Tic‑Tac‑Toe boards laid out in a 3 × 3 grid
forming a **meta** board:

    0 | 1 | 2
    --+---+--
    3 | 4 | 5
    --+---+--
    6 | 7 | 8              ← indices of local boards

*Players*: **X** is encoded as ``1``, **O** as ``‑1``.
*Empty cells* are ``0``.

Turn order
~~~~~~~~~~
1.  A move is specified by the tuple ``(local_board_index, cell_index)`` where both
    indices are in ``0…8``.
2.  The very first move may target *any* empty cell on *any* local board.
3.  After a move is played the **cell index** (0–8) of that move dictates the **local
    board** in which the *next* player **must** play, **unless** that target board
    has already been won or is full (draw).
    In that case the next player may choose *any* empty cell on *any* undecided
    local board.  We refer to this as a *free move*.

Deciding a local board
~~~~~~~~~~~~~~~~~~~~~~
* A player **wins** a local board by completing any of the 8 standard 3‑in‑a‑row
  Tic‑Tac‑Toe lines.  The entry in ``board_status`` for that board is set to the
  winning player's value (``1`` or ``‑1``).
* A local board is a **draw** if no cells are empty and no player has three in a
  row.  Its status is set to ``2``.

Deciding the game
~~~~~~~~~~~~~~~~~
* The **meta board** is treated as a 3 × 3 board whose cells are the
  ``board_status`` values (with draws treated as blanks).
  A player **wins the game** by winning three local boards that form a
  straight line on the meta board.
* If **all** local boards are decided (no status ``0``) and neither player has
  a three‑in‑a‑row on the meta board, the game is a **draw**.

This implementation conforms exactly to the rule set above and is compatible
with the `GameState` API expected by the MCTS engine.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

from .game_state import GameState

SudoAction = Tuple[int, int]

_WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


def _is_local_win(board: Sequence[int], player: int) -> bool:
    """Return ``True`` iff *player* has a winning line on *board*."""
    return any(all(board[i] == player for i in line) for line in _WIN_LINES)


def _is_meta_win(board_status: Sequence[int], player: int) -> bool:
    """Return ``True`` iff *player* has won three local boards in a row."""
    return any(all(board_status[i] == player for i in line) for line in _WIN_LINES)


class SudoTicTacToeState(GameState):
    """Concrete `GameState` for Sudo / Ultimate Tic‑Tac‑Toe."""

    __slots__ = ("boards", "board_status", "_current_player", "forced_board")

    # --- class‑level metadata ------------------------------------------- #
    game_title: str = "Sudo Tic‑Tac‑Toe"
    simulations_per_move: int = 400

    # --- instance attributes ------------------------------------------- #
    boards: List[List[int]]
    board_status: List[int]  # 0 undecided, 1 X won, -1 O won, 2 draw
    _current_player: int
    forced_board: Optional[int]

    # ------------------------------------------------------------------- #
    #                          Lifecycle                                  #
    # ------------------------------------------------------------------- #
    def __init__(
        self,
        boards: Optional[List[List[int]]] = None,
        board_status: Optional[List[int]] = None,
        current_player: int = 1,
        forced_board: Optional[int] = None,
    ) -> None:
        self.boards = boards if boards is not None else [[0] * 9 for _ in range(9)]
        self.board_status = board_status if board_status is not None else [0] * 9
        self._current_player = current_player
        self.forced_board = forced_board

    # ------------------------------------------------------------------- #
    #                      GameState interface                            #
    # ------------------------------------------------------------------- #

    # --- current player ------------------------------------------------- #
    @property
    def current_player(self) -> int:  # pragma: no cover
        return self._current_player

    # --- legal actions -------------------------------------------------- #
    def get_legal_actions(self) -> List[SudoAction]:
        """Return all legal moves for the current player."""

        def _empty_cells(b_idx: int) -> List[SudoAction]:
            return [
                (b_idx, c_idx)
                for c_idx, cell in enumerate(self.boards[b_idx])
                if cell == 0
            ]

        # Forced‑board rule applies if that board is not yet decided.
        if self.forced_board is not None and self.board_status[self.forced_board] == 0:
            return _empty_cells(self.forced_board)

        # Otherwise the player may move in any undecided local board.
        legal: List[SudoAction] = []
        for b_idx, status in enumerate(self.board_status):
            if status == 0:
                legal.extend(_empty_cells(b_idx))
        return legal

    # --- state transition ---------------------------------------------- #
    def move(self, action: SudoAction) -> "SudoTicTacToeState":
        """Return the successor state resulting from *action*."""
        board_idx, cell_idx = action

        # Validate action ------------------------------------------------- #
        if not (0 <= board_idx <= 8 and 0 <= cell_idx <= 8):
            raise ValueError("Action coordinates out of bounds 0–8.")
        if self.boards[board_idx][cell_idx] != 0:
            raise ValueError("Illegal move: cell already occupied.")
        if self.board_status[board_idx] != 0:
            raise ValueError("Illegal move: local board already decided.")
        if (
            self.forced_board is not None
            and self.forced_board != board_idx
            and self.board_status[self.forced_board] == 0
        ):
            raise ValueError(
                f"Illegal move: must play in forced board {self.forced_board}"
            )

        # Apply move ------------------------------------------------------ #
        new_boards = copy.deepcopy(self.boards)
        new_status = self.board_status.copy()

        new_boards[board_idx][cell_idx] = self._current_player

        # Update local board status -------------------------------------- #
        if _is_local_win(new_boards[board_idx], self._current_player):
            new_status[board_idx] = self._current_player
        elif 0 not in new_boards[board_idx]:
            new_status[board_idx] = 2  # local draw

        # Determine forced board for the next player --------------------- #
        next_forced: Optional[int] = cell_idx
        if next_forced is not None and new_status[next_forced] != 0:
            next_forced = None  # free move if target board decided

        return SudoTicTacToeState(
            boards=new_boards,
            board_status=new_status,
            current_player=-self._current_player,
            forced_board=next_forced,
        )

    # --- terminal checks ------------------------------------------------ #
    def is_game_over(self) -> bool:
        """Return ``True`` if the game is over (meta-win or draw)."""
        return (
            _is_meta_win(self.board_status, 1)
            or _is_meta_win(self.board_status, -1)
            or all(status != 0 for status in self.board_status)
        )

    def game_result(self) -> int:
        """Outcome based on meta-board: 1 (X win), -1 (O win), 0 (draw/unfinished)."""
        if _is_meta_win(self.board_status, 1):
            return 1
        if _is_meta_win(self.board_status, -1):
            return -1
        if all(status != 0 for status in self.board_status):
            return 0  # draw
        return 0  # unfinished

    # --- display -------------------------------------------------------- #
    def __str__(self) -> str:  # pragma: no cover
        symbols = {1: "X", -1: "O"}
        rows: List[str] = []
        separator = "───────┼───────┼───────"
        header = " B {} │ B {} │ B {} "

        for big_row in range(3):
            rows.append(header.format(3 * big_row, 3 * big_row + 1, 3 * big_row + 2))
            for small_row in range(3):
                parts: List[str] = []
                for big_col in range(3):
                    b_idx = 3 * big_row + big_col
                    start = 3 * small_row
                    cells = [
                        symbols.get(self.boards[b_idx][start + i], str(start + i))
                        for i in range(3)
                    ]
                    parts.append(" ".join(cells))
                rows.append(" │ ".join(parts))
            if big_row < 2:
                rows.append(separator)
        return "\n".join(rows)

    # --- hashing / equality -------------------------------------------- #
    def __hash__(self) -> int:  # pragma: no cover
        return hash(
            (
                tuple(tuple(board) for board in self.boards),
                tuple(self.board_status),
                self._current_player,
                self.forced_board,
            )
        )

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        if not isinstance(other, SudoTicTacToeState):
            return False
        return (
            self.boards == other.boards
            and self.board_status == other.board_status
            and self._current_player == other._current_player
            and self.forced_board == other.forced_board
        )

    # --- CLI helpers ---------------------------------------------------- #
    def get_action_prompt(self) -> str:  # pragma: no cover
        prompt = "Enter your move as <board> <cell> (0–8)."
        if self.forced_board is not None and self.board_status[self.forced_board] == 0:
            prompt += f" You *must* play in local board {self.forced_board}."
        return prompt

    def parse_action(self, input_str: str) -> SudoAction:  # pragma: no cover
        try:
            b_str, c_str = input_str.strip().split()
            b, c = int(b_str), int(c_str)
            if not (0 <= b <= 8 and 0 <= c <= 8):
                raise ValueError
            return b, c
        except ValueError as exc:
            raise ValueError("Input must be two integers 0–8 separated by a space.") from exc

    def action_to_string(self, action: SudoAction) -> str:  # pragma: no cover
        return f"board {action[0]}, cell {action[1]}"
