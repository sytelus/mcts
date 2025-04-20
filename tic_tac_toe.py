from __future__ import annotations
import copy
from typing import List, Optional, Tuple, Any

from game_state import GameState

# Action for Tic-Tac-Toe is now a tuple containing the cell index
TttAction = Tuple[int]

# Pre-computed winning lines for a 3x3 board
_WIN_LINES_TTT: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)

class TicTacToeState(GameState):
    """Represents the state of a standard Tic-Tac-Toe game.

    Action is represented as Tuple[int].
    """
    __slots__ = ("board", "_current_player")

    # --- GameState required class variables ---
    game_title: str = "Standard Tic-Tac-Toe"
    simulations_per_move: int = 100 # Fewer needed for simple TTT

    # --- Instance variables ---
    board: List[int]  # 0: empty, 1: X, -1: O
    _current_player: int

    def __init__(self, board: Optional[List[int]] = None, current_player: int = 1) -> None:
        self.board = board if board is not None else [0] * 9
        self._current_player = current_player

    @property
    def current_player(self) -> int:
        return self._current_player

    def get_legal_actions(self) -> List[TttAction]:
        """Return indices of empty cells, wrapped in tuples."""
        return [(i,) for i, cell in enumerate(self.board) if cell == 0]

    def move(self, action: TttAction) -> TicTacToeState:
        """Place the current player's mark at the cell index from the action tuple."""
        if not isinstance(action, tuple) or len(action) != 1:
            raise ValueError("Action must be a tuple containing a single integer.")
        cell_index = action[0]

        if not (0 <= cell_index <= 8):
             raise ValueError("Action cell index must be between 0 and 8.")
        if self.board[cell_index] != 0:
            raise ValueError("Illegal move: cell already occupied.")

        new_board = self.board.copy()
        new_board[cell_index] = self.current_player
        return TicTacToeState(board=new_board, current_player=-self.current_player)

    def _check_win(self) -> int:
        """Check if a player has won. Returns winning player (1 or -1) or 0."""
        for player in [1, -1]:
            if any(all(self.board[i] == player for i in line) for line in _WIN_LINES_TTT):
                return player
        return 0

    def is_game_over(self) -> bool:
        """Game is over if there is a win or the board is full."""
        return self._check_win() != 0 or all(cell != 0 for cell in self.board)

    def game_result(self) -> int:
        """Return 1 if X won, -1 if O won, 0 for a draw."""
        winner = self._check_win()
        if winner != 0:
            return winner
        if all(cell != 0 for cell in self.board): # Draw
            return 0
        return 0 # Game not finished (technically shouldn't be called if not is_game_over)

    def __str__(self) -> str:
        """Display board using cell indices 0-8 for empty cells."""
        symbols = {1: "X", -1: "O"}
        rows = []
        for i in range(0, 9, 3):
            # Show index for empty cells
            row_cells = [symbols.get(self.board[j], str(j)) for j in range(i, i + 3)]
            rows.append(" ".join(row_cells))
        return "\n".join(rows)

    def __hash__(self) -> int:
        return hash((tuple(self.board), self.current_player))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TicTacToeState):
            return False
        return self.board == other.board and self.current_player == other.current_player

    # --- Optional CLI methods ---

    def get_action_prompt(self) -> str:
        return "Enter your move as cell number (0-8):\n0 1 2\n3 4 5\n6 7 8"

    def parse_action(self, input_str: str) -> TttAction:
        try:
            cell_index = int(input_str.strip())
            if not (0 <= cell_index <= 8):
                raise ValueError("Cell number must be between 0 and 8.")
            return (cell_index,) # Return as a tuple
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input: {e}") from e

    def action_to_string(self, action: TttAction) -> str:
        # Extract cell index from tuple for display
        return f"cell {action[0]}"