# MCTS Game Engine with Tic-Tac-Toe Variants

This project implements a flexible game-playing framework featuring a Monte Carlo Tree Search (MCTS) AI, alongside other search algorithms, applied to different turn-based games like standard Tic-Tac-Toe and Sudo (Ultimate) Tic-Tac-Toe.

The core design utilizes abstract base classes (`GameState`, `SearchAlgorithm`) to allow easy extension with new games and AI search strategies.

## Features

*   **Multiple Games:**
    *   Standard Tic-Tac-Toe (3x3)
    *   Sudo / Ultimate Tic-Tac-Toe (9 local 3x3 boards with forced-move rule)
*   **Multiple AI Algorithms:**
    *   Monte Carlo Tree Search (MCTS): A probabilistic search algorithm suitable for complex games. Configurable simulation count per game.
    *   Brute-Force Minimax: A deterministic search exploring the game tree (with optional depth limit).
*   **Command-Line Interface (CLI):**
    *   Select which game to play.
    *   Select which AI algorithm to play against.
    *   Choose to play as Player 1 (X) or Player 2 (O).
    *   Interactive move input for the human player.
    *   Clear display of the game board state.
*   **Extensible Design:**
    *   Add new games by inheriting from `GameState` and implementing the required methods.
    *   Add new search algorithms by inheriting from `SearchAlgorithm`.

## File Structure

```
.
├── main.py                 # Main entry point, CLI logic
├── games/                  # Package for game implementations
│   ├── __init__.py
│   ├── game_state.py       # Abstract Base Class (ABC) for game states
│   ├── tic_tac_toe.py      # Standard Tic-Tac-Toe implementation
│   └── sudo_tic_tac_toe.py # Sudo Tic-Tac-Toe implementation
├── algorithms/             # Package for AI search algorithms
│   ├── __init__.py
│   ├── search_algorithm.py # Abstract Base Class (ABC) for search algorithms
│   ├── mcts.py             # MCTS implementation
│   └── brute_force_search.py # Brute-Force Minimax implementation
└── README.md               # This file
```

## Requirements

*   Python 3.x
*   NumPy (`pip install numpy`)

## Installation

1.  Clone or download the repository.
2.  Install the required library:
    ```bash
    pip install numpy
    ```
    *(Or, if you have a `requirements.txt` file: `pip install -r requirements.txt`)*

## Usage

Run the game from your terminal using:

```bash
python main.py
```

You will be prompted to:

1.  **Select Game:** Choose between "Standard Tic-Tac-Toe" and "Sudo Tic-Tac-Toe".
2.  **Select AI Algorithm:** Choose between "MCTS" and "BruteForce (Minimax)".
    *   If you choose BruteForce, you can optionally specify a maximum search depth. Higher depths are stronger but much slower, especially for Sudo Tic-Tac-Toe.
3.  **Choose Player:** Decide if you want to play as 'X' (Player 1, starts first) or 'O' (Player 2).
4.  **Play:**
    *   The current board state will be displayed.
    *   If it's your turn, follow the prompt to enter your move:
        *   **Tic-Tac-Toe:** Enter the cell number (0-8).
        *   **Sudo Tic-Tac-Toe:** Enter the board number (0-8) and then the cell number (0-8) within that board, separated by a space (e.g., `4 0`). Pay attention to the "must play in board X" messages if applicable.
    *   The AI will think and display its chosen move.
    *   The game continues until a win, loss, or draw occurs.

## Extending the Project

### Adding a New Game

1.  Create a new file (e.g., `my_game.py`).
2.  Define a class that inherits from `GameState` (from `game_state.py`).
3.  Implement all the abstract methods defined in `GameState` (`current_player`, `get_legal_actions`, `move`, `is_game_over`, `game_result`, `__str__`, `__hash__`, `__eq__`). Remember that actions must be represented as `Tuple`.
4.  Implement the required class variables `game_title` and `simulations_per_move` (for MCTS).
5.  Optionally, override the CLI helper methods (`get_action_prompt`, `parse_action`, `action_to_string`) for better user interaction.
6.  Import your new game class in `main.py` and add it to the `AVAILABLE_GAMES` dictionary.

### Adding a New Search Algorithm

1.  Create a new file (e.g., `my_search.py`).
2.  Define a class that inherits from `SearchAlgorithm` (from `search_algorithm.py`).
3.  Implement the `__init__` method if your algorithm requires specific parameters.
4.  Implement the abstract method `next_action(self, state: GameState) -> Tuple`. This method should contain the core logic to determine the best move from the given state.
5.  Import your new algorithm class in `main.py` and add it to the `AVAILABLE_ALGORITHMS` dictionary with a descriptive name.

## Credits

*   The MCTS implementation is based on the tutorial found at: <https://ai-boson.github.io/mcts/>