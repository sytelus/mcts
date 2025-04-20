#!/usr/bin/env python3
"""Main entry point for playing Tic-Tac-Toe or Sudo Tic-Tac-Toe with MCTS AI."""

import sys
import random
import numpy as np
from typing import Type, TypeVar, Generic # Import Generic

# Project imports
from game_state import GameState, Action as ActionType # ActionType alias
from mcts import MonteCarloTreeSearchNode
from tic_tac_toe import TicTacToeState, TttAction     # Import standard TicTacToe
from sudo_tic_tac_toe import SudoTicTacToeState, SudoAction # Import Sudo TicTacToe

# Generic Action type bound for local use
Action = TypeVar('Action')

SIMULATIONS_PER_MOVE = 400 # MCTS simulation budget

def _prompt_human_move(state: GameState[Action]) -> Action:
    """Ask the user for a move until a valid one is entered for the given game state."""
    print(state.get_action_prompt()) # Use the method from the state
    while True:
        try:
            input_str = input("Your move> ").strip()
            action = state.parse_action(input_str) # Use the state's parser

            # Check if the parsed action is actually legal in the current state
            legal_actions = state.get_legal_actions()
            if action in legal_actions:
                return action
            else:
                print(f"Illegal move. Valid moves are: {legal_actions}")
                print("Please try again.")
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
        except EOFError:
            print("\nExiting game.")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nExiting game.")
            sys.exit(0)

def select_game() -> Type[GameState]:
    """Prompts the user to select which game to play."""
    print("Select game:")
    print("  1: Standard Tic-Tac-Toe")
    print("  2: Sudo Tic-Tac-Toe")
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                return TicTacToeState # type: ignore[return-value]
            elif choice == '2':
                return SudoTicTacToeState # type: ignore[return-value]
            else:
                print("Invalid choice, please enter 1 or 2.")
        except EOFError:
            print("\nExiting setup.")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nExiting setup.")
            sys.exit(0)

def play_cli() -> None:
    """Main CLI loop for playing a selected game against MCTS AI."""

    GameStateClass = select_game() # Type is Type[GameState], but known to be concrete
    game_name = "Tic-Tac-Toe" if GameStateClass is TicTacToeState else "Sudo Tic-Tac-Toe"
    print(f"\nWelcome to {game_name}!")

    print("Do you want to be X (player 1) and start? [y/N] ", end="")
    try:
        human_is_x = input().strip().lower().startswith("y")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting setup.")
        return

    # Initialize the selected game state
    # Suppress the constructor warning as we know GameStateClass is concrete
    state = GameStateClass(current_player=1) # type: ignore[call-arg]

    while True:
        print("\nCurrent board state:")
        print(state) # Uses state.__str__()

        if state.is_game_over():
            result = state.game_result()
            winner_map = {1: "X (Player 1)", -1: "O (Player 2)", 0: "Draw"}
            print(f"\n===== GAME OVER =====")
            print(f"Result: {winner_map.get(result, 'Unknown')}")
            return

        current_player_name = "X" if state.current_player == 1 else "O"
        is_human_turn = (state.current_player == 1 and human_is_x) or \
                          (state.current_player == -1 and not human_is_x)

        if is_human_turn:
            print(f"\nPlayer {current_player_name}'s turn (Human).")
            action = _prompt_human_move(state)
        else:
            print(f"\nPlayer {current_player_name}'s turn (AI).")
            print("AI is thinking ...", file=sys.stderr)
            # Create MCTS root node without explicit generic type - it infers from state
            root = MonteCarloTreeSearchNode(state=state)
            try:
                action = root.best_action(simulations_number=SIMULATIONS_PER_MOVE)
                print(f"AI plays {state.action_to_string(action)}")
            except (RuntimeError, ValueError) as e:
                 print(f"\nError during AI move: {e}")
                 print("This might happen if the game state has no legal moves.")
                 break

        # Apply the chosen action to get the next state
        try:
            # The type of state is preserved across the move
            state = state.move(action)
        except ValueError as e:
            print(f"\nError applying move: {e}")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred during move: {e}")
            break


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed() # Seed for Python's random
    np.random.seed() # Seed for NumPy's random (used by MCTS indirectly via np.argmax)
    play_cli()