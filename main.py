#!/usr/bin/env python3
"""Main entry point for playing Tic-Tac-Toe or Sudo Tic-Tac-Toe with MCTS AI."""

import sys
import random
import numpy as np
from typing import Type, TypeVar, Dict, List, Tuple

# Project imports
from game_state import GameState
from mcts import MonteCarloTreeSearchNode
from tic_tac_toe import TicTacToeState
from sudo_tic_tac_toe import SudoTicTacToeState

# SIMULATIONS_PER_MOVE = 400 # REMOVED - Now defined per game state class

# --- Game Registry --- #
# Discovered game state classes are registered here.
# We explicitly import and list them for simplicity, but this could be automated
# using metaclasses or module scanning for larger projects.
AVAILABLE_GAMES: Dict[str, Type[GameState]] = {
    TicTacToeState.game_title: TicTacToeState,
    SudoTicTacToeState.game_title: SudoTicTacToeState
}

def _prompt_human_move(state: GameState) -> Tuple:
    """Ask the user for a move until a valid one is entered for the given game state."""
    print(state.get_action_prompt())
    while True:
        try:
            input_str = input("Your move> ").strip()
            action = state.parse_action(input_str)
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
    """Prompts the user to select a game from the available registry."""
    print("Select game:")
    game_options: List[Type[GameState]] = list(AVAILABLE_GAMES.values())
    for i, game_class in enumerate(game_options):
        print(f"  {i+1}: {game_class.game_title}")
    while True:
        try:
            choice_str = input(f"Enter choice (1-{len(game_options)}): ").strip()
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(game_options):
                return game_options[choice_idx]
            else:
                print(f"Invalid choice, please enter a number between 1 and {len(game_options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
            print("\nExiting setup.")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nExiting setup.")
            sys.exit(0)

def play_cli() -> None:
    """Main CLI loop for playing a selected game against MCTS AI."""

    GameStateClass = select_game()
    game_name = GameStateClass.game_title
    print(f"\nWelcome to {game_name}!")

    print("Do you want to be X (player 1) and start? [y/N] ", end="")
    try:
        human_is_x = input().strip().lower().startswith("y")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting setup.")
        return

    state: GameState = GameStateClass(current_player=1) # type: ignore[call-arg]

    while True:
        print("\nCurrent board state:")
        print(state)

        if state.is_game_over():
            result = state.game_result()
            winner_map = {1: "X (Player 1)", -1: "O (Player 2)", 0: "Draw"}
            print(f"\n===== GAME OVER =====")
            print(f"Result: {winner_map.get(result, 'Unknown')}")
            return

        current_player_name = "X" if state.current_player == 1 else "O"
        is_human_turn = (state.current_player == 1 and human_is_x) or \
                          (state.current_player == -1 and not human_is_x)

        action: Tuple
        if is_human_turn:
            print(f"\nPlayer {current_player_name}'s turn (Human).")
            action = _prompt_human_move(state)
        else:
            print(f"\nPlayer {current_player_name}'s turn (AI).")
            print("AI is thinking ...", file=sys.stderr)
            root = MonteCarloTreeSearchNode(state=state)
            try:
                num_sims = state.simulations_per_move
                action = root.best_action(simulations_number=num_sims)
                print(f"AI plays {state.action_to_string(action)}")
            except (RuntimeError, ValueError) as e:
                 print(f"\nError during AI move: {e}")
                 print("This might happen if the game state has no legal moves.")
                 break

        try:
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
    random.seed()
    np.random.seed()
    play_cli()