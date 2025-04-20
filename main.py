#!/usr/bin/env python3
"""Main entry point for playing Tic-Tac-Toe or Sudo Tic-Tac-Toe with MCTS AI."""

import sys
import random
import numpy as np
import math
from typing import Type, TypeVar, Dict, List, Tuple

# Updated Project imports
from games.game_state import GameState          # Base class for games
from games.tic_tac_toe import TicTacToeState     # TicTacToe game
from games.sudo_tic_tac_toe import SudoTicTacToeState # Sudo game

from algorithms.search_algorithm import SearchAlgorithm # Base class for search
from algorithms.mcts import MCTSAlgorithm              # MCTS implementation
from algorithms.minimax_search import MinimaxSearch # Updated import path

# SIMULATIONS_PER_MOVE = 400 # REMOVED - Now defined per game state class

# --- Game Registry --- #
# Uses imported classes from the games package
AVAILABLE_GAMES: Dict[str, Type[GameState]] = {
    TicTacToeState.game_title: TicTacToeState,
    SudoTicTacToeState.game_title: SudoTicTacToeState
}

# --- Algorithm Registry --- #
# Uses imported classes from the algorithms package
AVAILABLE_ALGORITHMS: Dict[str, Type[SearchAlgorithm]] = {
    "MCTS": MCTSAlgorithm,
    "Minimax": MinimaxSearch # Value is already correct
}

def _prompt_human_move(state: GameState) -> Tuple:
    """Ask the user for a move until a valid one is entered for the given game state."""
    print(state.get_action_prompt())
    while True:
        try:
            input_str = input("Your move> ").strip()
            action = state.parse_action(input_str)
            available_actions = state.available_actions()
            if action in available_actions:
                return action
            else:
                print(f"Illegal move. Valid moves are: {available_actions}")
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

def select_algorithm() -> Type[SearchAlgorithm]:
    """Prompts the user to select an AI search algorithm."""
    print("Select AI Algorithm:")
    algo_options: List[Type[SearchAlgorithm]] = list(AVAILABLE_ALGORITHMS.values())
    algo_names: List[str] = list(AVAILABLE_ALGORITHMS.keys())
    for i, name in enumerate(algo_names):
        print(f"  {i+1}: {name}")

    while True:
        try:
            choice_str = input(f"Enter choice (1-{len(algo_options)}): ").strip()
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(algo_options):
                return algo_options[choice_idx]
            else:
                print(f"Invalid choice, please enter a number between 1 and {len(algo_options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
            print("\nExiting setup.")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nExiting setup.")
            sys.exit(0)

def play_cli() -> None:
    """Main CLI loop for playing a selected game against a selected AI algorithm."""

    GameStateClass = select_game()
    game_name = GameStateClass.game_title
    AlgorithmClass = select_algorithm()
    algo_name = next(name for name, cls in AVAILABLE_ALGORITHMS.items() if cls is AlgorithmClass)

    print(f"\nWelcome to {game_name} playing against {algo_name} AI!")

    # Optionally configure algorithm parameters (e.g., depth for BruteForce)
    algo_params = {}
    if AlgorithmClass is MCTSAlgorithm:
        # MCTSAlgorithm will use the simulations_per_move from the GameState class by default if not passed
        # We pass it explicitly here based on the selected game
        algo_params['simulations_per_move'] = GameStateClass.simulations_per_move
    elif AlgorithmClass is MinimaxSearch: # Updated check
        try:
            depth_str = input("Enter max search depth for Minimax (e.g., 4, or leave blank for default): ").strip()
            if depth_str:
                algo_params['max_depth'] = int(depth_str)
        except ValueError:
            print("Invalid depth, using default.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting setup.")
            return

    print("Do you want to be X (player 1) and start? [y/N] ", end="")
    try:
        human_is_x = input().strip().lower().startswith("y")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting setup.")
        return

    state: GameState = GameStateClass(current_player=1) # type: ignore[call-arg]
    search_algorithm: SearchAlgorithm = AlgorithmClass(**algo_params)

    while True:
        print("\nCurrent board state:")
        print(state)

        if state.is_game_over():
            result: float = state.game_result()
            winner_str = "Unknown"
            if math.isclose(result, 1.0):
                winner_str = "X (Player 1)"
            elif math.isclose(result, -1.0):
                winner_str = "O (Player 2)"
            elif math.isclose(result, 0.0):
                winner_str = "Draw"
            print(f"\n===== GAME OVER =====")
            print(f"Result: {winner_str}")
            return

        current_player_name = "X" if state.current_player == 1 else "O"
        is_human_turn = (state.current_player == 1 and human_is_x) or \
                          (state.current_player == -1 and not human_is_x)

        action: Tuple
        if is_human_turn:
            print(f"\nPlayer {current_player_name}'s turn (Human).")
            action = _prompt_human_move(state)
        else:
            # AI Turn using the selected search algorithm
            print(f"\nPlayer {current_player_name}'s turn (AI - {algo_name}).")
            print("AI is thinking ...", file=sys.stderr)
            try:
                action = search_algorithm.next_action(state)
                print(f"AI plays {state.action_to_string(action)}")
            except (RuntimeError, ValueError) as e:
                 print(f"\nError during AI move: {e}")
                 print("This might happen if the game state has no legal moves or search failed.")
                 break
            except Exception as e:
                 print(f"\nAn unexpected error occurred during AI search: {e}")
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