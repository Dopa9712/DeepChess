import os
import sys
import chess
import torch
import time
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.network import ChessNetwork
from src.models.policy import RandomPolicy, NetworkPolicy
from src.environment.chess_env import ChessEnv


def evaluate_model(model_path, num_games=100, device='cuda' if torch.cuda.is_available() else 'cpu', visualize=True):
    """
    Evaluate the model against random moves.

    Args:
        model_path: Path to the model file
        num_games: Number of games to play
        device: Device to use for model
        visualize: Whether to create visualization of results

    Returns:
        dict: Dictionary with game statistics
    """
    print(f"\nEvaluating model: {model_path}")
    print(f"Playing {num_games} games against random moves...")

    # Load the model
    policy_output_size = 64 * 64 + 64 * 64 * 4
    model = ChessNetwork(
        input_channels=14,
        num_res_blocks=10,
        num_filters=128,
        policy_output_size=policy_output_size
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Create policies
    # VERBESSERT: Niedrigere Temperatur (0.1) und keine Exploration für bessere Spielleistung
    model_policy = NetworkPolicy(model, device=device, temperature=0.1, exploration_factor=0.0)
    random_policy = RandomPolicy()

    # Statistics
    stats = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'model_as_white_wins': 0,
        'model_as_black_wins': 0,
        'white_wins': 0,
        'black_wins': 0,
        'checkmate_victories': 0,
        'game_lengths': [],
        'results': [],  # List of 1 (win), 0 (draw), -1 (loss)
        'material_balance_history': [],  # Material balance at the end of each game
        'timeout_games': 0  # Games that reached move limit
    }

    # Play games
    for game_idx in tqdm(range(num_games), desc="Playing games"):
        env = ChessEnv()
        board = env.board
        move_count = 0
        max_moves = 200  # Limit to avoid infinite games

        # Randomly assign colors (50% chance for model to play white)
        model_plays_white = game_idx % 2 == 0

        # Track material balance throughout the game
        game_material_history = []

        while not board.is_game_over() and move_count < max_moves:
            # Calculate material balance
            material_balance = 0
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                material_balance += len(board.pieces(piece_type, chess.WHITE))
                material_balance -= len(board.pieces(piece_type, chess.BLACK))
            # From model's perspective
            if not model_plays_white:
                material_balance = -material_balance
            game_material_history.append(material_balance)

            # Determine which policy to use
            is_model_turn = (board.turn == chess.WHITE) == model_plays_white
            policy_to_use = model_policy if is_model_turn else random_policy

            # Choose and make move
            move = policy_to_use.get_action(board)
            board.push(move)
            move_count += 1

        # Record game length
        stats['game_lengths'].append(move_count)

        # Record final material balance
        if game_material_history:  # Ensure we have at least one value
            stats['material_balance_history'].append(game_material_history[-1])

        # Check if game reached move limit
        if move_count >= max_moves:
            stats['timeout_games'] += 1

        # Determine game result
        if board.is_checkmate():
            winner_is_white = not board.turn

            if (winner_is_white and model_plays_white) or (not winner_is_white and not model_plays_white):
                stats['wins'] += 1
                stats['results'].append(1)
                stats['checkmate_victories'] += 1

                if model_plays_white:
                    stats['model_as_white_wins'] += 1
                    stats['white_wins'] += 1
                else:
                    stats['model_as_black_wins'] += 1
                    stats['black_wins'] += 1
            else:
                stats['losses'] += 1
                stats['results'].append(-1)

                if not model_plays_white:
                    stats['white_wins'] += 1
                else:
                    stats['black_wins'] += 1
        else:
            # Draw or move limit reached
            stats['draws'] += 1
            stats['results'].append(0)

    # Calculate additional statistics
    stats['win_rate'] = stats['wins'] / num_games * 100
    stats['draw_rate'] = stats['draws'] / num_games * 100
    stats['loss_rate'] = stats['losses'] / num_games * 100
    stats['avg_game_length'] = sum(stats['game_lengths']) / len(stats['game_lengths'])
    stats['white_win_rate'] = stats['white_wins'] / num_games * 100
    stats['black_win_rate'] = stats['black_wins'] / num_games * 100
    stats['model_as_white_win_rate'] = (stats['model_as_white_wins'] / (num_games / 2)) * 100 if num_games > 0 else 0
    stats['model_as_black_win_rate'] = (stats['model_as_black_wins'] / (num_games / 2)) * 100 if num_games > 0 else 0
    stats['avg_material_balance'] = sum(stats['material_balance_history']) / len(stats['material_balance_history']) if \
    stats['material_balance_history'] else 0

    # Print statistics
    print("\nEvaluation Results:")
    print(f"Total games: {num_games}")
    print(f"Wins: {stats['wins']} ({stats['win_rate']:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draw_rate']:.1f}%)")
    print(f"Losses: {stats['losses']} ({stats['loss_rate']:.1f}%)")
    print(f"Average game length: {stats['avg_game_length']:.1f} moves")
    print(f"White wins: {stats['white_wins']} ({stats['white_win_rate']:.1f}%)")
    print(f"Black wins: {stats['black_wins']} ({stats['black_win_rate']:.1f}%)")
    print(f"Model as White win rate: {stats['model_as_white_win_rate']:.1f}%")
    print(f"Model as Black win rate: {stats['model_as_black_win_rate']:.1f}%")
    print(f"Checkmate victories: {stats['checkmate_victories']}")
    print(f"Average material balance: {stats['avg_material_balance']:.2f}")
    print(f"Games reaching move limit: {stats['timeout_games']} ({stats['timeout_games'] / num_games * 100:.1f}%)")

    # Create visualization if requested
    if visualize:
        create_visualization(stats, num_games)

    return stats


def create_visualization(stats, num_games):
    """Create visualizations for the evaluation results."""
    plt.figure(figsize=(15, 12))

    # Plot 1: Overall results
    plt.subplot(2, 2, 1)
    labels = ['Wins', 'Draws', 'Losses']
    sizes = [stats['wins'], stats['draws'], stats['losses']]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Overall Results')

    # Plot 2: Game lengths histogram
    plt.subplot(2, 2, 2)
    plt.hist(stats['game_lengths'], bins=20, color='#2196F3')
    plt.xlabel('Game Length (moves)')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Game Lengths')

    # Plot 3: White vs Black wins
    plt.subplot(2, 2, 3)
    labels = ['White Wins', 'Black Wins', 'Draws']
    sizes = [stats['white_wins'], stats['black_wins'], stats['draws']]
    colors = ['#E0E0E0', '#424242', '#9E9E9E']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('White vs Black Win Rate')

    # Plot 4: Model as White vs Black performance
    plt.subplot(2, 2, 4)
    labels = ['As White', 'As Black']
    win_rates = [stats['model_as_white_win_rate'], stats['model_as_black_win_rate']]
    plt.bar(labels, win_rates, color=['#64B5F6', '#7986CB'])
    plt.ylabel('Win Rate (%)')
    plt.title('Model Performance by Color')
    plt.ylim(0, 100)

    # Save the figure
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("Visualization saved as 'evaluation_results.png'")

    # Additional visualization: Material balance histogram
    plt.figure(figsize=(10, 6))
    plt.hist(stats['material_balance_history'], bins=15, color='#4CAF50', alpha=0.8)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=stats['avg_material_balance'], color='red', linestyle='-',
                label=f'Avg: {stats["avg_material_balance"]:.2f}')
    plt.xlabel('Material Balance (positive = model advantage)')
    plt.ylabel('Frequency')
    plt.title('Material Balance Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('material_balance.png')
    print("Material balance visualization saved as 'material_balance.png'")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Evaluate chess model against random moves')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to the model file')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')

    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    start_time = time.time()
    evaluate_model(
        model_path=model_path,
        num_games=args.games,
        device=args.device,
        visualize=not args.no_viz
    )
    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()