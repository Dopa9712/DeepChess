import os
import sys
import chess
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.network import ChessNetwork
from src.models.policy import NetworkPolicy
from src.environment.utils import board_to_planes
from src.environment.chess_env import ChessEnv


def debug_model_output(model_path, device='cpu'):
    """Debug the model's outputs for some simple positions"""
    print(f"Debugging model from {model_path}...")

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
    model.eval()

    # Create a policy
    policy = NetworkPolicy(model, device=device, temperature=1.0, exploration_factor=0.0)

    # Set up a few simple positions to test
    positions = [
        # Starting position
        chess.Board(),

        # Simple e4 opening
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),

        # Position with obvious best move (checkmate in 1)
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"),

        # Position with a free queen capture
        chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 3"),
    ]

    for i, board in enumerate(positions):
        print(f"\nPosition {i + 1}:")
        print(board)

        # Get legal moves for this position
        legal_moves = list(board.legal_moves)
        print(f"Legal moves: {[move.uci() for move in legal_moves]}")

        # Get the model's policy output directly
        board_planes = board_to_planes(board)
        board_tensor = torch.FloatTensor(board_planes).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = model(board_tensor)
            policy_logits = policy_logits.cpu().numpy()[0]
            value = value.item()

        # Get the move probabilities from the policy
        move_probs, _ = policy.get_action_probs(board)

        # Print the top 5 moves according to the model
        top_moves_indices = np.argsort(-move_probs)[:5]  # Get indices of top 5 highest probabilities

        print(f"Board evaluation: {value:.4f} (-1.0 to 1.0, higher is better for current player)")
        print("Top 5 moves according to model:")
        for idx in top_moves_indices:
            move = legal_moves[idx]
            prob = move_probs[idx]
            print(f"  {move.uci()} with probability {prob:.4f}")

        # Now let's see what move the policy would actually make
        selected_move = policy.get_action(board)
        print(f"Selected move: {selected_move.uci()}")

        # Check if the model is choosing legal and reasonable moves
        if selected_move not in legal_moves:
            print("ERROR: Selected move is not legal!")

        # Try making the move and see what happens
        board.push(selected_move)
        if board.is_checkmate():
            print("This move leads to checkmate!")
        elif board.is_check():
            print("This move gives check!")
        elif board.is_stalemate():
            print("This move results in stalemate!")

    print("\nDebug complete. Check if the model's move selection makes sense.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug chess model's outputs")
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to the model file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    debug_model_output(model_path, args.device)