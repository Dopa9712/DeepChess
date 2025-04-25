import argparse
import chess
import torch
import os
import time
import numpy as np
from tqdm import tqdm

from src.models.enhanced_network import EnhancedChessNetwork
from src.training.enhanced_trainer import EnhancedRLTrainer, EnhancedSelfPlayWorker
from src.environment.chess_env import ChessEnv
from src.models.policy import RandomPolicy, NetworkPolicy
from src.utils.visual_eval import create_training_plots


def setup_gpu():
    """
    Configure GPU settings for optimal training.

    Returns:
        str: Device to use for training
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Enable cuDNN benchmark for optimized performance
        torch.backends.cudnn.benchmark = True

        # Clear GPU cache before training
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print("GPU not available, using CPU")

    return device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced DeepChess Training Script')
    parser.add_argument('--iterations', type=int, default=50, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=1000, help='Number of self-play games per iteration')
    parser.add_argument('--eval-games', type=int, default=200, help='Number of evaluation games')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--l2', type=float, default=2e-4, help='L2 regularization strength')
    parser.add_argument('--checkpoint-dir', type=str, default='./enhanced_checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--residual-blocks', type=int, default=20, help='Number of residual blocks in the network')
    parser.add_argument('--filters', type=int, default=256, help='Number of filters in convolutional layers')
    parser.add_argument('--value-weight', type=float, default=3.0, help='Weight for value loss in total loss')
    parser.add_argument('--plot-stats', action='store_true', help='Create plots of training statistics')
    parser.add_argument('--save-freq', type=int, default=1, help='Frequency of saving checkpoints (iterations)')
    return parser.parse_args()


def main():
    """Main enhanced training function."""
    args = parse_args()

    # Set up device
    device = args.device if args.device else setup_gpu()

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print("\n=== Enhanced DeepChess Training ===")
    print(f"Training for {args.iterations} iterations with {args.games} self-play games per iteration")
    print(f"Network: {args.residual_blocks} residual blocks, {args.filters} filters")
    print(f"Training: LR={args.lr}, L2={args.l2}, batch size={args.batch_size}, epochs={args.epochs}")
    print(f"Value loss weight: {args.value_weight}")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    print("=" * 40)

    # Create model
    print(f"\nCreating EnhancedChessNetwork with {args.residual_blocks} residual blocks and {args.filters} filters...")

    # Calculate policy output size for action space (64*64 normal moves + 64*64*4 promotion moves)
    policy_output_size = 64 * 64 + 64 * 64 * 4

    model = EnhancedChessNetwork(
        input_channels=14,
        num_res_blocks=args.residual_blocks,
        num_filters=args.filters,
        policy_output_size=policy_output_size,
        fc_size=512
    )
    model.to(device)

    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")

    # Load model if specified
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model from {args.load_model}...")
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model successfully loaded!")

        # Check if training history exists in checkpoint
        training_history = checkpoint.get('training_history', None)
        best_win_rate = checkpoint.get('best_win_rate', 0.0)
        print(f"Loaded best win rate: {best_win_rate:.4f}")
    else:
        # Proper weight initialization
        print("Initializing model weights...")
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    # Create trainer
    trainer = EnhancedRLTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        l2_reg=args.l2,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        num_self_play_games=args.games,
        evaluation_games=args.eval_games,
        value_loss_weight=args.value_weight
    )

    # Main training loop
    print(f"\nStarting enhanced training for {args.iterations} iterations...")
    total_start_time = time.time()

    for iteration in range(args.iterations):
        print(f"\n{'=' * 50}")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"{'=' * 50}")

        start_time = time.time()

        # Training
        train_stats = trainer.train_iteration()

        # Evaluation against multiple opponents
        print("\nEvaluating against multiple opponents...")
        eval_stats = trainer.evaluate_against_multiple_opponents()

        # Save model checkpoint based on save frequency
        if (iteration + 1) % args.save_freq == 0:
            checkpoint_filename = f"model_iter_{iteration + 1}.pt"
            trainer.save_model(checkpoint_filename)
            print(f"Checkpoint saved as {checkpoint_filename}")

        # Always save as latest model
        trainer.save_model("latest_model.pt")

        # Show progress
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nIteration {iteration + 1} completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(
            f"Losses: Policy={train_stats['policy_loss']:.4f}, Value={train_stats['value_loss']:.4f}, Total={train_stats['train_loss']:.4f}")
        print(f"Evaluation: Win rate={eval_stats['overall_win_rate']:.4f}, Checkmates={eval_stats['total_checkmates']}")
        print(f"Current learning rate: {train_stats['lr']:.6f}")

        # Create visualization periodically or if explicitly requested
        if args.plot_stats or (iteration + 1) % 5 == 0:
            create_training_plots(trainer)

        # Estimate remaining time
        if iteration < args.iterations - 1:
            avg_iter_time = (time.time() - total_start_time) / (iteration + 1)
            remaining_time = avg_iter_time * (args.iterations - (iteration + 1))
            remaining_hours, remainder = divmod(remaining_time, 3600)
            remaining_minutes, _ = divmod(remainder, 60)
            print(f"Estimated remaining time: {int(remaining_hours)}h {int(remaining_minutes)}m")

    # Calculate total time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Save final model
    trainer.save_model("final_model.pt")

    print(f"\n{'=' * 50}")
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s!")
    print(f"Best win rate: {trainer.best_model_win_rate:.4f}")
    print(f"Final models saved to {args.checkpoint_dir}")
    print(f"{'=' * 50}")

    # Final visualization
    create_training_plots(trainer)

    # Optional: Play a sample game
    play_sample_game = input("\nWould you like to see the trained model play a sample game? (y/n): ").lower() == 'y'
    if play_sample_game:
        print("\nPlaying sample game with final model vs. random opponent...")
        play_and_display_sample_game(model, device)


def play_and_display_sample_game(model, device, max_moves=100):
    """
    Play and display a sample game between the model and a random opponent.

    Args:
        model: The trained model
        device: Device to run the model on
        max_moves: Maximum number of moves before terminating
    """
    # Set up environment and policies
    env = ChessEnv()
    board = env.board

    model_policy = NetworkPolicy(model, device=device, temperature=0.1, exploration_factor=0.0)
    random_policy = RandomPolicy()

    # Model plays as white
    model_plays_white = True

    # Play the game
    move_count = 0
    print("\nInitial position:")
    print(board)
    print()

    while not board.is_game_over() and move_count < max_moves:
        # Determine whose turn it is
        is_model_turn = (board.turn == chess.WHITE) == model_plays_white
        current_player = "Model" if is_model_turn else "Random"

        # Get and make move
        policy = model_policy if is_model_turn else random_policy
        move = policy.get_action(board)

        print(f"Move {move_count + 1}: {current_player} plays {move.uci()}")
        board.push(move)
        move_count += 1

        # Display board every few moves
        if move_count % 5 == 0:
            print(f"\nPosition after {move_count} moves:")
            print(board)
            print()

    # Show final position
    print("\nFinal position:")
    print(board)

    # Determine result
    result = "1-0" if board.is_checkmate() and not board.turn else "0-1" if board.is_checkmate() else "1/2-1/2"
    reason = ""

    if board.is_checkmate():
        reason = "checkmate"
    elif board.is_stalemate():
        reason = "stalemate"
    elif board.is_insufficient_material():
        reason = "insufficient material"
    elif board.is_fifty_moves():
        reason = "fifty-move rule"
    elif board.is_repetition():
        reason = "threefold repetition"
    elif move_count >= max_moves:
        reason = "move limit reached"

    winner = "Model" if (result == "1-0" and model_plays_white) or (
                result == "0-1" and not model_plays_white) else "Random" if result != "1/2-1/2" else "None"

    print(f"\nGame result: {result} ({reason})")
    print(f"Winner: {winner}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()