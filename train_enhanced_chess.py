#!/usr/bin/env python
"""
Training script for the enhanced chess AI model.
This script provides a simple way to launch training with recommended parameters.
"""

import os
import sys
import subprocess
import argparse


def parse_args():
    """Parse command line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description='Train enhanced chess AI')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of training iterations')
    parser.add_argument('--games', type=int, default=500,
                        help='Number of self-play games per iteration')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='Number of evaluation games')

    # Network parameters
    parser.add_argument('--residual-blocks', type=int, default=20,
                        help='Number of residual blocks in network')
    parser.add_argument('--filters', type=int, default=256,
                        help='Number of filters in convolutional layers')

    # Optimization parameters
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--l2', type=float, default=2e-4,
                        help='L2 regularization')
    parser.add_argument('--value-weight', type=float, default=3.0,
                        help='Weight for value loss')

    # Checkpoint parameters
    parser.add_argument('--checkpoint-dir', type=str, default='./enhanced_checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--save-freq', type=int, default=2,
                        help='Save frequency (in iterations)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to model to resume training from')

    # GPU parameters
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')

    # Output parameters
    parser.add_argument('--plot-stats', action='store_true',
                        help='Create plots of training statistics')

    # Training configurations
    parser.add_argument('--quick-test', action='store_true',
                        help='Run a quick test with minimal iterations and games')
    parser.add_argument('--strong-model', action='store_true',
                        help='Use stronger configuration (more blocks, filters, games)')

    return parser.parse_args()


def setup_environment():
    """Create necessary directories."""
    os.makedirs('enhanced_checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)


def set_gpu_environment(gpu_id):
    """Set CUDA environment variables."""
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU {gpu_id}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        print("Using CPU for training")


def build_command(args):
    """Build the command to run the enhanced training script."""
    # Start with base command
    cmd = ["python", "src/main_enhanced.py"]

    # Override with quick test parameters if requested
    if args.quick_test:
        args.iterations = 3
        args.games = 50
        args.eval_games = 20
        args.batch_size = 256
        print("Running in quick test mode with reduced parameters")

    # Override with strong model parameters if requested
    if args.strong_model:
        args.residual_blocks = 30
        args.filters = 384
        args.games = 1000
        print("Running with stronger model configuration")

    # Add all parameters
    cmd.extend(["--iterations", str(args.iterations)])
    cmd.extend(["--games", str(args.games)])
    cmd.extend(["--eval-games", str(args.eval_games)])
    cmd.extend(["--residual-blocks", str(args.residual_blocks)])
    cmd.extend(["--filters", str(args.filters)])
    cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(["--lr", str(args.lr)])
    cmd.extend(["--l2", str(args.l2)])
    cmd.extend(["--value-weight", str(args.value_weight)])
    cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
    cmd.extend(["--save-freq", str(args.save_freq)])

    # Add optional parameters
    if args.load_model:
        cmd.extend(["--load-model", args.load_model])
    if args.plot_stats:
        cmd.append("--plot-stats")

    return cmd


def main():
    """Main function to set up and start training."""
    args = parse_args()

    # Setup environment
    setup_environment()
    set_gpu_environment(args.gpu)

    # Build command
    cmd = build_command(args)

    # Print command
    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")

    # Run command
    try:
        print("\n" + "=" * 60)
        print("Starting enhanced chess AI training...")
        print("=" * 60 + "\n")

        process = subprocess.run(cmd)

        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print(f"Training failed with return code {process.returncode}")
            print("=" * 60)
            return 1

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("You can resume training using --load-model enhanced_checkpoints/latest_model.pt")
        print("=" * 60)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())