import os
import torch
import argparse
import sys
from pathlib import Path

# Make sure the project root is in the Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.network import ChessNetwork
from src.gui.chess_gui import play_with_gui  # This is the correct import


def list_models(checkpoint_dir):
    """
    List all model files in the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing model checkpoints

    Returns:
        list: List of model file paths
    """
    model_files = []

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return model_files

    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            model_files.append(file)

    return model_files


def select_model(checkpoint_dir):
    """
    Let the user select a model from the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing model checkpoints

    Returns:
        str: Path to the selected model file, or None if no model was selected
    """
    model_files = list_models(checkpoint_dir)

    if not model_files:
        print(f"No model files found in {checkpoint_dir}")
        return None

    print("\nAvailable models:")
    for i, model_file in enumerate(model_files):
        print(f"[{i + 1}] {model_file}")

    while True:
        try:
            choice = input("\nSelect a model number (or 'q' to quit): ")

            if choice.lower() == 'q':
                return None

            choice = int(choice)
            if 1 <= choice <= len(model_files):
                selected_model = os.path.join(checkpoint_dir, model_files[choice - 1])
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please select a number between 1 and {len(model_files)}")
        except ValueError:
            print("Please enter a valid number or 'q'")


def main():
    """Main function for playing against a trained model."""
    parser = argparse.ArgumentParser(description='Play chess against a trained model')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model file to use (if not selecting interactively)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    model_path = None

    # If a specific model was provided, use that
    if args.model:
        if os.path.exists(args.model):
            model_path = args.model
        else:
            full_path = os.path.join(args.checkpoint_dir, args.model)
            if os.path.exists(full_path):
                model_path = full_path
            else:
                print(f"Model file not found: {args.model}")
                # Fall back to interactive selection
                model_path = select_model(args.checkpoint_dir)
    else:
        # Interactive model selection
        model_path = select_model(args.checkpoint_dir)

    if model_path:
        # Launch the GUI with the selected model
        play_with_gui(model_path=model_path, device=args.device)
    else:
        print("No model selected. Exiting.")


if __name__ == "__main__":
    main()