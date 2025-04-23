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
from src.gui.chess_gui import play_with_gui


def find_model_files(start_dir=None):
    """
    Find all .pt model files in the project by recursively searching directories.

    Args:
        start_dir: Directory to start searching from (defaults to current directory)

    Returns:
        dict: Dictionary mapping display paths to full file paths
    """
    if start_dir is None:
        start_dir = os.getcwd()

    model_files = {}

    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.pt'):
                # Create a relative path for display
                rel_path = os.path.relpath(os.path.join(root, file), start_dir)
                model_files[rel_path] = os.path.join(root, file)

    return model_files


def list_and_select_models():
    """
    List all model files in the project and let the user select one.

    Returns:
        str: Path to the selected model file, or None if no model was selected
    """
    print("\nSearching for model files in the project...")
    model_files = find_model_files()

    if not model_files:
        print("No model files (.pt) found in the project.")
        return None

    # Sort the paths for better display
    sorted_paths = sorted(model_files.keys())

    print("\nFound model files:")
    for i, path in enumerate(sorted_paths):
        print(f"[{i + 1}] {path}")

    while True:
        try:
            choice = input("\nSelect a model number (or 'q' to quit): ")

            if choice.lower() == 'q':
                return None

            choice = int(choice)
            if 1 <= choice <= len(sorted_paths):
                selected_model = model_files[sorted_paths[choice - 1]]
                print(f"Selected model: {sorted_paths[choice - 1]}")
                return selected_model
            else:
                print(f"Please select a number between 1 and {len(sorted_paths)}")
        except ValueError:
            print("Please enter a valid number or 'q'")


def main():
    """Main function for playing against a trained model."""
    parser = argparse.ArgumentParser(description='Play chess against a trained model')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model file to use (if not selecting interactively)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--search', action='store_true',
                        help='Search for model files in the entire project')

    args = parser.parse_args()

    model_path = None

    # If a specific model was provided, use that
    if args.model:
        if os.path.exists(args.model):
            model_path = args.model
        else:
            print(f"Model file not found: {args.model}")
            # Fall back to interactive selection
            model_path = list_and_select_models() if args.search else None
    else:
        # Interactive model selection
        model_path = list_and_select_models() if args.search else None

    if model_path:
        # Launch the GUI with the selected model
        play_with_gui(model_path=model_path, device=args.device)
    else:
        print("No model selected. Exiting.")


if __name__ == "__main__":
    main()