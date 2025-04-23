import argparse

import chess
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.network import ChessNetwork
from src.training.trainer import RLTrainer
from src.training.experience_buffer import ExperienceBuffer
from src.environment.chess_env import ChessEnv
from src.models.policy import RandomPolicy, NetworkPolicy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepChess Training Script')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=100, help='Number of self-play games per iteration')
    parser.add_argument('--eval-games', type=int, default=20, help='Number of evaluation games')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--plot-stats', action='store_true', help='Plot training statistics')
    parser.add_argument('--residual-blocks', type=int, default=10, help='Number of residual blocks in the network')
    return parser.parse_args()


def plot_training_stats(stats):
    """Plot training statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    iterations = range(1, len(stats['policy_loss']) + 1)
    ax1.plot(iterations, stats['policy_loss'], label='Policy Loss')
    ax1.plot(iterations, stats['value_loss'], label='Value Loss')
    ax1.plot(iterations, stats['total_loss'], label='Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot win rate
    ax2.plot(iterations, stats['win_rate'], 'g-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Win Rate vs Random')
    ax2.grid(True)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('training_stats.png')
    plt.close()


def play_game_against_human(model, device='cpu'):
    """Let a human play against the trained model."""
    env = ChessEnv()
    board = env.board
    policy = NetworkPolicy(model, device=device, temperature=0.5, exploration_factor=0.0)

    print("\nSpiel gegen das trainierte Modell")
    print("Beende mit Ctrl+C")

    try:
        while not board.is_game_over():
            print("\n" + str(board))

            if board.turn == chess.WHITE:  # Menschlicher Spieler (Weiß)
                legal_moves = list(board.legal_moves)
                print("Gültige Züge:", [move.uci() for move in legal_moves])

                while True:
                    try:
                        move_uci = input("Dein Zug (UCI-Format, z.B. 'e2e4'): ")
                        move = chess.Move.from_uci(move_uci)
                        if move in legal_moves:
                            break
                        else:
                            print("Ungültiger Zug. Versuche es erneut.")
                    except ValueError:
                        print("Ungültiges Format. Benutze UCI-Format (z.B. 'e2e4').")
            else:  # KI-Spieler (Schwarz)
                print("KI denkt...")
                move = policy.get_action(board)
                print(f"KI-Zug: {move.uci()}")

            board.push(move)

        # Spielergebnis
        print("\nSpiel beendet!")
        print(board)
        if board.is_checkmate():
            winner = "Weiß (Du)" if board.turn == chess.BLACK else "Schwarz (KI)"
            print(f"Schachmatt! {winner} hat gewonnen.")
        elif board.is_stalemate():
            print("Patt! Das Spiel endet unentschieden.")
        elif board.is_insufficient_material():
            print("Unzureichendes Material! Das Spiel endet unentschieden.")
        else:
            print("Remis durch Regel (50-Züge-Regel oder Zugwiederholung).")

    except KeyboardInterrupt:
        print("\nSpiel abgebrochen.")


def main():
    """Main training function."""
    args = parse_args()

    # Stelle sicher, dass das checkpoint-Verzeichnis existiert
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Modell erstellen
    print(f"Erstelle ChessNetwork mit {args.residual_blocks} Residual-Blöcken...")
    # Berechne die Ausgabegröße für den Aktionsraum (64*64 normale Züge + 64*64*4 Umwandlungszüge)
    policy_output_size = 64 * 64 + 64 * 64 * 4
    model = ChessNetwork(
        input_channels=14,
        num_res_blocks=args.residual_blocks,
        num_filters=128,
        policy_output_size=policy_output_size
    )
    model.to(args.device)

    # Modell laden, falls angegeben
    if args.load_model and os.path.exists(args.load_model):
        print(f"Lade vortrainiertes Modell aus {args.load_model}...")
        checkpoint = torch.load(args.load_model, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Trainer erstellen
    trainer = RLTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        num_self_play_games=args.games,
        evaluation_games=args.eval_games
    )

    # Statistik-Tracking
    stats = {
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'win_rate': []
    }

    # Haupttrainingschleife
    print(f"Starte Training für {args.iterations} Iterationen...")
    for i in range(args.iterations):
        print(f"\nIteration {i + 1}/{args.iterations}")
        start_time = time.time()

        # Training
        train_stats = trainer.train_iteration()

        # Statistiken speichern
        stats['policy_loss'].append(train_stats['policy_loss'])
        stats['value_loss'].append(train_stats['value_loss'])
        stats['total_loss'].append(train_stats['train_loss'])

        # Evaluation gegen zufällige Richtlinie
        print("Evaluiere gegen zufällige Richtlinie...")
        eval_stats = trainer.evaluate()
        stats['win_rate'].append(eval_stats['win_rate'])

        # Speichere Modell-Checkpoint (KORRIGIERT: Nur Dateiname übergeben)
        checkpoint_filename = f"model_iter_{i + 1}.pt"
        trainer.save_model(checkpoint_filename)

        # Zeige Fortschritt
        elapsed = time.time() - start_time
        print(f"Iteration {i + 1} abgeschlossen in {elapsed:.2f}s")
        print(f"Verluste: Policy={train_stats['policy_loss']:.4f}, Value={train_stats['value_loss']:.4f}")
        print(f"Evaluation: Siege={eval_stats['wins']}, Niederlagen={eval_stats['losses']}, "
              f"Remis={eval_stats['draws']}, Gewinnrate={eval_stats['win_rate']:.4f}")

    # Speichere finales Modell
    trainer.save_model("final_model.pt")

    # Plotte Statistiken, falls gewünscht
    if args.plot_stats:
        print("Plotte Trainingsstatistiken...")
        plot_training_stats(stats)

    print("Training abgeschlossen!")

    # Optional: Spiel gegen den Menschen
    play_against_human = input("Möchtest du gegen das trainierte Modell spielen? (j/n): ").lower() == 'j'
    if play_against_human:
        play_game_against_human(model, device=args.device)


if __name__ == "__main__":
    main()