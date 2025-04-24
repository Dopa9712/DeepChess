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
    parser.add_argument('--games', type=int, default=200, help='Number of self-play games per iteration')
    parser.add_argument('--eval-games', type=int, default=50, help='Number of evaluation games')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--plot-stats', action='store_true', help='Plot training statistics')
    parser.add_argument('--residual-blocks', type=int, default=10, help='Number of residual blocks in the network')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')
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


def check_parameter_update(model, iteration):
    """
    Überprüft, ob sich die Parameter des Modells seit der letzten Iteration geändert haben.

    Args:
        model: Das zu überprüfende Modell
        iteration: Die aktuelle Iterationsnummer

    Returns:
        bool: True, wenn Parameter aktualisiert wurden, sonst False
    """
    # Beim ersten Aufruf initialisieren
    if not hasattr(check_parameter_update, "last_params"):
        check_parameter_update.last_params = {}
        for name, param in model.named_parameters():
            check_parameter_update.last_params[name] = param.clone().detach().cpu().numpy()
        return True

    # Nur jede zweite Iteration überprüfen, um Zeit zu sparen
    if iteration % 2 != 0:
        return True

    changed_params = 0
    total_params = 0
    max_diff = 0.0

    for name, param in model.named_parameters():
        total_params += 1
        param_data = param.detach().cpu().numpy()

        # Vergleiche mit letztem bekannten Zustand
        if name in check_parameter_update.last_params:
            # Berechne maximale absolute Differenz
            param_diff = np.max(np.abs(param_data - check_parameter_update.last_params[name]))
            max_diff = max(max_diff, param_diff)

            if not np.array_equal(param_data, check_parameter_update.last_params[name]):
                changed_params += 1

        # Aktualisierten Parameter speichern
        check_parameter_update.last_params[name] = param_data

    # Ausgabe der Ergebnisse
    print(f"\nParameterüberprüfung (Iteration {iteration}):")
    print(f"  - Geänderte Parameter: {changed_params}/{total_params} ({changed_params / total_params * 100:.1f}%)")
    print(f"  - Maximale Parameteränderung: {max_diff:.6f}")

    # Warnung, wenn keine Parameter aktualisiert wurden
    if changed_params == 0:
        print("\n⚠️ WARNUNG: Keine Parameter haben sich geändert! Das Modell wird nicht trainiert.")
        return False

    return True

def play_game_against_human(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Let a human play against the trained model."""
    env = ChessEnv()
    board = env.board
    # VERBESSERT: Niedrigere Temperatur für bessere Spielstärke
    policy = NetworkPolicy(model, device=device, temperature=0.1, exploration_factor=0.0)

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
    # Befehlszeilenargumente erweitern
    parser = argparse.ArgumentParser(description='DeepChess Training Script')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=200, help='Number of self-play games per iteration')
    parser.add_argument('--eval-games', type=int, default=50, help='Number of evaluation games')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--plot-stats', action='store_true', help='Plot training statistics')
    parser.add_argument('--residual-blocks', type=int, default=10, help='Number of residual blocks in the network')
    # Neue Argumente
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')

    args = parse_args()

    # VERBESSERT: GPU-Optimierungen
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Nutze Benchmarking für optimierte GPU-Nutzung
        torch.backends.cudnn.benchmark = True
        # Reduziere Speicherverbrauch, wenn möglich
        torch.cuda.empty_cache()
    else:
        print("Using CPU for training")

    # Stelle sicher, dass das checkpoint-Verzeichnis existiert
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Finde den neuesten Checkpoint, wenn --resume angegeben ist
    start_iteration = 0
    latest_checkpoint = None

    if args.resume:
        checkpoint_files = [f for f in os.listdir(args.checkpoint_dir)
                            if f.startswith("model_iter_") and f.endswith(".pt")]
        if checkpoint_files:
            # Sortiere nach Iterationsnummer
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_checkpoint = os.path.join(args.checkpoint_dir, checkpoint_files[-1])
            start_iteration = int(checkpoint_files[-1].split("_")[-1].split(".")[0])
            print(f"Neuester Checkpoint gefunden: {latest_checkpoint} (Iteration {start_iteration})")

    # Override, wenn --resume-from angegeben ist
    if args.resume_from:
        latest_checkpoint = args.resume_from
        # Versuche, Iterationsnummer aus dem Dateinamen zu extrahieren
        try:
            iter_part = os.path.basename(latest_checkpoint).split("_")[-1].split(".")[0]
            if iter_part.isdigit():
                start_iteration = int(iter_part)
        except:
            start_iteration = 0
        print(f"Fortsetzen von angegebenem Checkpoint: {latest_checkpoint}")

    # Verwende --load-model, wenn kein Fortsetzungs-Checkpoint gefunden wurde
    if not latest_checkpoint and args.load_model:
        latest_checkpoint = args.load_model
        print(f"Verwende vortrainiertes Modell als Ausgangspunkt: {latest_checkpoint}")

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

    # Modell laden, falls angegeben
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        print(f"Lade Modell aus {latest_checkpoint}...")
        trainer.load_model(latest_checkpoint)
        print(f"Starte Training bei Iteration {start_iteration + 1}")
    else:
        # VERBESSERT: Gewichtinitialisierung für bessere Konvergenz
        print("Initialisiere Modellgewichte...")
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

    # Statistik-Tracking
    stats = {
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'win_rate': [],
        'checkmate_wins': [],  # VERBESSERT: Zusätzliche Statistiken
        'avg_material': []  # VERBESSERT: Zusätzliche Statistiken
    }

    # Haupttrainingschleife
    print(f"Starte Training für {args.iterations} Iterationen...")
    total_start_time = time.time()

    for i in range(start_iteration, args.iterations):
        print(f"\n{'=' * 50}")
        print(f"Iteration {i + 1}/{args.iterations}")
        print(f"{'=' * 50}")
        start_time = time.time()

        # Training
        train_stats = trainer.train_iteration()

        # Überprüfe, ob die Parameter aktualisiert wurden
        if not check_parameter_update(model, i + 1):
            print("Versuche, das Training mit angepassten Parametern fortzusetzen...")
            # Hier könnten Anpassungen vorgenommen werden, z.B. Lernrate erhöhen

        # Statistiken speichern
        stats['policy_loss'].append(train_stats['policy_loss'])
        stats['value_loss'].append(train_stats['value_loss'])
        stats['total_loss'].append(train_stats['train_loss'])

        # Evaluation gegen zufällige Richtlinie
        print("Evaluiere gegen zufällige Richtlinie...")
        eval_stats = trainer.evaluate()
        stats['win_rate'].append(eval_stats['win_rate'])

        # VERBESSERT: Zusätzliche Statistiken speichern
        stats['checkmate_wins'].append(eval_stats.get('checkmate_wins', 0))
        stats['avg_material'].append(eval_stats.get('avg_material', 0))

        # Speichere Modell-Checkpoint
        checkpoint_filename = f"model_iter_{i + 1}.pt"
        trainer.save_model(checkpoint_filename)

        # Zeige Fortschritt
        elapsed = time.time() - start_time
        print(f"Iteration {i + 1} abgeschlossen in {elapsed:.2f}s")
        print(f"Verluste: Policy={train_stats['policy_loss']:.4f}, Value={train_stats['value_loss']:.4f}")
        print(f"Evaluation: Siege={eval_stats['wins']}, Niederlagen={eval_stats['losses']}, "
              f"Remis={eval_stats['draws']}, Gewinnrate={eval_stats['win_rate']:.4f}")
        print(f"Schachmatt-Siege: {eval_stats.get('checkmate_wins', 0)}")
        print(f"Durchschn. Materialvorteil: {eval_stats.get('avg_material', 0):.2f}")

        # VERBESSERT: Geschätzte verbleibende Zeit anzeigen
        if i < args.iterations - 1:
            avg_iter_time = (time.time() - total_start_time) / (i + 1 - start_iteration)
            remaining_time = avg_iter_time * (args.iterations - (i + 1))
            remaining_hours = int(remaining_time // 3600)
            remaining_minutes = int((remaining_time % 3600) // 60)
            print(f"Geschätzte verbleibende Zeit: {remaining_hours}h {remaining_minutes}m")

    # Gesamtzeit berechnen
    total_time = time.time() - total_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    # Speichere finales Modell
    trainer.save_model("final_model.pt")

    print(f"\n{'=' * 50}")
    print(f"Training abgeschlossen in {hours}h {minutes}m {seconds}s!")
    print(f"Beste Gewinnrate: {trainer.best_model_win_rate:.4f}")
    print(f"{'=' * 50}")

    # Plotte Statistiken, falls gewünscht
    if args.plot_stats:
        print("Plotte Trainingsstatistiken...")
        plot_training_stats(stats)

    # Optional: Spiel gegen den Menschen
    play_against_human = input("Möchtest du gegen das trainierte Modell spielen? (j/n): ").lower() == 'j'
    if play_against_human:
        play_game_against_human(model, device=args.device)




if __name__ == "__main__":
    main()