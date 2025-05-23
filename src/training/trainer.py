import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import os
import time
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm

from src.environment.chess_env import ChessEnv
from src.models.network import ChessNetwork
from src.models.policy import RandomPolicy, NetworkPolicy
from src.training.experience_buffer import ExperienceBuffer
from src.environment.utils import board_to_planes


class SelfPlayWorker:
    """
    Worker für das Selbstspiel zur Datensammlung.
    Generiert Trainingsdaten durch Spiele der KI gegen sich selbst.
    """

    def __init__(
            self,
            model: nn.Module,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            num_games: int = 100,
            max_moves: int = 1000,
            temperature: float = 0.8,  # VERBESSERT: Niedrigere Temperatur für bessere Spielqualität
            temperature_drop_move: int = 30,
            exploration_factor: float = 0.1,
            experience_buffer: Optional[ExperienceBuffer] = None
    ):
        """
        Initialisiert den SelfPlayWorker.

        Args:
            model: Das zu verwendende neuronale Netzwerk
            device: Gerät für das Modell ('cpu' oder 'cuda')
            num_games: Anzahl der zu spielenden Spiele
            max_moves: Maximale Anzahl von Zügen pro Spiel
            temperature: Anfängliche Temperatur für die Exploration
            temperature_drop_move: Zug, ab dem die Temperatur gesenkt wird
            exploration_factor: Faktor für zufällige Exploration
            experience_buffer: Optional, Buffer zum Speichern der Erfahrungen
        """
        self.model = model
        self.device = device
        self.num_games = num_games
        self.max_moves = max_moves
        self.temperature = temperature
        self.temperature_drop_move = temperature_drop_move
        self.exploration_factor = exploration_factor
        self.experience_buffer = experience_buffer or ExperienceBuffer()

    def generate_games(self) -> ExperienceBuffer:
        """
        Generiert eine Reihe von Selbstspielen und sammelt Trainingsdaten.

        Returns:
            ExperienceBuffer: Buffer mit den gesammelten Trainingsdaten
        """
        for game_idx in tqdm(range(self.num_games), desc="Selbstspiele"):
            # Umgebung initialisieren
            env = ChessEnv()
            board = env.board

            # Richtlinien für die beiden Spieler (gleiche für beide, mit Exploration)
            policy = NetworkPolicy(
                self.model,
                device=self.device,
                temperature=self.temperature,
                exploration_factor=self.exploration_factor
            )

            # Für das Spiel relevante Variablen
            game_memory = []  # Speichert (state, policy_probs, player) Tupel
            move_count = 0
            current_player = chess.WHITE  # Weiß beginnt

            # Spiel bis zum Ende durchführen
            while not board.is_game_over() and move_count < self.max_moves:
                # Temperatur reduzieren nach bestimmter Anzahl von Zügen
                if move_count == self.temperature_drop_move:
                    policy.temperature = 0.3  # VERBESSERT: Noch niedrigere Temperatur für spätere Züge

                # Aktuelle Brettdarstellung
                board_rep = board_to_planes(board)

                # Zugauswahl basierend auf der Richtlinie
                move_probs, legal_moves = policy.get_action_probs(board)

                # Zug in game_memory speichern (für später, wenn das Ergebnis bekannt ist)
                game_memory.append({
                    'state': board_rep,
                    'policy_probs': move_probs,
                    'legal_moves': legal_moves.copy(),
                    'player': current_player
                })

                # Zug auswählen und ausführen
                selected_idx = np.random.choice(len(legal_moves), p=move_probs)
                move = legal_moves[selected_idx]
                board.push(move)

                # Aktualisiere Spieler und Zähler
                current_player = not current_player
                move_count += 1

            # Spielergebnis bestimmen
            if board.is_checkmate():
                # Der Spieler, der nicht am Zug ist, hat gewonnen
                winner = not board.turn
                # VERBESSERT: Höhere Bewertung für Schachmatt
                result = 5.0  # Erhöhter Wert für Sieg (war 1.0)
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
                # Unentschieden
                winner = None
                result = 0.0  # Remis
            else:
                # Spiellimit erreicht
                winner = None
                result = 0.0  # Als Remis behandeln

            # Spielergebnis in den Erfahrungspuffer aufnehmen
            for memory_item in game_memory:
                # Wert aus der Perspektive des jeweiligen Spielers
                player_result = result if memory_item['player'] == winner else (
                    -result if winner is not None else 0.0
                )

                # Zum Erfahrungspuffer hinzufügen
                self.experience_buffer.add(
                    state=memory_item['state'],
                    policy_probs=memory_item['policy_probs'].tolist(),  # Konvertiert numpy array zu Liste
                    value=player_result,
                    legal_moves=memory_item['legal_moves']
                )

        return self.experience_buffer


class RLTrainer:
    """
    Trainer für das Reinforcement Learning der Schach-KI.
    Verwendet selbstspielbased reinforcement learning ähnlich zu AlphaZero.
    """

    def __init__(
            self,
            model: nn.Module,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate: float = 0.001,
            l2_reg: float = 1e-4,
            batch_size: int = 256,
            num_epochs: int = 10,
            checkpoint_dir: str = './checkpoints',
            num_self_play_games: int = 100,
            evaluation_games: int = 20
    ):
        """
        Initialisiert den RL-Trainer.

        Args:
            model: Das zu trainierende neuronale Netzwerk
            device: Gerät für das Training ('cpu' oder 'cuda')
            learning_rate: Lernrate für den Optimierer
            l2_reg: L2-Regularisierungsstärke
            batch_size: Batch-Größe für das Training
            num_epochs: Anzahl der Epochen pro Trainingsphase
            checkpoint_dir: Verzeichnis für Modell-Checkpoints
            num_self_play_games: Anzahl der Selbstspiele pro Trainingsiteration
            evaluation_games: Anzahl der Evaluierungsspiele
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # VERBESSERT: Optimierer-Parameter angepasst
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=l2_reg,
            betas=(0.9, 0.999)  # Standard-Betas, aber explizit definiert
        )

        # Scheduler für die Lernrate hinzugefügt
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Verlustfunktionen
        self.value_loss_fn = nn.MSELoss()

        # Training Parameter
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.num_self_play_games = num_self_play_games
        self.evaluation_games = evaluation_games

        # Für das Speichern der besten Modelle
        self.best_model_win_rate = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _move_to_index(self, move: chess.Move) -> int:
        """
        Wandelt einen Zug in einen eindeutigen Index um.
        Diese Methode muss mit der gleichen Indizierungslogik wie in der Umgebung und Policy implementiert werden.

        Args:
            move: Das zu indizierende chess.Move-Objekt

        Returns:
            int: Eindeutiger Index für den Zug
        """
        from_square = move.from_square  # 0-63
        to_square = move.to_square  # 0-63

        # Indexberechnung für normale Züge ohne Umwandlung
        if move.promotion is None:
            # 64*64 mögliche Kombinationen von Ausgangs- und Zielfeldern
            return from_square * 64 + to_square
        else:
            # Umwandlungszüge: Verwende zusätzliche Indizes nach den 64*64 normalen Zügen
            # Es gibt 4 Umwandlungstypen (Springer, Läufer, Turm, Dame)
            promotion_offset = 64 * 64

            # Offset basierend auf from_square, to_square und Umwandlungstyp
            # Wir kodieren den Umwandlungstyp als 0=Springer, 1=Läufer, 2=Turm, 3=Dame
            promotion_type = move.promotion - 2  # Konvertiere von chess.KNIGHT(2) zu 0, etc.
            return promotion_offset + (from_square * 64 + to_square) * 4 + promotion_type

    def train_iteration(self, experience_buffer: Optional[ExperienceBuffer] = None) -> Dict[str, float]:
        """
        Führt eine vollständige Trainingsiteration durch:
        1. Selbstspiele zur Datensammlung
        2. Training auf den gesammelten Daten
        3. Evaluation gegen ältere Version

        Args:
            experience_buffer: Optional, vorhandener Erfahrungspuffer

        Returns:
            Dict[str, float]: Statistiken über die Trainingsiteration
        """
        # Fortschritt verfolgen
        self.track_progress()

        # Phase 1: Selbstspiele für Datensammlung
        if experience_buffer is None:
            self.model.eval()  # Evaluierungsmodus für Selbstspiele

            # VERBESSERT: Adaptive Exploration basierend auf bisherigem Erfolg
            current_exploration = max(0.05, min(0.2, 0.2 - self.best_model_win_rate / 5.0))

            worker = SelfPlayWorker(
                model=self.model,
                device=self.device,
                num_games=self.num_self_play_games,
                temperature=0.8,
                exploration_factor=current_exploration
            )
            experience_buffer = worker.generate_games()

        # Phase 2: Training auf den gesammelten Daten
        train_stats = self.train_on_buffer(experience_buffer)

        # Lerrate anpassen basierend auf Trainingsverlust
        self.scheduler.step(train_stats['total_loss'])

        # Trainingsstatistiken speichern
        if hasattr(self, 'training_stats'):
            self.training_stats['policy_losses'].append(train_stats['policy_loss'])
            self.training_stats['value_losses'].append(train_stats['value_loss'])
            self.training_stats['total_losses'].append(train_stats['total_loss'])

        return {
            'train_loss': train_stats['total_loss'],
            'policy_loss': train_stats['policy_loss'],
            'value_loss': train_stats['value_loss'],
            'samples': len(experience_buffer)
        }

    def train_on_buffer(self, experience_buffer: ExperienceBuffer) -> Dict[str, float]:
        """
        Trainiert das Modell auf einem Erfahrungspuffer.

        Args:
            experience_buffer: Puffer mit Trainingsbeispielen

        Returns:
            Dict[str, float]: Trainingsstatistiken
        """
        self.model.train()  # Trainingsmodus aktivieren

        print(f"Starte Training auf {len(experience_buffer)} gesammelten Erfahrungen")

        # Trainingsstatistiken
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Daten in Batches aufteilen und Modell trainieren
        for epoch in range(self.num_epochs):
            print(f"Trainings-Epoche {epoch + 1}/{self.num_epochs}...")
            epoch_start_time = time.time()

            # Daten mischen
            indices = np.arange(len(experience_buffer))
            np.random.shuffle(indices)

            # In Batches durchlaufen
            batch_count = 0
            for i in range(0, len(indices), self.batch_size):
                if i + self.batch_size > len(indices):
                    continue  # Letzten unvollständigen Batch überspringen

                batch_indices = indices[i:i + self.batch_size]

                # Debug-Ausgabe alle 10 Batches
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Verarbeite Batch {batch_count}...")

                try:
                    # Daten laden
                    states, policy_targets, value_targets, legal_moves = experience_buffer.sample_batch(batch_indices)

                    # Konvertierung zu Torch-Tensoren und Umordnung der Dimensionen
                    # Von (batch_size, 8, 8, 14) zu (batch_size, 14, 8, 8)
                    states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)

                    # Da policy_targets jetzt eine Liste von Arrays unterschiedlicher Länge ist,
                    # können wir nicht einfach einen Tensor erstellen.
                    # Stattdessen berechnen wir den Verlust direkt für jeden Eintrag in der Liste.
                    value_targets = torch.FloatTensor(value_targets).view(-1, 1).to(self.device)

                    # Forward-Pass
                    policy_logits, value_preds = self.model(states)

                    # Für den Wert: MSE-Verlust
                    value_loss = self.value_loss_fn(value_preds, value_targets)

                    # Für die Policy: Berechnung des Verlustes für jedes Beispiel im Batch einzeln
                    policy_loss = 0.0

                    for j, (probs, moves) in enumerate(zip(policy_targets, legal_moves)):
                        # Wir erstellen eine Maske für gültige Züge
                        logits = policy_logits[j]
                        mask = torch.zeros_like(logits)

                        # Konvertiere die Wahrscheinlichkeiten in einen Tensor
                        target_probs = torch.zeros_like(logits)

                        # Setze die Wahrscheinlichkeiten für gültige Züge
                        for prob, move in zip(probs, moves):
                            move_idx = self._move_to_index(move)
                            mask[move_idx] = 1
                            target_probs[move_idx] = prob

                        # VERBESSERT: Berechne den Verlust richtig mit log_softmax
                        masked_logits = logits * mask
                        log_probs = torch.log_softmax(masked_logits + (1 - mask) * -1e9, dim=0)
                        cross_entropy = -torch.sum(target_probs * log_probs * mask)
                        policy_loss += cross_entropy

                    policy_loss = policy_loss / len(policy_targets)  # Durchschnitt über den Batch

                    # VERBESSERT: Gesamtverlust mit angepasster Gewichtung
                    # Erhöhe die Gewichtung des Value-Loss für bessere Stellungsbewertungen
                    loss = policy_loss + 2.0 * value_loss

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()

                    # VERBESSERT: Gradient Clipping für bessere Stabilität
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    # Statistiken aktualisieren
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"Fehler beim Verarbeiten von Batch {batch_count}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Epoche-Statistiken
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / (num_batches or 1)
            avg_policy_loss = total_policy_loss / (num_batches or 1)
            avg_value_loss = total_value_loss / (num_batches or 1)

            print(
                f"Epoche {epoch + 1} abgeschlossen in {epoch_time:.2f}s. Verluste: Total={avg_loss:.4f}, Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")

        # Durchschnittliche Verluste berechnen
        avg_loss = total_loss / (num_batches or 1)
        avg_policy_loss = total_policy_loss / (num_batches or 1)
        avg_value_loss = total_value_loss / (num_batches or 1)

        print(
            f"Training abgeschlossen. Gesamt-Durchschnitt: Total={avg_loss:.4f}, Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")

        return {
            'total_loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }

    def evaluate(self, opponent_model: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Evaluiert das aktuelle Modell gegen ein anderes Modell oder eine Baseline.

        Args:
            opponent_model: Optional, das gegnerische Modell (falls None, wird RandomPolicy verwendet)

        Returns:
            Dict[str, float]: Evaluierungsstatistiken
        """
        self.model.eval()  # Evaluierungsmodus

        # Falls kein Gegnermodell angegeben, verwende zufällige Richtlinie
        if opponent_model is None:
            opponent_policy = RandomPolicy()
        else:
            opponent_model.to(self.device)
            opponent_model.eval()
            opponent_policy = NetworkPolicy(opponent_model, self.device, temperature=0.5)

        # VERBESSERT: Niedrigere Temperatur und keine Exploration für die Evaluation
        current_policy = NetworkPolicy(
            self.model,
            self.device,
            temperature=0.1,  # VERBESSERT: Niedrigere Temperatur für Evaluation
            exploration_factor=0.0  # Keine Exploration während der Evaluation
        )

        # Statistiken
        wins = 0
        losses = 0
        draws = 0
        # VERBESSERT: Erfasse mehr Statistiken
        checkmate_wins = 0
        checkmate_losses = 0
        material_stats = []

        # Evaluierungsspiele durchführen
        for game_idx in tqdm(range(self.evaluation_games), desc="Evaluation"):
            env = ChessEnv()
            board = env.board
            move_count = 0

            # Wähle zufällig aus, wer beginnt (Weiß oder Schwarz)
            current_starts = game_idx % 2 == 0

            while not board.is_game_over() and move_count < 200:  # Limit von 200 Zügen
                # Bestimme, welche Richtlinie verwendet werden soll
                is_current_turn = (board.turn == chess.WHITE) == current_starts
                policy_to_use = current_policy if is_current_turn else opponent_policy

                # Wähle Zug und führe ihn aus
                move = policy_to_use.get_action(board)
                board.push(move)
                move_count += 1

            # Spielergebnis bestimmen
            if board.is_checkmate():
                winner_is_white = not board.turn  # Der Spieler, der nicht am Zug ist, hat gewonnen
                if (winner_is_white == current_starts):
                    wins += 1
                    checkmate_wins += 1
                else:
                    losses += 1
                    checkmate_losses += 1
            else:
                # Remis oder Spiellimit erreicht
                draws += 1

            # VERBESSERT: Materialbilanz am Ende erfassen
            material_balance = 0
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                material_balance += len(board.pieces(piece_type, chess.WHITE))
                material_balance -= len(board.pieces(piece_type, chess.BLACK))
            # Materialbilanz aus Sicht des aktuellen Modells
            if not current_starts:  # Wenn das Modell Schwarz spielt
                material_balance = -material_balance
            material_stats.append(material_balance)

        # Berechne Gewinnrate und ELO-Schätzung (vereinfacht)
        win_rate = (wins + 0.5 * draws) / self.evaluation_games

        # VERBESSERT: Detailliertere Statistiken
        avg_material_advantage = sum(material_stats) / len(material_stats) if material_stats else 0

        # Speichere das beste Modell basierend auf der Gewinnrate
        if win_rate > self.best_model_win_rate:
            self.best_model_win_rate = win_rate
            self.save_model("best_model.pt")

            # Aktualisiere Trainingsstatistiken
            if hasattr(self, 'training_stats'):
                self.training_stats['best_win_rate'] = win_rate
                self.training_stats['last_improvement_iter'] = self.training_stats['iterations']

        # Speichere Gewinnrate in Trainingsstatistiken
        if hasattr(self, 'training_stats'):
            self.training_stats['win_rates'].append(win_rate)

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'checkmate_wins': checkmate_wins,
            'checkmate_losses': checkmate_losses,
            'avg_material': avg_material_advantage
        }

    def save_model(self, filename: str) -> None:
        """
        Speichert das Modell in einer Datei mit zusätzlichen Metadaten.

        Args:
            filename: Name der Datei, in der das Modell gespeichert werden soll
        """
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Mehr Informationen im Checkpoint speichern
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  # Scheduler-Status speichern
            'best_win_rate': self.best_model_win_rate,
            'timestamp': time.time(),
            'training_stats': getattr(self, 'training_stats', {})  # Historische Daten speichern, falls vorhanden
        }, filepath)

        print(f"Modell gespeichert in {filepath}")

        # Auch das beste Modell in einer separaten Datei speichern, wenn es sich verbessert hat
        if "best" in filename or "iter" in filename:
            # Kopie von best_model.pt als best_model_iter_X.pt speichern
            if self.best_model_win_rate > 0 and "iter" in filename:
                iter_num = filename.split("_")[-1].split(".")[0]
                best_copy_path = os.path.join(self.checkpoint_dir, f"best_model_iter_{iter_num}.pt")
                import shutil
                best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                if os.path.exists(best_model_path):
                    shutil.copy2(best_model_path, best_copy_path)
                    print(f"Kopie des besten Modells gespeichert in {best_copy_path}")

    def load_model(self, filepath: str) -> None:
        """
        Lädt ein Modell aus einer Datei mit zusätzlichen Metadaten.

        Args:
            filepath: Pfad zur Modelldatei
        """
        if not os.path.exists(filepath):
            print(f"Modelldatei {filepath} nicht gefunden.")
            return

        print(f"Lade Modell aus {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)

        # Modell und Optimierer laden
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Scheduler laden, falls vorhanden
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Weitere Metadaten laden
        if 'best_win_rate' in checkpoint:
            self.best_model_win_rate = checkpoint['best_win_rate']
            print(f"Beste Gewinnrate: {self.best_model_win_rate:.4f}")

        if 'timestamp' in checkpoint:
            load_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint['timestamp']))
            print(f"Modell wurde gespeichert am: {load_time}")

        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']

        print(f"Modell erfolgreich geladen!")

    def track_progress(self):
        """
        Initialisiert oder aktualisiert die Tracking-Informationen für das Training.
        """
        if not hasattr(self, 'training_stats'):
            self.training_stats = {
                'iterations': 0,
                'win_rates': [],
                'policy_losses': [],
                'value_losses': [],
                'total_losses': [],
                'best_win_rate': 0.0,
                'last_improvement_iter': 0
            }

        self.training_stats['iterations'] += 1