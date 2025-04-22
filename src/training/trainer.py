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
from src.models.policy import RandomPolicy, NetworkPolicy, MCTSPolicy
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
            device: str = 'cpu',
            num_games: int = 100,
            max_moves: int = 1000,
            temperature: float = 1.0,
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
                    policy.temperature = 0.5  # Niedrigere Temperatur für spätere Züge

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
                result = 1.0  # Sieg
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
                    policy_probs=memory_item['policy_probs'],
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
            device: str = 'cpu',
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

        # Optimierer und Verlustfunktion
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=l2_reg
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
        # Phase 1: Selbstspiele für Datensammlung
        if experience_buffer is None:
            self.model.eval()  # Evaluierungsmodus für Selbstspiele
            worker = SelfPlayWorker(
                model=self.model,
                device=self.device,
                num_games=self.num_self_play_games
            )
            experience_buffer = worker.generate_games()

        # Phase 2: Training auf den gesammelten Daten
        train_stats = self.train_on_buffer(experience_buffer)

        # Phase 3: Evaluation gegen ältere Version (falls verfügbar)
        # Hier könntest du ein früheres Modell laden und gegen das neue testen
        # Aktuell vereinfacht: Keine Evaluierung gegen ältere Version

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

        # Trainingsstatistiken
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Daten in Batches aufteilen und Modell trainieren
        for _ in range(self.num_epochs):
            # Daten mischen
            indices = np.arange(len(experience_buffer))
            np.random.shuffle(indices)

            # In Batches durchlaufen
            for i in range(0, len(indices), self.batch_size):
                if i + self.batch_size > len(indices):
                    continue  # Letzten unvollständigen Batch überspringen

                batch_indices = indices[i:i + self.batch_size]
                states, policy_targets, value_targets, _ = experience_buffer.sample_batch(batch_indices)

                # Konvertierung zu Torch-Tensoren
                states = torch.FloatTensor(states).to(self.device)
                policy_targets = torch.FloatTensor(policy_targets).to(self.device)
                value_targets = torch.FloatTensor(value_targets).view(-1, 1).to(self.device)

                # Forward-Pass
                policy_logits, value_preds = self.model(states)

                # Verluste berechnen
                # Für die Policy: Kreuzentropie-Verlust (negative log-likelihood)
                policy_loss = -torch.mean(torch.sum(policy_targets * policy_logits, dim=1))

                # Für den Wert: MSE
                value_loss = self.value_loss_fn(value_preds, value_targets)

                # Gesamtverlust
                loss = policy_loss + value_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Statistiken aktualisieren
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        # Durchschnittliche Verluste berechnen
        avg_loss = total_loss / (num_batches or 1)
        avg_policy_loss = total_policy_loss / (num_batches or 1)
        avg_value_loss = total_value_loss / (num_batches or 1)

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

        # Richtlinie für das aktuelle Modell
        current_policy = NetworkPolicy(self.model, self.device, temperature=0.5)

        # Statistiken
        wins = 0
        losses = 0
        draws = 0

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
                else:
                    losses += 1
            else:
                # Remis oder Spiellimit erreicht
                draws += 1

        # Berechne Gewinnrate und ELO-Schätzung (vereinfacht)
        win_rate = (wins + 0.5 * draws) / self.evaluation_games

        # Speichere das beste Modell basierend auf der Gewinnrate
        if win_rate > self.best_model_win_rate:
            self.best_model_win_rate = win_rate
            self.save_model("best_model.pt")

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate
        }

    def save_model(self, filename: str) -> None:
        """
        Speichert das Modell in einer Datei.

        Args:
            filename: Name der Datei, in der das Modell gespeichert werden soll
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Modell gespeichert in {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Lädt ein Modell aus einer Datei.

        Args:
            filepath: Pfad zur Modelldatei
        """
        if not os.path.exists(filepath):
            print(f"Modelldatei {filepath} nicht gefunden.")
            return

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Modell geladen aus {filepath}")