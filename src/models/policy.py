import torch
import torch.nn.functional as F
import numpy as np
import chess
import random
from typing import List, Tuple, Dict, Any, Optional

from src.environment.utils import encode_move, decode_move, board_to_planes


class RandomPolicy:
    """
    Eine einfache zufällige Richtlinie, die zufällige gültige Züge auswählt.
    Nützlich als Baseline und für frühe Trainingsphasen.
    """

    def __init__(self):
        pass

    def get_action(self, board: chess.Board) -> chess.Move:
        """
        Wählt einen zufälligen gültigen Zug.

        Args:
            board: Das aktuelle Schachbrett

        Returns:
            chess.Move: Ein zufälliger gültiger Zug
        """
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

    def get_action_probs(self, board: chess.Board) -> Tuple[np.ndarray, List[chess.Move]]:
        """
        Erstellt eine Gleichverteilung über alle gültigen Züge.

        Args:
            board: Das aktuelle Schachbrett

        Returns:
            Tuple aus:
            - np.ndarray: Wahrscheinlichkeiten für alle gültigen Züge (gleichverteilt)
            - List[chess.Move]: Liste der entsprechenden gültigen Züge
        """
        legal_moves = list(board.legal_moves)
        n_moves = len(legal_moves)

        if n_moves == 0:
            return np.array([]), []

        # Gleichverteilung über alle gültigen Züge
        probs = np.ones(n_moves) / n_moves

        return probs, legal_moves


class NetworkPolicy:
    """
    Eine Richtlinie, die auf einem neuronalen Netzwerk basiert.
    Diese Klasse verwendet ein trainiertes Modell, um Züge zu bewerten und auszuwählen.
    """

    def __init__(self, model, device='cpu', temperature=1.0, exploration_factor=0.0):
        """
        Initialisiert die Netzwerk-Richtlinie.

        Args:
            model: Das trainierte neuronale Netzwerk
            device: Gerät, auf dem das Modell ausgeführt wird ('cpu' oder 'cuda')
            temperature: Temperaturparameter für die Exploration (höhere Werte -> mehr Exploration)
            exploration_factor: Faktor für zufällige Exploration (0 = keine zufällige Exploration)
        """
        self.model = model
        self.device = device
        self.temperature = temperature
        self.exploration_factor = exploration_factor
        self.model.to(device)
        self.model.eval()  # Setzt das Modell in den Evaluationsmodus

    def get_action_probs(self, board: chess.Board) -> Tuple[np.ndarray, List[chess.Move]]:
        """
        Berechnet die Wahrscheinlichkeiten für alle gültigen Züge basierend auf dem Netzwerk.

        Args:
            board: Das aktuelle Schachbrett

        Returns:
            Tuple aus:
            - np.ndarray: Wahrscheinlichkeiten für alle gültigen Züge
            - List[chess.Move]: Liste der entsprechenden gültigen Züge
        """
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return np.array([]), []

        # Brett in Netzwerk-Eingabeformat umwandeln
        board_planes = board_to_planes(board)
        # Umordnen der Dimensionen von (8, 8, 14) zu (1, 14, 8, 8) für das Konvolutionsnetzwerk
        board_tensor = torch.FloatTensor(board_planes).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Vorhersage vom Netzwerk
        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)
            policy_logits = policy_logits.cpu().numpy()[0]

        # Masking: Nur gültige Züge berücksichtigen
        mask = np.zeros(policy_logits.shape, dtype=np.float32)
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            mask[move_idx] = 1

        # Anwendung der Maske und Normalisierung
        masked_logits = policy_logits * mask
        masked_logits = masked_logits - np.max(masked_logits)  # Numerische Stabilität
        masked_exp = np.exp(masked_logits / self.temperature) * mask
        sum_exp = np.sum(masked_exp)

        # Überprüfen, ob die Summe gültig ist und nicht 0
        if sum_exp <= 0 or np.isnan(sum_exp) or np.isinf(sum_exp):
            # Fallback: Gleichverteilung über alle gültigen Züge
            n_moves = len(legal_moves)
            move_probs = np.ones(n_moves) / n_moves
            return move_probs, legal_moves

        probs = masked_exp / sum_exp

        # Exploration durch zufällige Züge beimischen
        if self.exploration_factor > 0:
            n_moves = len(legal_moves)
            uniform_probs = np.ones(policy_logits.shape) * mask / n_moves
            probs = (1 - self.exploration_factor) * probs + self.exploration_factor * uniform_probs

        # Extrahiere die Wahrscheinlichkeiten nur für die gültigen Züge
        move_probs = []
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            move_probs.append(probs[move_idx])

        # Sicherstellen, dass die Summe der Wahrscheinlichkeiten 1 ist
        move_probs = np.array(move_probs)
        sum_probs = np.sum(move_probs)

        if sum_probs <= 0 or np.isnan(sum_probs) or np.isinf(sum_probs):
            # Fallback: Gleichverteilung über alle gültigen Züge
            n_moves = len(legal_moves)
            move_probs = np.ones(n_moves) / n_moves
        else:
            # Normalisieren, um sicherzustellen, dass die Summe 1 ist
            move_probs = move_probs / sum_probs

        return move_probs, legal_moves

    def get_action(self, board: chess.Board) -> chess.Move:
        """
        Wählt einen Zug basierend auf den Netzwerkwahrscheinlichkeiten aus.

        Args:
            board: Das aktuelle Schachbrett

        Returns:
            chess.Move: Der ausgewählte Zug
        """
        probs, legal_moves = self.get_action_probs(board)

        if not legal_moves:
            return None

        # Zug basierend auf den Wahrscheinlichkeiten auswählen
        move_idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[move_idx]

    def _move_to_index(self, move: chess.Move) -> int:
        """
        Wandelt einen Zug in einen Index um (für die Modellausgabe).

        Args:
            move: Der umzuwandelnde Zug

        Returns:
            int: Der entsprechende Index
        """
        # Verbesserte Implementierung mit größerem Aktionsraum
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


class MCTSPolicy:
    """
    Eine Richtlinie basierend auf Monte Carlo Tree Search (MCTS).
    Diese fortgeschrittene Richtlinie verwendet MCTS mit einem neuronalen Netzwerk.
    """

    def __init__(self, model, device='cpu', num_simulations=800, c_puct=1.0):
        """
        Initialisiert die MCTS-Richtlinie.

        Args:
            model: Das neuronale Netzwerk für die Bewertung
            device: Gerät, auf dem das Modell ausgeführt wird
            num_simulations: Anzahl der MCTS-Simulationen pro Zug
            c_puct: Exploration/Exploitation-Parameter (höher = mehr Exploration)
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.model.to(device)
        self.model.eval()

        # MCTS-Statistiken
        self.Q = {}  # Q-Werte: Erwartete Belohnung für Kante (s,a)
        self.N = {}  # Besuchszähler für Kante (s,a)
        self.W = {}  # Kumulative Belohnung für Kante (s,a)
        self.P = {}  # Grundwahrscheinlichkeiten für Züge aus dem Netzwerk

    def get_action(self, board: chess.Board) -> chess.Move:
        """
        Führt MCTS durch und wählt den besten Zug.

        Args:
            board: Das aktuelle Schachbrett

        Returns:
            chess.Move: Der beste gefundene Zug
        """
        # MCTS-Simulationen durchführen
        for _ in range(self.num_simulations):
            self._search(board.copy())

        # FEN-String als Schlüssel für den aktuellen Zustand
        s = board.fen()

        # Zug mit der höchsten Besuchszahl wählen
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        counts = [self.N.get((s, self._move_to_str(a)), 0) for a in legal_moves]
        best_move = legal_moves[np.argmax(counts)]

        return best_move

    def _search(self, board: chess.Board) -> float:
        """
        Führt eine einzelne MCTS-Simulation durch.

        Args:
            board: Das aktuelle Schachbrett

        Returns:
            float: Der berechnete Wert für diesen Knoten
        """
        # FEN-String als Schlüssel für den aktuellen Zustand
        s = board.fen()

        # Spielende überprüfen
        if board.is_game_over():
            # Bewertung aus dem Spielergebnis
            result = board.result()
            if result == "1-0":
                return 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                return -1.0 if board.turn == chess.WHITE else 1.0
            else:  # Unentschieden
                return 0.0

        # Blatt im Suchbaum (noch nicht expandiert)
        if s not in self.P:
            # Board in Netzwerk-Eingabeformat umwandeln
            board_planes = board_to_planes(board)
            # Umordnen der Dimensionen von (8, 8, 14) zu (1, 14, 8, 8)
            board_tensor = torch.FloatTensor(board_planes).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Vorhersage vom Netzwerk
            with torch.no_grad():
                policy_logits, value = self.model(board_tensor)
                policy_logits = policy_logits.cpu().numpy()[0]
                value = value.item()

            # Grundwahrscheinlichkeiten für alle gültigen Züge speichern
            self.P[s] = {}
            legal_moves = list(board.legal_moves)
            policy_sum = 0

            for move in legal_moves:
                move_str = self._move_to_str(move)
                move_idx = self._move_to_index(move)
                self.P[s][move_str] = np.exp(policy_logits[move_idx])
                policy_sum += self.P[s][move_str]

            # Normalisierung, falls nötig
            if policy_sum > 0:
                for move_str in self.P[s]:
                    self.P[s][move_str] /= policy_sum

            # Für diesen Knoten noch keine Statsitiken vorhanden
            for move_str in self.P[s]:
                self.Q[(s, move_str)] = 0
                self.N[(s, move_str)] = 0
                self.W[(s, move_str)] = 0

            # Rückgabe des Netzwerkwerts aus Sicht des aktuellen Spielers
            return value

        # Wähle Aktion mit höchstem Q + U Wert
        legal_moves = list(board.legal_moves)
        best_move = None
        best_value = -float('inf')

        for move in legal_moves:
            move_str = self._move_to_str(move)

            if move_str in self.P[s]:
                # Q-Wert: Durchschnittliche Belohnung
                q = self.Q.get((s, move_str), 0)

                # Exploration-Term
                u = self.c_puct * self.P[s][move_str] * np.sqrt(
                    sum(self.N.get((s, a_str), 0) for a_str in self.P[s])) / (1 + self.N.get((s, move_str), 0))

                # Gesamtwert
                value = q + u

                if value > best_value:
                    best_value = value
                    best_move = move

        # Wenn kein Zug gefunden wurde (sollte nicht passieren)
        if best_move is None:
            return 0

        # Zug ausführen
        board.push(best_move)
        move_str = self._move_to_str(best_move)

        # Rekursiver Aufruf (mit negativem Wert, da Perspektivwechsel)
        v = -self._search(board)

        # Aktualisiere Statistiken
        self.W[(s, move_str)] = self.W.get((s, move_str), 0) + v
        self.N[(s, move_str)] = self.N.get((s, move_str), 0) + 1
        self.Q[(s, move_str)] = self.W[(s, move_str)] / self.N[(s, move_str)]

        return v

    def _move_to_str(self, move: chess.Move) -> str:
        """
        Konvertiert einen Zug in eine String-Repräsentation für den Schlüssel.

        Args:
            move: Der umzuwandelnde Zug

        Returns:
            str: String-Repräsentation des Zugs
        """
        return move.uci()

    def _move_to_index(self, move: chess.Move) -> int:
        """
        Wandelt einen Zug in einen Index um (für die Modellausgabe).

        Args:
            move: Der umzuwandelnde Zug

        Returns:
            int: Der entsprechende Index
        """
        # Verbesserte Implementierung mit größerem Aktionsraum
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