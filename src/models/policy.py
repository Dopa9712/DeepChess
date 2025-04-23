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

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', temperature=0.3,
                 exploration_factor=0.0):
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
        self.temperature = temperature  # VERBESSERT: Standard-Temperatur auf 0.3 reduziert (war 1.0)
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

        # VERBESSERT: Heuristische Anpassung der Bewertung für offensichtlich gute Züge
        move_heuristic_bonus = self._calculate_move_heuristics(board, legal_moves)

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

        # VERBESSERT: Füge heuristische Boni zu den Logits hinzu
        for i, move in enumerate(legal_moves):
            move_idx = self._move_to_index(move)
            masked_logits[move_idx] += move_heuristic_bonus[i]

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

        # Extrahiere die Wahrscheinlichkeiten nur für die gültigen Züge
        move_probs = []
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            move_probs.append(probs[move_idx])

        # Sicherstellen, dass die Summe der Wahrscheinlichkeiten 1 ist
        move_probs = np.array(move_probs)

        # KORRIGIERT: Exploration durch zufällige Züge beimischen
        if self.exploration_factor > 0:
            n_moves = len(legal_moves)
            uniform_probs = np.ones(n_moves) / n_moves
            move_probs = (1 - self.exploration_factor) * move_probs + self.exploration_factor * uniform_probs

        sum_probs = np.sum(move_probs)

        if sum_probs <= 0 or np.isnan(sum_probs) or np.isinf(sum_probs):
            # Fallback: Gleichverteilung über alle gültigen Züge
            n_moves = len(legal_moves)
            move_probs = np.ones(n_moves) / n_moves
        else:
            # Normalisieren, um sicherzustellen, dass die Summe 1 ist
            move_probs = move_probs / sum_probs

        return move_probs, legal_moves

    def _calculate_move_heuristics(self, board: chess.Board, legal_moves: List[chess.Move]) -> List[float]:
        """
        Berechnet heuristische Bewertungen für Züge, um offensichtlich gute Züge zu bevorzugen.

        Args:
            board: Das aktuelle Schachbrett
            legal_moves: Liste der gültigen Züge

        Returns:
            List[float]: Liste mit heuristischen Bonuswerten für jeden Zug
        """
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.25,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0
        }

        bonus_values = []

        for move in legal_moves:
            bonus = 0.0

            # Bonus für Schlagzüge basierend auf dem Wert der geschlagenen Figur
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    # Gewichteter Wert für Schlagen ohne Gegenschlag
                    bonus += piece_values[captured_piece.piece_type] * 0.5

                    # Extra Bonus für Schlagen mit weniger wertvoller Figur (guter Tausch)
                    moving_piece = board.piece_at(move.from_square)
                    if moving_piece and piece_values[moving_piece.piece_type] < piece_values[captured_piece.piece_type]:
                        bonus += (piece_values[captured_piece.piece_type] - piece_values[moving_piece.piece_type]) * 0.3

            # Bonus für Schachgebung
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_check():
                bonus += 0.2

                # Extra Bonus für Schachmatt
                if board_copy.is_checkmate():
                    bonus += 10.0

            # Bonus für Rochade (Königssicherheit)
            if board.is_castling(move):
                bonus += 0.3

            # Bonus für Bauernumwandlung
            if move.promotion:
                bonus += piece_values[move.promotion] - piece_values[chess.PAWN]

            bonus_values.append(bonus)

        return bonus_values

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

        # VERBESSERT: Finde offensichtlich gewinnende Züge
        for i, move in enumerate(legal_moves):
            # Teste, ob dieser Zug zu Schachmatt führt
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_checkmate():
                return move  # Sofortiges Schachmatt immer wählen

        # VERBESSERT: Für die Evaluation verwenden wir häufiger den besten Zug
        # (niedrigere Temperatur während Evaluation)
        if self.exploration_factor == 0 and random.random() < 0.9:  # 90% Chance den besten Zug zu nehmen
            best_move_idx = np.argmax(probs)
            return legal_moves[best_move_idx]
        else:
            # Sonst Zug basierend auf den Wahrscheinlichkeiten auswählen
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