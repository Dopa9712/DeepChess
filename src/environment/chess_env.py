import chess
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class ChessEnv:
    """
    Eine Schachspielumgebung für Reinforcement Learning basierend auf python-chess.
    Diese Klasse verwaltet den Spielzustand und die Interaktion mit dem RL-Agenten.
    """

    def __init__(self, fen: Optional[str] = None):
        """
        Initialisiert eine neue Schachspielumgebung.

        Args:
            fen: Optional, FEN-String zur Initialisierung des Bretts.
                 Wenn None, wird die Standardstartposition verwendet.
        """
        self.board = chess.Board(fen) if fen else chess.Board()
        self.reset_count = 0
        self.move_count = 0
        self.result = None

    def reset(self, fen: Optional[str] = None) -> np.ndarray:
        """
        Setzt das Schachbrett zurück, entweder auf die Standardstartposition oder
        auf eine durch FEN definierte Position.

        Args:
            fen: Optional, FEN-String zur Initialisierung des Bretts.

        Returns:
            np.ndarray: Die Darstellung des Anfangszustands des Bretts.
        """
        self.board = chess.Board(fen) if fen else chess.Board()
        self.reset_count += 1
        self.move_count = 0
        self.result = None
        return self.get_observation()

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Führt einen Zug im Spiel aus und gibt den neuen Zustand, die Belohnung,
        ein Flag für das Ende des Spiels und zusätzliche Informationen zurück.

        Args:
            action: Der auszuführende Zug, entweder als UCI-String (z.B. "e2e4")
                    oder als chess.Move-Objekt.

        Returns:
            Tuple mit:
            - np.ndarray: Neue Spielbrettdarstellung
            - float: Belohnung für diesen Schritt
            - bool: Ist das Spiel beendet?
            - Dict[str, Any]: Zusätzliche Informationen
        """
        if isinstance(action, str):
            move = chess.Move.from_uci(action)
        else:
            move = action

        # Überprüfen, ob der Zug gültig ist
        if move not in self.board.legal_moves:
            # Bestrafung für ungültigen Zug
            return self.get_observation(), -10.0, True, {"termination": "illegal_move"}

        # Zug ausführen
        self.board.push(move)
        self.move_count += 1

        # Belohnung und Spielende bestimmen
        reward = 0.0
        done = False
        info = {"move_count": self.move_count}

        # Spiel abgeschlossen?
        if self.board.is_checkmate():
            # Belohnung für Schachmatt (positiv für den Spieler, der Matt gesetzt hat)
            reward = 1.0 if not self.board.turn else -1.0
            done = True
            self.result = "1-0" if not self.board.turn else "0-1"
            info["termination"] = "checkmate"
        elif self.board.is_stalemate():
            # Unentschieden bei Patt
            reward = 0.0
            done = True
            self.result = "1/2-1/2"
            info["termination"] = "stalemate"
        elif self.board.is_insufficient_material():
            # Unentschieden bei unzureichendem Material
            reward = 0.0
            done = True
            self.result = "1/2-1/2"
            info["termination"] = "insufficient_material"
        elif self.board.is_fifty_moves():
            # Unentschieden bei 50-Züge-Regel
            reward = 0.0
            done = True
            self.result = "1/2-1/2"
            info["termination"] = "fifty_moves"
        elif self.board.is_repetition():
            # Unentschieden bei Zugwiederholung
            reward = 0.0
            done = True
            self.result = "1/2-1/2"
            info["termination"] = "repetition"

        # Zusätzliche Belohnungen für gute Spielaktionen
        if self.board.is_check():
            # Kleiner Bonus für Schach
            reward += 0.01
            info["check"] = True

        # Materialwert könnte auch als Teil der Belohnung dienen
        # Hier einfache Implementierung, könnte verbessert werden
        if not done:
            material_balance = self._get_material_balance()
            reward += 0.001 * material_balance  # Kleiner Einfluss des Materials

        return self.get_observation(), reward, done, info

    def _get_material_balance(self) -> float:
        """
        Berechnet die Materialbilanz aus Sicht des aktuellen Spielers.

        Returns:
            float: Materialbilanz (positiv = Vorteil für aktuellen Spieler)
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.25,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # König hat keinen Materialwert
        }

        material_balance = 0

        for piece_type in piece_values:
            # Zähle Weiße Figuren als positiv
            material_balance += piece_values[piece_type] * len(self.board.pieces(piece_type, chess.WHITE))
            # Zähle Schwarze Figuren als negativ
            material_balance -= piece_values[piece_type] * len(self.board.pieces(piece_type, chess.BLACK))

        # Aus Sicht des aktuellen Spielers
        return material_balance if self.board.turn == chess.WHITE else -material_balance

    def get_observation(self) -> np.ndarray:
        """
        Wandelt den aktuellen Spielzustand in eine für das RL-Modell geeignete Beobachtung um.

        Returns:
            np.ndarray: 8x8x12 Array, das die Position der Figuren repräsentiert
                        (6 Figurtypen x 2 Farben x 8x8 Brett)
        """
        observation = np.zeros((8, 8, 12), dtype=np.float32)

        # Mapping von Figuren zu Kanalindizes
        piece_to_channel = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11
        }

        # Figuren auf dem Brett in die entsprechenden Kanäle setzen
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                row, col = 7 - (square // 8), square % 8  # Umkehren der Reihenfolge für 0-indexierung
                channel = piece_to_channel[(piece.piece_type, piece.color)]
                observation[row, col, channel] = 1

        return observation

    def get_legal_moves(self) -> List[chess.Move]:
        """
        Gibt eine Liste aller gültigen Züge zurück.

        Returns:
            List[chess.Move]: Liste der gültigen Züge
        """
        return list(self.board.legal_moves)

    def get_legal_moves_mask(self) -> np.ndarray:
        """
        Erstellt eine Maske für gültige Züge.
        Diese kann verwendet werden, um ungültige Aktionen im RL-Modell zu maskieren.

        Returns:
            np.ndarray: Binäre Maske für gültige Züge
        """
        # Einfache Implementierung mit fester Größe
        # Dies könnte je nach Aktionsraumdarstellung angepasst werden
        # Hier verwenden wir ein vereinfachtes Schema mit 4672 möglichen Zügen (64*73)
        # (64 Ausgangsfelder × ca. 73 mögliche Züge pro Feld einschließlich Bauernumwandlungen)
        action_mask = np.zeros(4672, dtype=np.float32)

        # Diese Implementierung muss je nach Aktionsraumcodierung angepasst werden
        # Hier ein einfaches Beispiel:
        for move in self.board.legal_moves:
            move_idx = self._move_to_index(move)
            action_mask[move_idx] = 1

        return action_mask

    def _move_to_index(self, move: chess.Move) -> int:
        """
        Wandelt einen Zug in einen eindeutigen Index um.
        Diese Funktion muss an die gewählte Aktionsraumcodierung angepasst werden.

        Args:
            move: Das zu indizierende chess.Move-Objekt

        Returns:
            int: Eindeutiger Index für den Zug
        """
        # Dies ist eine vereinfachte Beispielimplementierung
        # Eine bessere Umsetzung würde die tatsächliche Struktur des Aktionsraums berücksichtigen
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion if move.promotion else 0

        # Einfache Formel: from_square * 64 * 5 + to_square * 5 + promotion
        # 5 mögliche Beförderungen (keine, Springer, Läufer, Turm, Dame)
        return from_square * 64 * 5 + to_square * 5 + (0 if promotion is None else promotion)

    def render(self, mode: str = 'unicode') -> str:
        """
        Stellt das Schachbrett dar.

        Args:
            mode: Darstellungsmodus ('unicode' oder 'ascii')

        Returns:
            str: String-Repräsentation des Bretts
        """
        if mode == 'unicode':
            return str(self.board)
        else:
            return self.board._repr_svg_()  # Für Jupyter-Notebooks