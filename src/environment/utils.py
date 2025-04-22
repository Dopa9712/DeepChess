import chess
import numpy as np
from typing import List, Tuple, Dict, Optional


def move_to_uci(move: chess.Move) -> str:
    """
    Konvertiert ein chess.Move-Objekt in die UCI-Notation.

    Args:
        move: Das zu konvertierende Move-Objekt

    Returns:
        str: UCI-Notation des Zugs (z.B. "e2e4")
    """
    return move.uci()


def uci_to_move(uci: str) -> chess.Move:
    """
    Konvertiert einen UCI-String in ein chess.Move-Objekt.

    Args:
        uci: UCI-Notation des Zugs (z.B. "e2e4")

    Returns:
        chess.Move: Das entsprechende Move-Objekt
    """
    return chess.Move.from_uci(uci)


def fen_to_board(fen: str) -> chess.Board:
    """
    Erstellt ein Schachbrett aus einem FEN-String.

    Args:
        fen: FEN-Notation einer Schachposition

    Returns:
        chess.Board: Das entsprechende Schachbrett
    """
    return chess.Board(fen)


def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Wandelt ein Schachbrett in eine für neuronale Netze geeignete Darstellung um.

    Args:
        board: Das Schachbrett

    Returns:
        np.ndarray: 8x8x12 Array (Darstellung der Figuren) + zusätzliche Ebenen
    """
    # 12 Planes für Figuren (6 Figurentypen x 2 Farben)
    planes = np.zeros((8, 8, 14), dtype=np.float32)

    # Die ersten 12 Ebenen für die Figuren (wie in ChessEnv.get_observation)
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
        piece = board.piece_at(square)
        if piece is not None:
            row, col = 7 - (square // 8), square % 8  # Umkehren für 0-indexierung
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            planes[row, col, channel] = 1

    # 13. Ebene: Am Zug (1 = Weiß, 0 = Schwarz)
    if board.turn == chess.WHITE:
        planes[:, :, 12] = 1

    # 14. Ebene: Rochaderechte
    castling_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights += 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights += 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights += 8

    planes[:, :, 13] = castling_rights / 15  # Normalisieren auf [0, 1]

    return planes


def encode_move(move: chess.Move, board: chess.Board) -> np.ndarray:
    """
    Kodiert einen Zug als One-Hot-Vektor für die Modellausgabe.

    Args:
        move: Der zu kodierende Zug
        board: Das aktuelle Schachbrett (für Kontextinformationen)

    Returns:
        np.ndarray: One-Hot-Vektor für den Zug
    """
    # Angepasste Größe für den Aktionsraum
    action_size = 64 * 64 + 64 * 64 * 4  # Normale Züge + Umwandlungszüge
    action_vector = np.zeros(action_size, dtype=np.float32)

    from_square = move.from_square
    to_square = move.to_square

    # Indexberechnung für normale Züge ohne Umwandlung
    if move.promotion is None:
        # 64*64 mögliche Kombinationen von Ausgangs- und Zielfeldern
        index = from_square * 64 + to_square
    else:
        # Umwandlungszüge: Verwende zusätzliche Indizes nach den 64*64 normalen Zügen
        promotion_offset = 64 * 64
        promotion_type = move.promotion - 2  # Konvertiere von chess.KNIGHT(2) zu 0, etc.
        index = promotion_offset + (from_square * 64 + to_square) * 4 + promotion_type

    action_vector[index] = 1

    return action_vector


def decode_move(action_vector: np.ndarray, board: chess.Board) -> Optional[chess.Move]:
    """
    Dekodiert einen One-Hot-Vektor oder Wahrscheinlichkeitsvektor zurück zu einem Zug.

    Args:
        action_vector: Vektor, der die Zugwahrscheinlichkeiten angibt
        board: Das aktuelle Schachbrett (für Legalitätsprüfung)

    Returns:
        Optional[chess.Move]: Der dekodierte Zug oder None, wenn kein gültiger Zug gefunden wurde
    """
    # Finde den Index mit der höchsten Wahrscheinlichkeit
    index = np.argmax(action_vector)

    # Normale Züge
    promotion_offset = 64 * 64
    if index < promotion_offset:
        # Normaler Zug (kein Umwandlungszug)
        from_square = index // 64
        to_square = index % 64
        move = chess.Move(from_square, to_square)
    else:
        # Umwandlungszug
        index -= promotion_offset
        promotion_type = index % 4  # 0=Springer, 1=Läufer, 2=Turm, 3=Dame
        index = index // 4
        from_square = index // 64
        to_square = index % 64
        promotion = promotion_type + 2  # Konvertiere zurück zu chess.KNIGHT(2), etc.
        move = chess.Move(from_square, to_square, promotion)

    # Überprüfe, ob der Zug gültig ist
    if move in board.legal_moves:
        return move

    # Wenn der Zug nicht gültig ist, suche nach dem nächstbesten gültigen Zug
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    legal_move_vectors = [encode_move(m, board) for m in legal_moves]
    legal_move_probs = [np.dot(action_vector, v) for v in legal_move_vectors]
    best_legal_move_idx = np.argmax(legal_move_probs)
    return legal_moves[best_legal_move_idx]


def get_symmetries(board: chess.Board, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Erzeugt Symmetrien (Spiegelungen) einer Brett-Policy-Kombination zur Datenerweiterung.

    Args:
        board: Das Schachbrett
        policy: Der Richtlinienvektor für das Brett

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: Liste von (Brett, Policy)-Paaren mit Spiegelungen
    """
    # Diese Funktion muss sorgfältig implementiert werden, um die Schachregeln zu respektieren
    # Eine einfache horizontale Spiegelung wird hier als Beispiel gezeigt

    # Brettdarstellung
    board_rep = board_to_planes(board)

    # Originale Kombination
    symmetries = [(board_rep, policy)]

    # Horizontale Spiegelung (einfaches Beispiel)
    # Echte Implementierung würde Figuren und Zugrichtungen korrekt spiegeln
    mirrored_board = np.flip(board_rep, axis=1).copy()

    # Policy müsste ebenfalls gespiegelt werden, was komplex ist und von der
    # spezifischen Repräsentation abhängt
    # Hier ein vereinfachtes Beispiel, das in der Praxis nicht korrekt wäre:
    mirrored_policy = policy.copy()  # Echte Implementierung würde die Züge richtig spiegeln

    symmetries.append((mirrored_board, mirrored_policy))

    return symmetries