import numpy as np
import chess
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import deque


class ExperienceBuffer:
    """
    Ein Puffer zum Speichern und Samplen von Erfahrungen für das Reinforcement Learning.
    Speichert Zustände, Richtlinienwahrscheinlichkeiten, Werte und gültige Züge.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialisiert einen leeren Erfahrungspuffer.

        Args:
            max_size: Optional, maximale Größe des Puffers (falls None, unbegrenzt)
        """
        self.max_size = max_size
        self.states = []
        self.policy_probs = []
        self.values = []
        self.legal_moves = []

        # Wenn max_size gesetzt ist, verwende deque mit fester Größe
        if max_size is not None:
            self.states = deque(maxlen=max_size)
            self.policy_probs = deque(maxlen=max_size)
            self.values = deque(maxlen=max_size)
            self.legal_moves = deque(maxlen=max_size)

    def add(self, state: np.ndarray, policy_probs: np.ndarray, value: float, legal_moves: List[chess.Move]) -> None:
        """
        Fügt eine neue Erfahrung zum Puffer hinzu.

        Args:
            state: Der Spielzustand (Brett-Darstellung)
            policy_probs: Wahrscheinlichkeiten für Züge aus der Richtlinie
            value: Wert/Belohnung für diesen Zustand
            legal_moves: Liste der gültigen Züge in diesem Zustand
        """
        self.states.append(state)
        self.policy_probs.append(policy_probs)
        self.values.append(value)
        self.legal_moves.append(legal_moves)

    def sample_batch(self, indices: Optional[np.ndarray] = None, batch_size: Optional[int] = None) -> Tuple[
        np.ndarray, List, np.ndarray, List]:
        """
        Sampelt einen Batch von Erfahrungen aus dem Puffer.

        Args:
            indices: Optional, spezifische Indizes, die gesampelt werden sollen
            batch_size: Optional, Größe des zu sampelnden Batches (falls indices nicht angegeben)

        Returns:
            Tuple aus:
            - np.ndarray: Batch von Zuständen
            - List: Batch von Richtlinienwahrscheinlichkeiten (verschiedene Längen)
            - np.ndarray: Batch von Werten
            - List: Batch von Listen gültiger Züge
        """
        if indices is None:
            if batch_size is None:
                batch_size = min(64, len(self))  # Standard-Batch-Größe

            if len(self) <= batch_size:
                indices = np.arange(len(self))
            else:
                indices = np.random.choice(len(self), batch_size, replace=False)

        # Konvertiere Listen zu NumPy-Arrays und sampele mit Indizes
        states_batch = np.array([self.states[i] for i in indices])  # Form (batch_size, 8, 8, 14)

        # Die Wahrscheinlichkeiten bleiben als Liste, da jede Zustandsposition unterschiedlich viele gültige Züge haben kann
        policy_probs_batch = [self.policy_probs[i] for i in indices]

        values_batch = np.array([self.values[i] for i in indices])
        legal_moves_batch = [self.legal_moves[i] for i in indices]

        return states_batch, policy_probs_batch, values_batch, legal_moves_batch

    def __len__(self) -> int:
        """
        Gibt die aktuelle Größe des Puffers zurück.

        Returns:
            int: Anzahl der gespeicherten Erfahrungen
        """
        return len(self.states)

    def save(self, filepath: str) -> None:
        """
        Speichert den Erfahrungspuffer in einer Datei.

        Args:
            filepath: Pfad zur Speicherdatei
        """
        # Konvertiere die Züge in UCI-Strings für das Speichern
        legal_moves_uci = [[move.uci() for move in moves] for moves in self.legal_moves]

        np.savez(
            filepath,
            states=np.array(self.states),
            policy_probs=np.array(self.policy_probs),
            values=np.array(self.values),
            legal_moves_uci=legal_moves_uci
        )
        print(f"Erfahrungspuffer gespeichert in {filepath}")

    def load(self, filepath: str) -> None:
        """
        Lädt einen Erfahrungspuffer aus einer Datei.

        Args:
            filepath: Pfad zur Datei, aus der geladen werden soll
        """
        data = np.load(filepath, allow_pickle=True)

        # Lade Daten
        self.states = list(data['states'])
        self.policy_probs = list(data['policy_probs'])
        self.values = list(data['values'])

        # Konvertiere UCI-Strings zurück zu chess.Move
        legal_moves_uci = data['legal_moves_uci']
        self.legal_moves = []
        for moves_uci in legal_moves_uci:
            self.legal_moves.append([chess.Move.from_uci(uci) for uci in moves_uci])

        print(f"Erfahrungspuffer geladen aus {filepath} mit {len(self)} Erfahrungen")


class ReplayBuffer:
    """
    Ein erweiterter Erfahrungspuffer für verschiedene RL-Algorithmen wie DQN.
    Speichert Übergänge (Zustand, Aktion, Belohnung, nächster Zustand, Endmarkierung).
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialisiert einen leeren Replay-Buffer.

        Args:
            max_size: Maximale Größe des Puffers
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, state: np.ndarray, action: Union[chess.Move, int], reward: float, next_state: np.ndarray,
            done: bool) -> None:
        """
        Fügt einen Übergang zum Puffer hinzu.

        Args:
            state: Aktueller Zustand
            action: Ausgeführte Aktion (Zug oder Index)
            reward: Erhaltene Belohnung
            next_state: Folgezustand
            done: Flag, ob die Episode beendet ist
        """
        # Konvertiere Zug zu einem String, wenn es ein chess.Move ist
        if isinstance(action, chess.Move):
            action = action.uci()

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, List, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sampelt einen zufälligen Batch von Übergängen.

        Args:
            batch_size: Größe des zu sampelnden Batches

        Returns:
            Tuple aus:
            - np.ndarray: Batch von Zuständen
            - List: Batch von Aktionen
            - np.ndarray: Batch von Belohnungen
            - np.ndarray: Batch von Folgezuständen
            - np.ndarray: Batch von Done-Flags
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Zerlege Batch in separate Arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            list(actions),  # Aktionen als Liste (können unterschiedliche Typen sein)
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """
        Gibt die aktuelle Größe des Puffers zurück.

        Returns:
            int: Anzahl der gespeicherten Übergänge
        """
        return len(self.buffer)