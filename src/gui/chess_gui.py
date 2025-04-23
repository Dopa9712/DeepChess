import pygame
import chess
import os
import sys
import torch
from typing import Tuple, Optional

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.chess_env import ChessEnv
from src.models.policy import NetworkPolicy
from src.models.network import ChessNetwork


class ChessGUI:
    """
    Graphical User Interface for playing chess against a trained model.
    Uses pygame for drawing the board and handling mouse interactions.
    """

    def __init__(self, model=None, device='cpu', width=600, height=600):
        """
        Initialize the Chess GUI.

        Args:
            model: The trained chess model (if None, will use random moves)
            device: Device to run the model on ('cpu' or 'cuda')
            width: Width of the window
            height: Height of the window
        """
        pygame.init()
        pygame.display.set_caption('DeepChess - Play Against AI')

        self.width = width
        self.height = height
        self.board_size = min(width, height)
        self.square_size = self.board_size // 8

        # Create the display
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Chess environment
        self.env = ChessEnv()
        self.board = self.env.board

        # Colors
        self.light_square = (240, 217, 181)  # Light beige
        self.dark_square = (181, 136, 99)  # Dark brown
        self.highlight_color = (124, 252, 0, 128)  # Semi-transparent green
        self.selected_color = (255, 255, 0, 128)  # Semi-transparent yellow
        self.move_hint_color = (173, 216, 230, 150)  # Semi-transparent light blue

        # State
        self.selected_square = None
        self.highlighted_moves = []
        self.game_over = False
        self.message = None

        # Model/AI
        self.model = model
        self.device = device
        self.policy = None
        if model:
            self.policy = NetworkPolicy(model, device=device, temperature=0.5)

        # Load chess piece images
        self.piece_images = self._load_piece_images()

        # Font for displaying messages
        self.font = pygame.font.SysFont('Arial', 24)

    def _load_piece_images(self) -> dict:
        """
        Load the chess piece images.
        This creates simple labeled images for each piece.
        In a real application, you would use actual chess piece images.

        Returns:
            dict: Dictionary mapping piece symbol to its image
        """
        piece_symbols = {
            'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
            'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'
        }

        piece_images = {}
        for symbol, label in piece_symbols.items():
            # Create a circular surface for the piece with color based on whether it's white or black
            color = (230, 230, 230) if symbol.isupper() else (50, 50, 50)
            surface = pygame.Surface((self.square_size - 10, self.square_size - 10), pygame.SRCALPHA)
            pygame.draw.circle(surface, color, (surface.get_width() // 2, surface.get_height() // 2),
                               surface.get_width() // 2)

            # Add the piece label
            text_color = (50, 50, 50) if symbol.isupper() else (230, 230, 230)
            font = pygame.font.SysFont('Arial', self.square_size // 3)
            text = font.render(symbol.upper(), True, text_color)
            text_rect = text.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))
            surface.blit(text, text_rect)

            piece_images[symbol] = surface

        return piece_images

    def draw_board(self):
        """Draw the chess board squares."""
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size

                color = self.light_square if (row + col) % 2 == 0 else self.dark_square
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))

                # Draw rank and file labels
                if col == 0:  # Ranks (1-8)
                    rank_label = str(8 - row)
                    font = pygame.font.SysFont('Arial', 14)
                    text = font.render(rank_label, True,
                                       self.dark_square if (row + col) % 2 == 0 else self.light_square)
                    self.screen.blit(text, (x + 2, y + 2))

                if row == 7:  # Files (a-h)
                    file_label = chr(97 + col)  # ASCII 'a' is 97
                    font = pygame.font.SysFont('Arial', 14)
                    text = font.render(file_label, True,
                                       self.dark_square if (row + col) % 2 == 0 else self.light_square)
                    self.screen.blit(text, (x + self.square_size - 14, y + self.square_size - 14))

    def draw_pieces(self):
        """Draw the chess pieces on the board."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                symbol = piece.symbol()

                # Convert square to coordinates (square 0 is a8 in chess notation)
                row, col = 7 - (square // 8), square % 8  # Flip row for display
                x = col * self.square_size + 5  # Center pieces in squares
                y = row * self.square_size + 5

                self.screen.blit(self.piece_images[symbol], (x, y))

    def draw_highlights(self):
        """Draw highlights for selected square and possible moves."""
        if self.selected_square is not None:
            # Highlight selected square
            row, col = 7 - (self.selected_square // 8), self.selected_square % 8
            x = col * self.square_size
            y = row * self.square_size

            # Create a semi-transparent surface for the highlight
            highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            highlight_surface.fill(self.selected_color)
            self.screen.blit(highlight_surface, (x, y))

            # Highlight possible moves
            for move in self.highlighted_moves:
                to_square = move.to_square
                row, col = 7 - (to_square // 8), to_square % 8
                x = col * self.square_size
                y = row * self.square_size

                highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                highlight_surface.fill(self.move_hint_color)
                self.screen.blit(highlight_surface, (x, y))

    def draw_message(self):
        """Draw a message at the bottom of the screen."""
        if self.message:
            text = self.font.render(self.message, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, self.height - 20))

            # Draw a semi-transparent background for the text
            background = pygame.Surface((text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA)
            background.fill((0, 0, 0, 180))  # Semi-transparent black
            self.screen.blit(background, (text_rect.left - 10, text_rect.top - 5))

            self.screen.blit(text, text_rect)

    def square_from_coords(self, x: int, y: int) -> Optional[int]:
        """
        Convert screen coordinates to chess square index.

        Args:
            x: x-coordinate
            y: y-coordinate

        Returns:
            Optional[int]: The square index (0-63) or None if out of bounds
        """
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return None

        col = x // self.square_size
        row = y // self.square_size

        # Convert to chess.Square (0 = a8, 63 = h1)
        square = chess.square(col, 7 - row)
        return square

    def handle_move(self, from_square: int, to_square: int) -> bool:
        """
        Handle a move from the player.

        Args:
            from_square: Source square index (0-63)
            to_square: Target square index (0-63)

        Returns:
            bool: True if the move was made, False otherwise
        """
        # Check if this is a valid move
        for move in self.board.legal_moves:
            if move.from_square == from_square and move.to_square == to_square:
                # Handle promotion if needed
                if move.promotion is not None:
                    # For simplicity, always promote to queen
                    # In a real app, you'd show a dialog here
                    move = chess.Move(from_square, to_square, chess.QUEEN)

                # Make the move
                self.board.push(move)
                self.selected_square = None
                self.highlighted_moves = []
                return True

        return False

    def make_ai_move(self):
        """Make a move for the AI."""
        if self.policy:
            move = self.policy.get_action(self.board)
            self.board.push(move)

            # Highlight AI's move
            from_square = move.from_square
            to_square = move.to_square

            # Create a temporary highlight for the move
            row, col = 7 - (from_square // 8), from_square % 8
            from_x = col * self.square_size
            from_y = row * self.square_size

            row, col = 7 - (to_square // 8), to_square % 8
            to_x = col * self.square_size
            to_y = row * self.square_size

            # Draw temporary highlight for AI move
            pygame.draw.rect(self.screen, self.highlight_color, (from_x, from_y, self.square_size, self.square_size), 3)
            pygame.draw.rect(self.screen, self.highlight_color, (to_x, to_y, self.square_size, self.square_size), 3)
            pygame.display.flip()
            pygame.time.delay(500)  # Show the highlight for 500ms

    def check_game_over(self):
        """Check if the game is over and set appropriate message."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.message = f"Checkmate! {winner} wins."
            self.game_over = True
        elif self.board.is_stalemate():
            self.message = "Stalemate! Game drawn."
            self.game_over = True
        elif self.board.is_insufficient_material():
            self.message = "Insufficient material! Game drawn."
            self.game_over = True
        elif self.board.is_fifty_moves():
            self.message = "Fifty-move rule! Game drawn."
            self.game_over = True
        elif self.board.is_repetition():
            self.message = "Threefold repetition! Game drawn."
            self.game_over = True

    def run(self):
        """Main game loop."""
        running = True
        self.message = "You play as White. Click on a piece to move."

        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if not self.game_over and self.board.turn == chess.WHITE:  # Human's turn
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
                        x, y = event.pos
                        square = self.square_from_coords(x, y)

                        if square is not None:
                            if self.selected_square is None:
                                # Select a piece
                                piece = self.board.piece_at(square)
                                if piece and piece.color == chess.WHITE:  # Can only select own pieces
                                    self.selected_square = square
                                    # Find legal moves for this piece
                                    self.highlighted_moves = [move for move in self.board.legal_moves
                                                              if move.from_square == square]
                            else:
                                # Try to move to the selected square
                                if self.handle_move(self.selected_square, square):
                                    # Move was successful, check if game is over
                                    self.check_game_over()

                                    if not self.game_over:
                                        # AI's turn
                                        self.message = "AI is thinking..."
                                        # Need to redraw to show the message
                                        self.draw()
                                        pygame.display.flip()

                                        self.make_ai_move()
                                        self.check_game_over()

                                        if not self.game_over:
                                            self.message = "Your turn."
                                else:
                                    # If the clicked square is another one of our pieces, select it instead
                                    piece = self.board.piece_at(square)
                                    if piece and piece.color == chess.WHITE:
                                        self.selected_square = square
                                        self.highlighted_moves = [move for move in self.board.legal_moves
                                                                  if move.from_square == square]
                                    else:
                                        # Invalid move, deselect
                                        self.selected_square = None
                                        self.highlighted_moves = []

                # New game with 'r' key
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.board = chess.Board()  # Reset the board
                        self.selected_square = None
                        self.highlighted_moves = []
                        self.game_over = False
                        self.message = "New game. You play as White. Click on a piece to move."

            # Draw everything
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)  # Limit to 30 FPS

        pygame.quit()

    def draw(self):
        """Draw the complete game state."""
        self.screen.fill((40, 40, 40))  # Dark background
        self.draw_board()
        self.draw_highlights()
        self.draw_pieces()
        self.draw_message()


# This function should be at the module level, not inside a class
def play_with_gui(model_path=None, device='cpu'):
    """
    Launch the chess GUI and play against the trained model.

    Args:
        model_path: Path to the trained model file
        device: Device to run the model on ('cpu' or 'cuda')
    """
    # Load the model if path is provided
    model = None
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")

        # Import here to avoid circular imports
        import torch
        from src.models.network import ChessNetwork

        # Compute the appropriate policy output size
        policy_output_size = 64 * 64 + 64 * 64 * 4

        # Create model with the same architecture as used during training
        model = ChessNetwork(
            input_channels=14,
            num_res_blocks=10,  # Use the same value as in training
            num_filters=128,
            policy_output_size=policy_output_size
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
    else:
        print("No model loaded. Will play randomly.")

    # Start the GUI
    gui = ChessGUI(model=model, device=device)
    gui.run()


if __name__ == "__main__":
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Play chess against trained model with GUI')
    parser.add_argument('--model', type=str, help='Path to trained model file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on (cuda or cpu)')

    args = parser.parse_args()
    play_with_gui(model_path=args.model, device=args.device)