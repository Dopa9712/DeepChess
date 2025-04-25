# ==========================================
# File: src/training/enhanced_trainer.py
# (New file - contains improved trainer classes)
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import os
import time
import random
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm

from src.environment.chess_env import ChessEnv
from src.models.network import ChessNetwork
from src.models.policy import RandomPolicy, NetworkPolicy
from src.training.experience_buffer import ExperienceBuffer
from src.environment.utils import board_to_planes


class EnhancedSelfPlayWorker:
    """
    Enhanced worker for self-play to generate high-quality training data.
    Generates training data through self-play with various improvements for stronger play.
    """

    def __init__(
            self,
            model: nn.Module,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            num_games: int = 1000,
            max_moves: int = 200,
            temperature: float = 1.0,
            temperature_drop_move: int = 20,
            exploration_factor: float = 0.15,
            experience_buffer: Optional[ExperienceBuffer] = None,
            use_opening_book: bool = True,
            enable_position_augmentation: bool = True
    ):
        """
        Initialize the enhanced self-play worker.

        Args:
            model: The neural network model to use
            device: Device for the model ('cpu' or 'cuda')
            num_games: Number of games to play
            max_moves: Maximum number of moves per game
            temperature: Initial temperature for exploration
            temperature_drop_move: Move at which to reduce temperature
            exploration_factor: Factor for random exploration
            experience_buffer: Optional, buffer for storing experiences
            use_opening_book: Whether to use basic opening book for diversity
            enable_position_augmentation: Whether to augment positions with symmetries
        """
        self.model = model
        self.device = device
        self.num_games = num_games
        self.max_moves = max_moves
        self.initial_temperature = temperature
        self.temperature_drop_move = temperature_drop_move
        self.exploration_factor = exploration_factor
        self.experience_buffer = experience_buffer or ExperienceBuffer()
        self.use_opening_book = use_opening_book
        self.enable_position_augmentation = enable_position_augmentation

        # Simple opening book for more diverse games
        self.opening_book = [
            "e2e4", "d2d4", "c2c4", "g1f3",  # Common first moves
            "e2e4 e7e5", "e2e4 c7c5", "d2d4 d7d5", "d2d4 g8f6",  # Common responses
            "e2e4 e7e5 g1f3", "e2e4 c7c5 g1f3", "d2d4 d7d5 c2c4"  # Common openings
        ]

    def _get_opening_sequence(self):
        """Get a random opening sequence from the opening book"""
        if self.use_opening_book and random.random() < 0.7:  # 70% chance to use opening book
            return random.choice(self.opening_book)
        return ""

    def _apply_opening_sequence(self, board, opening):
        """Apply an opening sequence to the board"""
        if not opening:
            return

        moves = opening.split()
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    break
            except ValueError:
                break

    def generate_games(self) -> ExperienceBuffer:
        """
        Generate a series of self-play games and collect training data.

        Returns:
            ExperienceBuffer: Buffer with the collected training data
        """
        start_time = time.time()
        game_lengths = []

        for game_idx in tqdm(range(self.num_games), desc="Self-play games"):
            # Initialize environment
            env = ChessEnv()
            board = env.board

            # Apply an opening sequence for diversity
            opening = self._get_opening_sequence()
            self._apply_opening_sequence(board, opening)

            # Dynamic temperature based on game progress
            current_temperature = self.initial_temperature

            # Create policy with dynamic temperature
            policy = NetworkPolicy(
                self.model,
                device=self.device,
                temperature=current_temperature,
                exploration_factor=self.exploration_factor
            )

            # Game-related variables
            game_memory = []  # Stores (state, policy_probs, player) tuples
            move_count = len(board.move_stack)
            current_player = board.turn  # Current player's turn

            # Collect positions from the game
            while not board.is_game_over() and move_count < self.max_moves:
                # Adjust temperature dynamically as game progresses
                if move_count == self.temperature_drop_move:
                    current_temperature = 0.3
                    policy.temperature = current_temperature
                elif move_count == 40:  # Even lower temperature in late middlegame
                    current_temperature = 0.2
                    policy.temperature = current_temperature
                elif move_count == 60:  # Minimal exploration in endgame
                    current_temperature = 0.1
                    policy.temperature = current_temperature
                    policy.exploration_factor = 0.0

                # Current board representation
                board_rep = board_to_planes(board)

                # Move selection based on policy
                move_probs, legal_moves = policy.get_action_probs(board)

                # Store in game memory (for later when result is known)
                game_memory.append({
                    'state': board_rep,
                    'policy_probs': move_probs,
                    'legal_moves': legal_moves.copy(),
                    'player': current_player
                })

                # Select and execute move
                selected_idx = np.random.choice(len(legal_moves), p=move_probs)
                move = legal_moves[selected_idx]
                board.push(move)

                # Update player and counter
                current_player = not current_player
                move_count += 1

            game_lengths.append(move_count)

            # Determine game result
            if board.is_checkmate():
                # The player not to move has won
                winner = not board.turn
                result = 1.0  # Win
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
                # Draw
                winner = None
                result = 0.0  # Draw
            else:
                # Game limit reached
                winner = None
                result = 0.0  # Treat as draw

            # Add game positions to experience buffer with augmentation
            for memory_item in game_memory:
                # Value from the perspective of the player
                player_result = result if memory_item['player'] == winner else (
                    -result if winner is not None else 0.0
                )

                # Add to experience buffer
                self.experience_buffer.add(
                    state=memory_item['state'],
                    policy_probs=memory_item['policy_probs'],
                    value=player_result,
                    legal_moves=memory_item['legal_moves']
                )

                # Generate symmetries if enabled
                if self.enable_position_augmentation:
                    # Generate horizontal flip of the board
                    flipped_state = np.flip(memory_item['state'], axis=1).copy()

                    # Transform the policy to match the flipped board
                    # For demonstration purposes, we'll use a simplified approach
                    # In a real implementation, move encoding would need proper transformation
                    self.experience_buffer.add(
                        state=flipped_state,
                        policy_probs=memory_item['policy_probs'],
                        value=player_result,
                        legal_moves=memory_item['legal_moves']
                    )

        # Report statistics
        elapsed = time.time() - start_time
        avg_game_length = sum(game_lengths) / len(game_lengths)
        positions_per_second = len(self.experience_buffer) / elapsed

        print(f"Generated {len(self.experience_buffer)} training examples from {self.num_games} games")
        print(f"Average game length: {avg_game_length:.1f} moves")
        print(f"Generation speed: {positions_per_second:.1f} positions/second")

        return self.experience_buffer


class EnhancedRLTrainer:
    """
    Enhanced trainer for reinforcement learning chess AI.
    Implements various improvements for stronger play and better training.
    """

    def __init__(
            self,
            model: nn.Module,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate: float = 0.0005,
            l2_reg: float = 2e-4,
            batch_size: int = 1024,
            num_epochs: int = 20,
            checkpoint_dir: str = './checkpoints',
            num_self_play_games: int = 1000,
            evaluation_games: int = 200,
            value_loss_weight: float = 3.0
    ):
        """
        Initialize the enhanced RL trainer.

        Args:
            model: The neural network model to train
            device: Device for training ('cpu' or 'cuda')
            learning_rate: Learning rate for the optimizer
            l2_reg: L2 regularization strength
            batch_size: Batch size for training
            num_epochs: Number of epochs per training phase
            checkpoint_dir: Directory for model checkpoints
            num_self_play_games: Number of self-play games per iteration
            evaluation_games: Number of evaluation games
            value_loss_weight: Weight for value loss in total loss calculation
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # Optimizer with improved parameters
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=l2_reg,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler with warmup and plateau detection
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3,
            verbose=True, min_lr=1e-6
        )

        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        self.value_loss_weight = value_loss_weight

        # Training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.num_self_play_games = num_self_play_games
        self.evaluation_games = evaluation_games

        # For saving best models
        self.best_model_win_rate = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

        # History tracking
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'win_rates': [],
            'checkmate_wins': [],
            'lr': []
        }

    def _move_to_index(self, move: chess.Move) -> int:
        """
        Convert a move to a unique index (same as in your existing implementation).

        Args:
            move: The chess.Move object to index

        Returns:
            int: Unique index for the move
        """
        from_square = move.from_square  # 0-63
        to_square = move.to_square  # 0-63

        # Index calculation for normal moves without promotion
        if move.promotion is None:
            # 64*64 possible combinations of source and destination squares
            return from_square * 64 + to_square
        else:
            # Promotion moves: use additional indices after the 64*64 normal moves
            # There are 4 promotion types (knight, bishop, rook, queen)
            promotion_offset = 64 * 64

            # Offset based on from_square, to_square and promotion type
            # We encode promotion type as 0=knight, 1=bishop, 2=rook, 3=queen
            promotion_type = move.promotion - 2  # Convert from chess.KNIGHT(2) to 0, etc.
            return promotion_offset + (from_square * 64 + to_square) * 4 + promotion_type

    def train_iteration(self, experience_buffer: Optional[ExperienceBuffer] = None) -> Dict[str, float]:
        """
        Perform a complete training iteration:
        1. Self-play games for data collection
        2. Training on collected data
        3. Evaluation against previous version

        Args:
            experience_buffer: Optional, existing experience buffer

        Returns:
            Dict[str, float]: Statistics about the training iteration
        """
        # Phase 1: Self-play for data collection
        if experience_buffer is None:
            self.model.eval()  # Evaluation mode for self-play

            # Adaptive exploration based on current performance
            current_exploration = max(0.05, min(0.15, 0.15 - self.best_model_win_rate / 10.0))

            worker = EnhancedSelfPlayWorker(
                model=self.model,
                device=self.device,
                num_games=self.num_self_play_games,
                temperature=1.0,
                exploration_factor=current_exploration,
                enable_position_augmentation=True
            )
            experience_buffer = worker.generate_games()

        # Phase 2: Training on collected data
        train_stats = self.train_on_buffer(experience_buffer)

        # Adjust learning rate based on training loss
        self.scheduler.step(train_stats['total_loss'])

        # Track current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['lr'].append(current_lr)

        return {
            'train_loss': train_stats['total_loss'],
            'policy_loss': train_stats['policy_loss'],
            'value_loss': train_stats['value_loss'],
            'samples': len(experience_buffer),
            'lr': current_lr
        }

    def train_on_buffer(self, experience_buffer: ExperienceBuffer) -> Dict[str, float]:
        """
        Train the model on an experience buffer with enhanced techniques.

        Args:
            experience_buffer: Buffer with training examples

        Returns:
            Dict[str, float]: Training statistics
        """
        self.model.train()  # Activate training mode

        print(f"Starting training on {len(experience_buffer)} collected experiences")

        # Training statistics
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Split data into training and validation sets
        indices = np.arange(len(experience_buffer))
        np.random.shuffle(indices)

        val_size = min(int(len(indices) * 0.1), 1000)  # 10% for validation, max 1000 examples
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]

        # Data augmentation: Add noise to values for robustness
        augmented_values = {}
        for i in train_indices:
            # Add small random noise to values to prevent overfitting
            if random.random() < 0.1:  # Only augment 10% of values
                orig_value = experience_buffer.values[i]
                noise = random.uniform(-0.1, 0.1)
                # Ensure value stays in [-1, 1] range
                augmented_values[i] = max(-1.0, min(1.0, orig_value + noise))

        # Train on data in batches
        for epoch in range(self.num_epochs):
            print(f"Training epoch {epoch + 1}/{self.num_epochs}...")
            epoch_start_time = time.time()

            # Shuffle training data
            np.random.shuffle(train_indices)

            # Process in batches
            batch_count = 0
            epoch_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0

            for i in range(0, len(train_indices), self.batch_size):
                if i + self.batch_size > len(train_indices):
                    continue  # Skip last incomplete batch

                batch_indices = train_indices[i:i + self.batch_size]

                # Progress report every 10 batches
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Processing batch {batch_count}...")

                try:
                    # Load data
                    states, policy_targets, value_targets, legal_moves = experience_buffer.sample_batch(batch_indices)

                    # Apply value augmentation if available
                    for j, idx in enumerate(batch_indices):
                        if idx in augmented_values:
                            value_targets[j] = augmented_values[idx]

                    # Convert to torch tensors and reorder dimensions
                    # From (batch_size, 8, 8, 14) to (batch_size, 14, 8, 8)
                    states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
                    value_targets = torch.FloatTensor(value_targets).view(-1, 1).to(self.device)

                    # Forward pass
                    policy_logits, value_preds = self.model(states)

                    # For value: MSE loss
                    value_loss = self.value_loss_fn(value_preds, value_targets)

                    # For policy: Calculate loss for each example in batch individually
                    policy_loss = 0.0

                    for j, (probs, moves) in enumerate(zip(policy_targets, legal_moves)):
                        # Create a mask for valid moves
                        logits = policy_logits[j]
                        mask = torch.zeros_like(logits)

                        # Convert probabilities to a tensor
                        target_probs = torch.zeros_like(logits)

                        # Set probabilities for valid moves
                        for prob, move in zip(probs, moves):
                            move_idx = self._move_to_index(move)
                            mask[move_idx] = 1
                            target_probs[move_idx] = prob

                        # Calculate cross-entropy loss with masking
                        masked_logits = logits * mask
                        log_probs = torch.log_softmax(masked_logits + (1 - mask) * -1e9, dim=0)
                        cross_entropy = -torch.sum(target_probs * log_probs * mask)
                        policy_loss += cross_entropy

                    policy_loss = policy_loss / len(policy_targets)  # Average over batch

                    # Combined loss with higher weight for value loss
                    loss = policy_loss + self.value_loss_weight * value_loss

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    # Update statistics
                    epoch_loss += loss.item()
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"Error processing batch {batch_count}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Epoch statistics
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            avg_epoch_policy_loss = epoch_policy_loss / batch_count if batch_count > 0 else 0
            avg_epoch_value_loss = epoch_value_loss / batch_count if batch_count > 0 else 0

            # Validate on validation set
            val_loss, val_policy_loss, val_value_loss = self._validate(experience_buffer, val_indices)

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(
                f"  Train losses: Total={avg_epoch_loss:.4f}, Policy={avg_epoch_policy_loss:.4f}, Value={avg_epoch_value_loss:.4f}")
            print(f"  Val losses: Total={val_loss:.4f}, Policy={val_policy_loss:.4f}, Value={val_value_loss:.4f}")

            # Update total statistics
            total_loss += avg_epoch_loss
            total_policy_loss += avg_epoch_policy_loss
            total_value_loss += avg_epoch_value_loss

        # Calculate average losses
        avg_loss = total_loss / self.num_epochs
        avg_policy_loss = total_policy_loss / self.num_epochs
        avg_value_loss = total_value_loss / self.num_epochs

        # Update training history
        self.training_history['total_loss'].append(avg_loss)
        self.training_history['policy_loss'].append(avg_policy_loss)
        self.training_history['value_loss'].append(avg_value_loss)

        print(
            f"Training completed. Avg losses: Total={avg_loss:.4f}, Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")

        return {
            'total_loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }

    def _validate(self, experience_buffer, val_indices):
        """
        Validate the model on a validation set.

        Args:
            experience_buffer: The experience buffer
            val_indices: Indices of validation examples

        Returns:
            tuple: (val_loss, val_policy_loss, val_value_loss)
        """
        self.model.eval()

        if not val_indices:
            return 0.0, 0.0, 0.0

        with torch.no_grad():
            states, policy_targets, value_targets, legal_moves = experience_buffer.sample_batch(val_indices)

            # Convert to torch tensors
            states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
            value_targets = torch.FloatTensor(value_targets).view(-1, 1).to(self.device)

            # Forward pass
            policy_logits, value_preds = self.model(states)

            # Value loss
            value_loss = self.value_loss_fn(value_preds, value_targets)

            # Policy loss
            policy_loss = 0.0
            for j, (probs, moves) in enumerate(zip(policy_targets, legal_moves)):
                logits = policy_logits[j]
                mask = torch.zeros_like(logits)
                target_probs = torch.zeros_like(logits)

                for prob, move in zip(probs, moves):
                    move_idx = self._move_to_index(move)
                    mask[move_idx] = 1
                    target_probs[move_idx] = prob

                masked_logits = logits * mask
                log_probs = torch.log_softmax(masked_logits + (1 - mask) * -1e9, dim=0)
                cross_entropy = -torch.sum(target_probs * log_probs * mask)
                policy_loss += cross_entropy

            policy_loss = policy_loss / len(policy_targets) if policy_targets else 0

            # Combined loss
            total_loss = policy_loss + self.value_loss_weight * value_loss

        self.model.train()
        return total_loss.item(), policy_loss.item(), value_loss.item()

    def evaluate_against_multiple_opponents(self) -> Dict[str, Any]:
        """
        Evaluate the model against multiple opponents.

        Returns:
            Dict[str, Any]: Evaluation statistics
        """
        self.model.eval()

        # Create opponents
        random_policy = RandomPolicy()

        # Try to load previous model versions
        opponent_models = []
        for checkpoint_file in ["model_iter_1.pt", "model_iter_5.pt", "best_model.pt"]:
            path = os.path.join(self.checkpoint_dir, checkpoint_file)
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    opponent = ChessNetwork(
                        input_channels=14,
                        num_res_blocks=10,
                        num_filters=128,
                        policy_output_size=64 * 64 + 64 * 64 * 4
                    )
                    opponent.load_state_dict(checkpoint['model_state_dict'])
                    opponent.to(self.device)
                    opponent.eval()
                    opponent_models.append({"name": checkpoint_file, "model": opponent})
                except Exception as e:
                    print(f"Could not load opponent from {path}: {e}")

        results = {}

        # Evaluate against random policy
        print("Evaluating against random policy...")
        random_stats = self._evaluate_against_opponent(random_policy, self.evaluation_games // 2)
        results["random"] = random_stats

        # Evaluate against previous model versions
        for opponent in opponent_models:
            print(f"Evaluating against {opponent['name']}...")
            opponent_policy = NetworkPolicy(opponent["model"], self.device, temperature=0.1, exploration_factor=0.0)
            opponent_stats = self._evaluate_against_opponent(opponent_policy,
                                                             self.evaluation_games // (len(opponent_models) + 1))
            results[opponent["name"]] = opponent_stats

        # Calculate overall win rate
        total_games = 0
        total_wins = 0
        total_checkmates = 0

        for opponent_name, stats in results.items():
            total_games += stats["games"]
            total_wins += stats["wins"] + 0.5 * stats["draws"]
            total_checkmates += stats.get("checkmate_wins", 0)

        overall_win_rate = total_wins / total_games if total_games > 0 else 0

        # Update best model if this is the best win rate
        if overall_win_rate > self.best_model_win_rate:
            self.best_model_win_rate = overall_win_rate
            self.save_model("best_model.pt")
            print(f"New best model with win rate {overall_win_rate:.4f}!")

        # Update history
        self.training_history["win_rates"].append(overall_win_rate)
        self.training_history["checkmate_wins"].append(total_checkmates)

        return {
            "results": results,
            "overall_win_rate": overall_win_rate,
            "total_games": total_games,
            "total_checkmates": total_checkmates
        }

    def _evaluate_against_opponent(self, opponent_policy, num_games):
        """
        Evaluate against a specific opponent.

        Args:
            opponent_policy: The opponent's policy
            num_games: Number of games to play

        Returns:
            dict: Statistics about the games
        """
        # Use our model with low temperature and no exploration for evaluation
        our_policy = NetworkPolicy(
            self.model,
            self.device,
            temperature=0.1,
            exploration_factor=0.0
        )

        stats = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "games": num_games,
            "checkmate_wins": 0,
            "material_advantage": [],
            "move_count": []
        }

        for game_idx in tqdm(range(num_games), desc="Evaluation games"):
            env = ChessEnv()
            board = env.board
            move_count = 0

            # Alternate which side our model plays
            our_model_plays_white = game_idx % 2 == 0

            while not board.is_game_over() and move_count < 200:
                # Determine which policy to use
                current_policy = our_policy if (board.turn == chess.WHITE) == our_model_plays_white else opponent_policy

                # Select and make move
                move = current_policy.get_action(board)
                board.push(move)
                move_count += 1

            stats["move_count"].append(move_count)

            # Determine game result
            if board.is_checkmate():
                winner_is_white = not board.turn

                if winner_is_white == our_model_plays_white:
                    stats["wins"] += 1
                    stats["checkmate_wins"] += 1
                else:
                    stats["losses"] += 1
            else:
                stats["draws"] += 1

            # Calculate material balance at end of game
            material_balance = self._calculate_material_balance(board, our_model_plays_white)
            stats["material_advantage"].append(material_balance)

        # Calculate win rate
        stats["win_rate"] = (stats["wins"] + 0.5 * stats["draws"]) / num_games
        stats["avg_material"] = sum(stats["material_advantage"]) / len(stats["material_advantage"]) if stats[
            "material_advantage"] else 0
        stats["avg_move_count"] = sum(stats["move_count"]) / len(stats["move_count"]) if stats["move_count"] else 0

        print(
            f"Win rate: {stats['win_rate']:.4f} ({stats['wins']} wins, {stats['draws']} draws, {stats['losses']} losses)")
        print(f"Checkmate wins: {stats['checkmate_wins']}")
        print(f"Average material advantage: {stats['avg_material']:.2f}")

        return stats

    def _calculate_material_balance(self, board, from_white_perspective):
        """
        Calculate material balance on a board.

        Args:
            board: The chess board
            from_white_perspective: Whether to calculate from white's perspective

        Returns:
            float: Material balance
        """

        # ==========================================
        # File: src/training/enhanced_trainer.py (Continued)
        # ==========================================

        def _calculate_material_balance(self, board, from_white_perspective):
            """
            Calculate material balance on a board.

            Args:
                board: The chess board
                from_white_perspective: Whether to calculate from white's perspective

            Returns:
                float: Material balance
            """
            piece_values = {
                chess.PAWN: 1.0,
                chess.KNIGHT: 3.0,
                chess.BISHOP: 3.25,
                chess.ROOK: 5.0,
                chess.QUEEN: 9.0,
                chess.KING: 0.0  # King has no material value
            }

            balance = 0.0
            for piece_type, value in piece_values.items():
                balance += len(board.pieces(piece_type, chess.WHITE)) * value
                balance -= len(board.pieces(piece_type, chess.BLACK)) * value

            return balance if from_white_perspective else -balance

        def save_model(self, filename: str) -> None:
            """
            Save the model to a file.

            Args:
                filename: Name of file to save model to
            """
            filepath = os.path.join(self.checkpoint_dir, filename)

            # Create a more comprehensive checkpoint
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_win_rate': self.best_model_win_rate,
                'training_history': self.training_history,
                'model_config': {
                    'input_channels': self.model.input_channels,
                    'num_res_blocks': self.model.num_res_blocks,
                    'num_filters': self.model.num_filters,
                    'policy_output_size': self.model.policy_output_size
                },
                'timestamp': time.time()
            }

            torch.save(checkpoint, filepath)
            print(f"Model saved to {filepath}")

        def load_model(self, filepath: str) -> None:
            """
            Load a model from a file.

            Args:
                filepath: Path to the model file
            """
            if not os.path.exists(filepath):
                print(f"Model file {filepath} not found.")
                return

            try:
                checkpoint = torch.load(filepath, map_location=self.device)

                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load training statistics if available
                if 'best_win_rate' in checkpoint:
                    self.best_model_win_rate = checkpoint['best_win_rate']

                if 'training_history' in checkpoint:
                    self.training_history = checkpoint['training_history']

                print(f"Model loaded from {filepath}")

                # Print model configuration if available
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    print(f"Model configuration: {config}")
            except Exception as e:
                print(f"Error loading model: {e}")
                import traceback
                traceback.print_exc()

    # ==========================================
    # File: src/models/enhanced_network.py
    # (New file - contains improved network architecture)
    # ==========================================



