import os

from matplotlib import pyplot as plt


def create_training_plots(trainer):
    """
    Create visualizations of training progress.

    Args:
        trainer: The RL trainer with training history
    """
    history = trainer.training_history

    # Create output directory
    os.makedirs('plots', exist_ok=True)

    # Plot loss curves
    plt.figure(figsize=(12, 8))

    epochs = range(1, len(history['total_loss']) + 1)

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['total_loss'], 'b-', label='Total Loss')
    plt.plot(epochs, history['policy_loss'], 'r-', label='Policy Loss')
    plt.plot(epochs, history['value_loss'], 'g-', label='Value Loss')
    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot win rates
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['win_rates'], 'g-')
    plt.title('Win Rate Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1.0)
    plt.grid(True)

    # Plot checkmate wins
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['checkmate_wins'], 'm-')
    plt.title('Checkmate Wins')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Checkmates')
    plt.grid(True)

    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['lr'], 'b-')
    plt.title('Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    timestamp = int(time.time())
    plt.savefig(f'plots/training_progress_{timestamp}.png')
    plt.close()

    print(f"Training plots saved to plots/training_progress_{timestamp}.png")

# ==========================================
