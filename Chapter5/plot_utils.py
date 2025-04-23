import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator


def plot_losses(
    epochs_seen: torch.Tensor,
    tokens_seen: list[int],
    train_losses: list[float],
    val_losses: list[float]
) -> None:
    """Plot the training and validation losses.

    Args:
        epochs_seen (torch.Tensor): The number of epochs seen.
        tokens_seen (torch.Tensor): The number of tokens seen.
        train_losses (torch.Tensor): The training losses.
        val_losses (torch.Tensor): The validation losses.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs_seen, train_losses, label="Train loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Creates invisible line to align axis
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()
