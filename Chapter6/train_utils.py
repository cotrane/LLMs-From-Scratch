
import torch
from torch.utils.data import DataLoader

from Chapter5.loss_utils import calc_loss_batch, calc_loss_loader
from Chapter6.calc_accuracy_loader import calc_accuracy_loader


def train_classifier_simple(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        eval_freq: int,
        eval_iter: int
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """Train a simple classifier.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on.
        num_epochs (int): The number of epochs to train.
        eval_freq (int): The frequency of evaluation.
        eval_iter (int): The number of batches to evaluate on.

    Returns:
        tuple[list[float], list[float], list[float], list[float], int]: A tuple containing 
            the training losses, the validation losses, the training accuracies, 
            the validation accuracies, and the number of examples seen.
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device
            )
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss: {train_loss:.4f}, "
                      f"Val loss: {val_loss:.4f}")

        train_acc = calc_accuracy_loader(
            data_loader=train_loader,
            model=model,
            device=device,
            num_batches=eval_iter
        )
        val_acc = calc_accuracy_loader(
            data_loader=val_loader,
            model=model,
            device=device,
            num_batches=eval_iter
        )

        print(f"Training accuracy: {train_acc*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_acc*100:.2f}%")

        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        eval_iter: int
) -> tuple[float, float]:
    """Evaluate the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The device to evaluate on.
        eval_iter (int): The number of batches to evaluate on.

    Returns:
        tuple[float, float]: A tuple containing the training loss and the validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            data_loader=train_loader,
            model=model,
            device=device,
            num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            data_loader=val_loader,
            model=model,
            device=device,
            num_batches=eval_iter
        )
    model.train()

    return train_loss, val_loss
