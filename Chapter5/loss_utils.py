import torch


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Calculate the loss for a batch of input and target tensors.

    Args:
        input_batch (torch.Tensor): The input tensor.
        target_batch (torch.Tensor): The target tensor.
        model (torch.nn.Module): The model to use.
        device (torch.device): The device to use.

    Returns:
        torch.Tensor: The loss.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    logits = logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    *,
    num_batches: int | None = None,
) -> float:
    """Calculate the loss for a loader of input and target tensors.

    Args:
        data_loader (torch.utils.data.DataLoader): The loader of input and target tensors.
        model (torch.nn.Module): The model to use.
        device (torch.device): The device to use.
        num_batches (int | None, optional): The number of batches to use. Defaults to None.

    Returns:
        float: The average loss.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches
