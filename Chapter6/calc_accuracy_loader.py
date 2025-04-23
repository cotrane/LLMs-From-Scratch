import torch
from torch.utils.data import DataLoader


def calc_accuracy_loader(
        data_loader: DataLoader, 
        model: torch.nn.Module, 
        device: torch.device, 
        num_batches: int = None
) -> float:
    """Calculate the accuracy of the model on the data loader.

    Args:
        data_loader (DataLoader): The data loader to calculate the accuracy on.
        model (torch.nn.Module): The model to calculate the accuracy on.
        device (torch.device): The device to calculate the accuracy on.
        num_batches (int, optional): The number of batches to calculate the accuracy on. Defaults to None.

    Returns:
        float: The accuracy of the model on the data loader.
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                outputs = model(input_batch)
                logits = outputs[:, -1, :]
                
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples