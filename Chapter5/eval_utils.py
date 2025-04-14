import os
import sys

import tiktoken
import torch
from loss_utils import calc_loss_loader
from tokenize_utils import text_to_token_ids, token_ids_to_text

sys.path.append(os.path.dirname(os.path.abspath(".")))
# pylint: disable=wrong-import-position
from Chapter4.generate_text_simple import generate_text_simple


def evaluate_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Evaluate the model on the train and validation sets.

    Args:
        model (torch.nn.Module): The model to evaluate.
        train_loader (torch.utils.data.DataLoader): The train loader.
        val_loader (torch.utils.data.DataLoader): The validation loader.
        device (torch.device): The device to use.
        eval_iter (int): The number of iterations to evaluate.

    Returns:
        tuple[float, float]: The train and validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
):
    """Generate and print a sample from the model.

    Args:
        model (torch.nn.Module): The model to generate from.
        tokenizer (tiktoken.Encoding): The tokenizer to use.
        device (torch.device): The device to use.
        start_context (str): The context to start the generation from.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
