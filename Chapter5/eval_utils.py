import os
import sys

import tiktoken
import torch

sys.path.append(os.path.dirname(os.path.abspath(".")))
# pylint: disable=wrong-import-position
from Chapter4.generate_text_simple import generate_text_simple
from Chapter5.loss_utils import calc_loss_batch, calc_loss_loader
from Chapter5.tokenize_utils import text_to_token_ids, token_ids_to_text


def train_model_simple(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding
) -> tuple[list[float], list[float], list[int]]:
    """Simple training loop for the model.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The train loader.
        val_loader (torch.utils.data.DataLoader): The validation loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use.
        num_epochs (int): The number of epochs to train.
        eval_freq (int): The frequency of evaluation.
        eval_iter (int): The number of iterations to evaluate.
        start_context (str): The context to start the generation from.
        tokenizer (tiktoken.Encoding): The tokenizer.

    Returns:
        tuple[list[float], list[float], list[int]]: The train and validation loss and the number 
            of tokens seen.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss: {train_loss:.3f}, "
                    f"Val loss: {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


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


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None
) -> torch.Tensor:
    """Generate text from the model.

    Args:
        model (torch.nn.Module): The model to generate from.
        idx (torch.Tensor): The input tensor.
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The context size.
        temperature (float, optional): The temperature. Defaults to 0.0.
        top_k (int | None, optional): The top k. Defaults to None.
        eos_id (int | None, optional): The end of sentence id. Defaults to None.

    Returns:
        torch.Tensor: The generated text.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(-float("inf")),
                other=logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=-1)

    return idx


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
