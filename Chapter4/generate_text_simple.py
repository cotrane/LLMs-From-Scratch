import torch

from Chapter4.gpt_model import GPTModel


def generate_text_simple(
    model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """Generate text using the model.

    Args:
        model (GPTModel): The model to use for generation.
        idx (torch.Tensor): The initial tokens to use for generation.
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The size of the context window to use for generation.

    Returns:
        torch.Tensor: The generated text.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Subset on the last time step which has the logits of the next predicted token
        logits = logits[:, -1, :]  # [batch_size, vocab_size]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(
            probs, dim=-1, keepdim=True
        )  # Picks the most likely token
        idx = torch.cat(
            (idx, idx_next), dim=1
        )  # And appends it to the running sequence

    return idx
