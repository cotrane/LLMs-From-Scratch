import tiktoken
import torch


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Convert a text string to a tensor of token IDs.

    Args:
        text (str): The text to convert.
        tokenizer (tiktoken.Encoding): The tokenizer to use.

    Returns:
        torch.Tensor: A tensor of token IDs.
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # Add the batch dimension as the first dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Convert a tensor of token IDs to a text string.

    Args:
        token_ids (torch.Tensor): The tensor of token IDs to convert.
        tokenizer (tiktoken.Encoding): The tokenizer to use.

    Returns:
        str: The text string.
    """
    # Remove the batch dimension here
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
