import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        """
        Args:
            d_in (int): The dimension of the input.
            d_out (int): The dimension of the output.
            context_length (int): The length of the context.
            dropout (float): The dropout rate.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        query = self.W_query(x)

        # Split the last dimension of the weighs into multiple heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        query = query.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose num_heads and num_tokens
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        query = query.transpose(1, 2)

        # Compute the dot prodcut for each head and apply causal mask
        attention_scores = query @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize the attention scores and apply dropout
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # Compute the context vector, flip back num_heads and num_tokens and combine
        # the heads again
        context_vector = attention_weights @ values
        context_vector = context_vector.transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)

        # Add optional output projection
        context_vector = self.out_proj(context_vector)

        return context_vector
