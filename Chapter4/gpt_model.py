import os
import sys

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(".")))
# pylint: disable=wrong-import-position
from Chapter3.multihead_attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """
    LayerNorm that applies the same normalization to each position in the sequence.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        """
        Args:
            emb_dim: The dimension of the embeddings.
            eps: A small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LayerNorm.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: The normalized tensor of shape (batch_size, seq_len, emb_dim).
        """
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False means we divide by N instead of N-1
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return out * self.scale + self.shift


class GELU(nn.Module):
    """
    GELU activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GELU activation function.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, emb_dim).
        """
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """
    FeedForward network.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration of the FeedForward network.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                config["emb_dim"], 4 * config["emb_dim"]
            ),  # Increase the hidden dimension
            GELU(),
            nn.Linear(
                4 * config["emb_dim"], config["emb_dim"]
            ),  # Decrease the output dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, emb_dim).
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer block.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration of the Transformer block.
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )
        self.feed_forward = FeedForward(config)
        self.ln1 = LayerNorm(config["emb_dim"])
        self.ln2 = LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, emb_dim).
        """
        # Apply the attention block with shortcut connection
        shortcut = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Apply the feed-forward network with shortcut connection
        shortcut = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    GPT model.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration of the GPT model.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT model.

        Args:
            in_idx (torch.Tensor): The input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape  # pylint: disable=unused-variable
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
