import torch
from torch.utils.data import DataLoader, Dataset

import tiktoken


class GPTDatasetV1(Dataset):
    """A PyTorch Dataset that samples sliding windows of text from a given text.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset
        text (str): Text to sample from
        tokenizer (SimpleTokenizerV2): Tokenizer to encode and decode text
        max_lengh (int): Maximum length of the sampled sequence
        stride (int): Stride of the sliding window
    """

    def __init__(
        self, text: str, tokenizer: tiktoken.Encoding, max_lengh: int, stride: int
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_lengh, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_lengh]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_lengh + 1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def created_dataloader_v1(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for the given text and Dataset.

    Args:
        text (str): Text to sample from
        batch_size (int, optional): Batch size. Defaults to 4.
        max_length (int, optional): Maximum length of the sampled sequence. Defaults to 256.
        stride (int, optional): Stride of the sliding window. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it is not full. Defaults to True.
        num_workers (int, optional): Number of workers for loading the data. Defaults to 0.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
