import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(
            self,
            csv_file: str,
            tokenizer: tiktoken.Encoding,
            max_length: int | None = None,
            pad_token_id: int = 50256
        ):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["message"]
        ]

        if max_length is not None:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:max_length]
                for encoded_text in self.encoded_texts
            ]
        else:
            self.max_length = self._longest_encoded_length()

        # Pad the encoded texts to the max length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]["label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self) -> int:
        return len(self.encoded_texts)

    def _longest_encoded_length(self) -> int:
        max_length = 0
        for encoded_text in self.encoded_texts:
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)
        return max_length
