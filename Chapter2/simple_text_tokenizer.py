import re


class SimpleTokenizerV1:
    """Simple tokenizer that uses a vocabulary to encode and decode text.

    Args:
        vocab (dict): Vocabulary mapping strings to integers
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        # Remove spaces before specified punctuation
        text = re.sub(f"\s+([,.?!\"()'])", r"\1", text)
        return text


class SimpleTokenizerV2:
    """Simple tokenizer that uses a vocabulary to encode and decode text. It takes care of
    unknown tokens.

    Args:
        vocab (dict): Vocabulary mapping strings to integers
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        preprocessed = [
            token if token in self.str_to_int else "<|unk|>" for token in preprocessed
        ]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        # Remove spaces before specified punctuation
        text = re.sub(f"\s+([,.?!\"()'])", r"\1", text)
        return text
