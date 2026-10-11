import logging

import torch
import tiktoken

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, data, device="cpu"):
        tokens = self.encoder.encode(data)
        tokens = torch.tensor(tokens, device=device)

        return tokens

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        return self.encoder.decode(ids)
