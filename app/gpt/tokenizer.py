import logging

import torch
from tokenizers import tiktoken
from train import DEVICE

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, data):
        tokens = self.encoder.encode(data)
        tokens = torch.tensor(tokens, device=DEVICE)

        return tokens

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        return self.encoder.decode(ids)
