import logging
import os
from pathlib import Path

import numpy as np
import torch
from hellaswag import get_formatted_example, iterate_examples
from tokenizer import Tokenizer
from train import DATASET_PATH, Dataset, console, load_data

logger = logging.getLogger(__name__)


def load_tokens(filepath: str):
    tokens = np.load(filepath)
    tokens = torch.tensor(tokens, dtype=torch.long)

    return tokens


class HellaSwagDataset(Dataset):
    def __init__(self, split="train"):
        self.examples = list(iterate_examples(split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        data, tokens, mask, label = get_formatted_example(example)

        return tokens, mask, label


class CustomDataLoader:
    """Data loader class implementation"""

    def __init__(self, B, T, split="train", dataset="shakespeare"):
        self.B = B
        self.T = T
        self.dataset = dataset

        if dataset == "shakespeare":
            data = load_data(DATASET_PATH)
            self.tokenizer = Tokenizer()
            self.tokens = self.tokenizer.encode(data)
        elif dataset == "fineweb":
            dataset_dir = Path(__file__).parent / "edu_fineweb10B"
            shards = os.listdir(dataset_dir)
            self.shards = [f"{dataset_dir}/{s}" for s in shards if split in s]
            # self.current_shard = 0
            # self.tokens = load_tokens(self.shards[self.current_shard])

            assert len(self.shards) > 0, f"No shards found for split {split}"

        assert split in {"train", "val"}

        self.reset()
        console.print(f"Loaded {len(self.tokens)} tokens")

    def reset(self):
        if self.dataset == "fineweb":
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        self.index = 0

    def next_batch(self):
        buffer = self.tokens[self.index : self.index + self.B * self.T + 1]

        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)

        self.index += self.B * self.T

        if self.index + (self.B * self.T + 1) > len(self.tokens):
            if self.dataset == "fineweb":
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
            self.index = 0

        return x, y
