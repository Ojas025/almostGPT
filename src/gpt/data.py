import logging
import os
from pathlib import Path

import numpy as np
import torch
from .config import SHAKESPEARE_PATH, SHARD_DIR, DataConfig
from ..dataset.hellaswag import get_formatted_example, iterate_examples
from .tokenizer import Tokenizer
from torch.utils.data import Dataset
from .console import console

logger = logging.getLogger(__name__)


def load_data(path: Path) -> str:
    """Read a raw-text dataset from disk.

    Lives here rather than in `train.py`: `train.py` imports this module, so importing it
    back would close a cycle and make `data` unimportable on its own.
    """
    with open(path, "r") as file:
        text = file.read()

    return text


def load_tokens(filepath: str | Path) -> torch.Tensor:
    tokens = np.load(filepath)
    tokens = torch.tensor(tokens, dtype=torch.long)

    return tokens


class HellaSwagDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        self.examples = list(iterate_examples(split))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        example = self.examples[index]
        data, tokens, mask, label = get_formatted_example(example)

        return tokens, mask, label


class CustomDataLoader:
    """Token-stream batch sampler over a sharded or single-file dataset."""

    def __init__(self, B: int, T: int, data_config: DataConfig, split: str = "train") -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.B = B
        self.T = T
        self.split = split
        self.data_config = data_config
        self.dataset = data_config.dataset
        self.shards: list[Path] = []
        self.current_shard: int | None = None

        if self.dataset == "shakespeare":
            if data_config.path is None:
                data_config.path = SHAKESPEARE_PATH

            if not data_config.path.is_file():
                raise FileNotFoundError(
                    f"Shakespeare corpus not found: {data_config.path}. "
                    "Set data.path in config.yml."
                )

            self.tokenizer = Tokenizer()
            self.tokens = self.tokenizer.encode(load_data(data_config.path))
        else:
            directory = data_config.path or SHARD_DIR

            if not directory.is_dir():
                raise FileNotFoundError(
                    f"FineWeb shard directory not found: {directory}. "
                    "Run `python -m src.dataset.fineweb` to build it first."
                )

            # Sorted so shard order is reproducible across machines -- os.listdir order
            # would make a resumed run replay a different data order than the original.
            self.shards = sorted(directory.glob(f"*{split}*.npy"))

            if not self.shards:
                raise FileNotFoundError(f"No '{split}' shards in {directory}")

        self.reset()
        console.print(f"Loaded {len(self.tokens)} tokens")

    def reset(self) -> None:
        if self.dataset == "fineweb":
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])

        self.index = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next (inputs, targets) pair, advancing or wrapping as needed."""
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