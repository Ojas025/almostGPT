"""Unit tests for CustomDataLoader (plan Component 15).

Contracts: Y is the shift-by-1 target of X; batches stream the token stream
sequentially; the loader advances to the next shard at exhaustion and wraps
after the last shard.
"""

import numpy as np
import pytest
import torch

from src.gpt.config import DataConfig
from src.gpt.data import CustomDataLoader

B, T = 2, 8
BATCH = B * T  # 16 tokens consumed per next_batch() call


@pytest.fixture
def make_loader(tmp_path, synthetic_tokens):
    """Two shards of `size` tokens each in a temp dir; returns (loader, shards)."""

    def make(size: int = 49) -> tuple:
        directory = tmp_path / "shards"
        directory.mkdir(exist_ok=True)
        shards = []
        for i in range(2):
            tokens = synthetic_tokens(size, offset=i * size)
            np.save(directory / f"edufineweb_train_{i:06d}.npy", tokens)
            shards.append(tokens)
        config = DataConfig(dataset="fineweb", path=directory)
        return CustomDataLoader(B, T, config, split="train"), shards

    return make


def test_custom_dataloader_batch_target_shift(make_loader):
    """Y is exactly X shifted by one token across the whole flattened batch."""
    loader, shards = make_loader()
    stream = torch.tensor(shards[0], dtype=torch.long)

    x, y = loader.next_batch()

    assert x.shape == (B, T) and y.shape == (B, T)
    # Reference: X = stream[0 : B*T].view(B,T), Y = stream[1 : B*T+1].view(B,T).
    assert torch.equal(x, stream[: BATCH + 0].view(B, T))
    assert torch.equal(y, stream[1 : BATCH + 1].view(B, T))
    # Explicit per-position shift for every intra-row position.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_custom_dataloader_sequential_coverage(make_loader):
    """Consecutive batches consume the stream with no gaps or re-reads."""
    loader, shards = make_loader()
    stream = torch.tensor(shards[0], dtype=torch.long)

    for batch_idx in range(3):
        x, _ = loader.next_batch()
        expected = stream[batch_idx * BATCH : (batch_idx + 1) * BATCH].view(B, T)
        assert torch.equal(x, expected), f"gap/reorder before batch {batch_idx}"


def test_custom_dataloader_shard_transition(make_loader):
    """Exhausting shard 0 advances to shard 1 and keeps streaming."""
    loader, shards = make_loader(size=49)

    assert loader.current_shard == 0
    # 49 tokens, 16/batch: batches at index 0,16,32 fill the shard. The 3rd
    # batch returns shard 0's tail (tokens 32..48) and flips current_shard.
    loader.next_batch()
    loader.next_batch()
    assert loader.current_shard == 0, "should still be inside shard 0"

    tail_x, _ = loader.next_batch()
    assert loader.current_shard == 1, "loader must advance to the next shard"
    shard0 = torch.tensor(shards[0], dtype=torch.long)
    assert torch.equal(tail_x, shard0[2 * BATCH : 49 - 1].view(B, T))

    # The next batch comes from shard 1's head.
    x, _ = loader.next_batch()
    shard1 = torch.tensor(shards[1], dtype=torch.long)
    assert torch.equal(x, shard1[:BATCH].view(B, T))


def test_custom_dataloader_wrap_around(make_loader):
    """After the last shard, iteration wraps back to shard 0."""
    loader, shards = make_loader(size=49)

    # 3 batches per shard: after the 6th (shard 1's tail) the loader wraps.
    for _ in range(5):
        loader.next_batch()
    assert loader.current_shard == 1

    loader.next_batch()
    assert loader.current_shard == 0, "must wrap to the first shard"

    x, _ = loader.next_batch()
    shard0 = torch.tensor(shards[0], dtype=torch.long)
    assert torch.equal(x, shard0[:BATCH].view(B, T))


def test_custom_dataloader_reset(make_loader):
    """reset() returns the loader to shard 0, index 0."""
    loader, _ = make_loader()
    loader.next_batch()
    loader.next_batch()

    loader.reset()

    assert loader.current_shard == 0
    assert loader.index == 0


def test_custom_dataloader_generic_shard_size_no_token_loss():
    """A batch-unglued shard must not silently drop its tail (plan 2.15/6.4).

    The other tests size shards as `len % (B*T) == 1` so full coverage holds;
    this companion uses a generic size (40 % 16 == 8) to verify the loader
    does not skip the sub-batch tail when the shard ends mid-window.
    """
    import tempfile
    from pathlib import Path

    B, T = 2, 8
    batch_tokens = B * T  # 16
    shard_len = 40        # generic: not batch-aligned

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        tokens = ((np.arange(shard_len) % 60) + 1).astype(np.uint16)
        np.save(directory / "edufineweb_train_000000.npy", tokens)
        np.save(directory / "edufineweb_train_000001.npy", tokens.copy())
        config = DataConfig(dataset="fineweb", path=directory)
        loader = CustomDataLoader(B, T, config, split="train")

        covered: set[int] = set()
        batch_idx = 0
        while loader.current_shard == 0:
            start = batch_idx * batch_tokens
            x, y = loader.next_batch()
            covered.update(range(start, start + batch_tokens))          # X tokens
            covered.update(range(start + 1, start + batch_tokens + 1))  # Y tokens
            batch_idx += 1

    missing = sorted(set(range(shard_len)) - covered)
    assert not missing, (
        f"tokens {missing} of the {shard_len}-token shard were never emitted -- "
        "the loader drops the sub-batch tail when advancing to the next shard"
    )
