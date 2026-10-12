"""E2E: FineWeb streaming -> tokenize -> shard -> dataloader -> GPT step (plan Phase 6, item 2).

Drives the real ingestion code over fake streamed documents, then trains one
optimizer step on the resulting shards.
"""

import numpy as np
import pytest
import torch

from src.gpt.config import AttentionConfig, DataConfig, ModelConfig, MoEConfig, TrainConfig
from src.gpt.data import CustomDataLoader
from src.gpt.gpt import GPT

B, T = 2, 8

DOCS = [
    "Language modelling predicts the next token given a context window.",
    "FineWeb is a filtered web corpus for language model pretraining.",
    "Tokenization splits text into subword units an embedding table can index.",
    "Shards let training stream a corpus larger than memory.",
    "Gradient descent nudges weights against the cross entropy objective.",
    "Attention lets every position attend to every previous position.",
]


def test_e2e_fineweb_pipeline(fineweb_pipeline):
    """Mock docs -> real tokenize/shard -> dataloader -> forward + backward."""
    cache_dir = fineweb_pipeline(DOCS, shard_size=49)

    shard_files = list(cache_dir.glob("*train*.npy"))
    assert shard_files, "pipeline must produce train shards"

    # Sanity: shards are uint16 token streams on disk.
    for f in shard_files:
        assert np.load(f).dtype == np.uint16

    config = ModelConfig(
        context_length=32,
        vocab_size=50304,  # real GPT-2 token ids flow through here
        n_layer=2,
        n_head=2,
        n_embed=32,
        attention=AttentionConfig(n_group=1, dropout=0.0),
        moe=MoEConfig(enabled=True, num_experts=4, d_hidden_multiplier=2, dropout=0.0),
    )

    loader = CustomDataLoader(B, T, DataConfig(dataset="fineweb", path=cache_dir), split="train")
    model = GPT(config)
    model.train()
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, device="cpu"
    )

    # One full optimizer step over a few micro-batches from the shards.
    optimizer.zero_grad()
    losses = []
    for _ in range(3):
        x, y = loader.next_batch()
        assert x.shape == (B, T) and y.shape == (B, T)
        assert (x < config.vocab_size).all(), "token id outside vocabulary bounds"

        _, loss = model(x, y)
        assert torch.isfinite(loss), "non-finite loss from shard data"
        (loss / 3).backward()
        losses.append(float(loss.detach()))

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert np.isfinite(float(grad_norm))
    optimizer.step()

    # The router actually exercised capacity bookkeeping on real token ids.
    block0_ff = model.transformer.blocks[0].feed_forward
    assert block0_ff.last_total_tokens == B * T
    assert block0_ff.last_drop_rate >= 0.0

    # Second pass: loss must remain finite after the update.
    x, y = loader.next_batch()
    _, loss_after = model(x, y)
    assert torch.isfinite(loss_after)
