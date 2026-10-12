"""Shared fixtures for the almostGPT test suite (plan Phase 1).

Every fixture here exists so individual test files stay declarative: configs,
seeds, synthetic corpora and shard directories are built once with deterministic
values, keeping the suite reproducible on CPU with zero network access.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gpt.config import (
    AttentionConfig,
    DataConfig,
    KVCacheConfig,
    ModelConfig,
    MoEConfig,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed every RNG each test so stochastic tests are reproducible (zero flakiness)."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    yield


# ---------------------------------------------------------------------------
# Tiny model configs
# ---------------------------------------------------------------------------

# Sizes kept small so a full forward/backward runs in milliseconds on CPU.
TINY_VOCAB = 64
TINY_N_EMBED = 32
TINY_N_HEAD = 2
TINY_CONTEXT = 32


def _tiny_model_config(
    moe: bool = False,
    norm_type: str = "rms",
    sliding_window: int | None = None,
    attention_sink: int | None = None,
    tie_embeddings: bool = True,
    **overrides,
) -> ModelConfig:
    """Factory for a tiny ModelConfig; `overrides` are passed straight to ModelConfig."""
    defaults = dict(
        context_length=TINY_CONTEXT,
        vocab_size=TINY_VOCAB,
        n_layer=2,
        n_head=TINY_N_HEAD,
        n_embed=TINY_N_EMBED,
        norm_type=norm_type,
        tie_embeddings=tie_embeddings,
        attention=AttentionConfig(
            n_group=1,
            is_flash_attention=True,
            sliding_window=sliding_window,
            attention_sink=attention_sink,
            dropout=0.0,  # dropout off: stochastic tests stay deterministic
        ),
        moe=MoEConfig(
            enabled=moe,
            num_experts=4,
            d_hidden_multiplier=2,
            dropout=0.0,
        ),
        kv_cache=KVCacheConfig(enabled=True, window=None, sink=2),
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


@pytest.fixture
def tiny_model_config():
    """`tiny_model_config(moe=False, norm_type='rms', sliding_window=None, ...)`."""
    return _tiny_model_config


@pytest.fixture
def tiny_train_config():
    """Minimal TrainConfig that satisfies the pydantic geometry validators."""

    def make(**overrides) -> TrainConfig:
        defaults = dict(
            batch_size=2,
            context=8,
            total_batch_size=16,  # == batch_size * context -> 1 grad-accum step
            learning_rate=1e-3,
            min_lr_ratio=0.1,
            warmup_steps=10,
            max_steps=100,
            weight_decay=0.1,
            grad_clip=1.0,
            device="cpu",
            amp=False,
            compile=False,
            eval_interval=5,
            checkpoint_interval=5,
            log_interval=1,
            seed=SEED,
        )
        defaults.update(overrides)
        return TrainConfig(**defaults)

    return make


# ---------------------------------------------------------------------------
# Synthetic corpora / shard directories
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_tokens():
    """A deterministic uint16 token stream usable as a fake corpus."""

    def make(n: int, offset: int = 0) -> np.ndarray:
        return ((np.arange(n) + offset) % (TINY_VOCAB - 1) + 1).astype(np.uint16)

    return make


@pytest.fixture
def shard_dir(tmp_path, synthetic_tokens):
    """Write tiny .npy shards into a temp dir; returns (dir_path, all_tokens_by_split)."""

    def make(sizes: list[int], split: str = "train") -> tuple:
        directory = tmp_path / "shards"
        directory.mkdir(exist_ok=True)
        written = []
        offset = 0
        for i, size in enumerate(sizes):
            tokens = synthetic_tokens(size, offset=offset)
            # Naming mirrors fineweb.py so CustomDataLoader's `*{split}*` glob matches.
            np.save(directory / f"edufineweb_{split}_{i:06d}.npy", tokens)
            written.append(tokens)
            offset += size
        return directory, written

    return make


# ---------------------------------------------------------------------------
# Mock data loaders (avoid the tokenizer/shakespeare dependency in unit tests)
# ---------------------------------------------------------------------------


class MockLoader:
    """Minimal stand-in for CustomDataLoader: `.reset()`, `.next_batch()`, `.index`."""

    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]]):
        self._batches = batches
        self.reset_calls = 0
        self.index = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.index = 0

    def next_batch(self):
        batch = self._batches[self.index % len(self._batches)]
        self.index += 1
        return batch


@pytest.fixture
def mock_loader():
    """Build a MockLoader over a synthetic [B, T] token stream (targets = shift-by-1)."""

    def make(num_batches: int = 4, B: int = 2, T: int = 8, vocab: int = TINY_VOCAB):
        data = torch.randint(0, vocab, (num_batches * B * T + 1,))
        batches = []
        for i in range(num_batches):
            buf = data[i * B * T : (i + 1) * B * T + 1]
            batches.append((buf[:-1].view(B, T), buf[1:].view(B, T)))
        return MockLoader(batches)

    return make


# ---------------------------------------------------------------------------
# FineWeb pipeline driver (no network: load_dataset / mp.Pool are faked)
# ---------------------------------------------------------------------------


class _FakePool:
    """Replaces mp.Pool: imap just maps, no processes needed in tests."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap(self, fn, iterable, chunksize=1):
        return map(fn, iterable)


class _SilentTqdm:
    """No-op progress bar so test output stays clean."""

    def __init__(self, *args, **kwargs):
        pass

    def update(self, n):
        pass


@pytest.fixture
def fineweb_pipeline(monkeypatch, tmp_path):
    """Drive the real fineweb.main() sharding loop over fake in-memory documents.

    Only the network/process edges are faked (load_dataset, mp.Pool, tqdm);
    tokenize / shard spilling / split assignment / write_file are the real code.
    Returns the output cache directory.
    """
    import src.dataset.fineweb as fineweb

    def run(texts: list[str], shard_size: int) -> Path:
        cache_dir = tmp_path / "fineweb_cache"
        cache_dir.mkdir(exist_ok=True)

        monkeypatch.setattr(fineweb, "load_dataset", lambda *a, **k: [{"text": t} for t in texts])
        monkeypatch.setattr(fineweb.mp, "Pool", _FakePool)
        monkeypatch.setattr(fineweb, "tqdm", _SilentTqdm)
        monkeypatch.setattr(fineweb, "shard_size", shard_size)
        monkeypatch.setattr(fineweb, "DATA_CACHE_DIR", str(cache_dir))

        fineweb.main()
        return cache_dir

    return run
