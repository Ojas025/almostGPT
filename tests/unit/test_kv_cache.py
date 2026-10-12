"""Unit tests for KVCache (plan Component 5).

Contract: append along sequence dim; rolling window retains attention sinks
([:sink]) plus the most recent `window` tokens; total never exceeds sink+window.
"""

import pytest
import torch

from src.gpt.kv_cache import KVCache

N_LAYERS = 2
D_HEAD = 4


def _kv(value: float, T: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V tensors filled with an identifiable constant."""
    k = torch.full((1, 1, T, D_HEAD), float(value))
    v = torch.full((1, 1, T, D_HEAD), float(value) + 100.0)
    return k, v


def test_kv_cache_initial_empty_state():
    """is_empty() is True before prefill; every layer cache is (None, None)."""
    cache = KVCache(n_layers=N_LAYERS, d_head=D_HEAD, window=8, sink=2, n_group=1)

    assert cache.is_empty()
    assert all(entry == (None, None) for entry in cache.cache)


def test_kv_cache_prefill_and_step_update():
    """Prefill with T0 tokens, then a T=1 step extends length by exactly 1."""
    cache = KVCache(n_layers=1, d_head=D_HEAD, window=16, sink=2, n_group=1)

    K, V = cache.update(0, *_kv(0.0, T=5))
    assert K.shape[2] == 5
    assert not cache.is_empty()

    K, V = cache.update(0, *_kv(1.0, T=1))
    assert K.shape[2] == 6
    assert V.shape[2] == 6


def test_kv_cache_rolling_window_sink_eviction():
    """sink=2, window=4: after 10 tokens the cache holds indices [0,1,6,7,8,9]."""
    cache = KVCache(n_layers=1, d_head=D_HEAD, window=4, sink=2, n_group=1)

    for token_id in range(10):
        K, _ = cache.update(0, *_kv(float(token_id)))

    # Identifiable values: K of token i carries value i.
    stored = K[0, 0, :, 0].tolist()

    assert len(stored) == 6, f"cache length {len(stored)} != sink + window = 6"
    assert stored == [0.0, 1.0, 6.0, 7.0, 8.0, 9.0], (
        "expected permanent sinks [0,1] followed by the last `window` tokens [6..9]"
    )


def test_kv_cache_reset():
    """reset() clears every layer and is_empty() returns True again."""
    cache = KVCache(n_layers=N_LAYERS, d_head=D_HEAD, window=8, sink=2, n_group=1)

    cache.update(0, *_kv(0.0, T=3))
    cache.update(1, *_kv(1.0, T=3))
    assert not cache.is_empty()

    cache.reset()

    assert cache.is_empty()
    assert all(entry == (None, None) for entry in cache.cache)
