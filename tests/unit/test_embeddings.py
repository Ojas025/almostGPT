"""Unit tests for positional embeddings (plan Component 3).

Contracts: Vaswani et al. 2017 (sinusoidal), Su et al. 2021 RoFormer (RoPE).
RoPE inner-product tests use the documented geometric properties:
rotation preserves norms, and the inner product of a rotated query/key pair
depends only on the relative distance m - n.
"""

import pytest
import torch

from src.gpt.embeddings import (
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

N_EMBED = 8
N_HEAD = 2
D_HEAD = N_EMBED // N_HEAD


def test_learned_positional_embedding_offset():
    """pos=k slices embedding rows [k, k+T) -- required for incremental inference."""
    module = LearnedPositionalEmbedding(context_length=32, n_embed=N_EMBED)
    x = torch.randn(1, 4, N_EMBED)

    at_zero = module(x, pos=0)
    at_k = module(x, pos=7)

    expected_k = module.embedding.weight[7:11].unsqueeze(0)
    assert torch.equal(at_zero, module.embedding.weight[0:4].unsqueeze(0))
    assert torch.equal(at_k, expected_k)


def test_sinusoidal_positional_embedding_buffer():
    """The table is a registered buffer (not a parameter) and follows .to(device)."""
    module = SinusoidalPositionalEmbedding(T=16, n_embed=N_EMBED)

    buffer_names = [name for name, _ in module.named_buffers()]
    param_names = [name for name, _ in module.named_parameters()]

    assert "PE" in buffer_names
    assert "PE" not in param_names

    # `.to()` moves registered buffers; on CPU this is a no-op but proves the wiring.
    module = module.to("cpu")
    assert module.PE.device.type == "cpu"

    # Deterministic values from the Vaswani formula.
    assert torch.all(module.PE[0, 0::2] == 0)    # sin(0)
    assert torch.all(module.PE[0, 1::2] == 1)    # cos(0)


def test_rope_l2_norm_preservation():
    """Rotation is orthogonal: ||RoPE(x, pos)||_2 == ||x||_2."""
    rope = RotaryPositionalEmbedding(N_EMBED, N_HEAD)
    x = torch.randn(2, 6, N_HEAD, D_HEAD)

    rotated = rope(x, pos=3)

    assert torch.allclose(
        rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5
    ), "RoPE must preserve vector L2 norms"


def test_rope_relative_distance_invariance():
    """<RoPE(q, m), RoPE(k, n)> depends only on m - n."""
    rope = RotaryPositionalEmbedding(N_EMBED, N_HEAD)
    q = torch.randn(1, 1, N_HEAD, D_HEAD)
    k = torch.randn(1, 1, N_HEAD, D_HEAD)

    shift = 5
    base_m, base_n = 2, 7

    pair_a = (rope(q, pos=base_m) * rope(k, pos=base_n)).sum()
    pair_b = (rope(q, pos=base_m + shift) * rope(k, pos=base_n + shift)).sum()

    assert torch.allclose(pair_a, pair_b, atol=1e-5), (
        "RoPE relative-distance invariance violated: "
        f"<R_{base_m}q, R_{base_n}k>={pair_a} vs <R_{base_m + shift}q, R_{base_n + shift}k>={pair_b}"
    )


def test_rope_identity_at_same_position():
    """At m = n the rotation cancels: <R_m q, R_m k> == <q, k>."""
    rope = RotaryPositionalEmbedding(N_EMBED, N_HEAD)
    q = torch.randn(1, 1, N_HEAD, D_HEAD)
    k = torch.randn(1, 1, N_HEAD, D_HEAD)

    rotated_inner = (rope(q, pos=4) * rope(k, pos=4)).sum()
    raw_inner = (q * k).sum()

    assert torch.allclose(rotated_inner, raw_inner, atol=1e-5)


def test_rope_cache_auto_expansion():
    """A sequence longer than the initial cache resizes without a shape error."""
    rope = RotaryPositionalEmbedding(N_EMBED, N_HEAD)
    short = torch.randn(1, 1, N_HEAD, D_HEAD)
    long = torch.randn(1, 12, N_HEAD, D_HEAD)

    rope(short, pos=0)
    initial_cache = rope.sin_cache.shape[0]

    out = rope(long, pos=0)

    assert out.shape == long.shape
    assert rope.sin_cache.shape[0] >= 12
    assert rope.sin_cache.shape[0] > initial_cache

    # Growing again from an offset must also work.
    out = rope(long, pos=10)
    assert out.shape == long.shape
    assert rope.sin_cache.shape[0] >= 22
