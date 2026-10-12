"""Unit tests for LayerNorm and RMSNorm (plan Component 2).

Contracts: Ba et al. 2016 (LayerNorm) and Zhang & Sennrich 2019 (RMSNorm).
"""

import pytest
import torch

from src.gpt.normalization import LayerNorm, RMSNorm

C = 16


def _random_input(shape=(2, 5, C)):
    return torch.randn(*shape)


def test_layernorm_statistical_moments():
    """With gamma=1, beta=0 output has ~0 mean and ~1 variance over the last dim."""
    module = LayerNorm(C)
    with torch.no_grad():
        module.gamma.fill_(1.0)
        module.beta.zero_()

    out = module(_random_input())

    assert out.mean(dim=-1).abs().max() < 1e-5
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(out.shape[:-1]), atol=1e-4)


def test_layernorm_shift_invariance():
    """LayerNorm(x + c) == LayerNorm(x) for any constant offset c."""
    module = LayerNorm(C)
    x = _random_input()

    shifted = module(x + 3.7)
    reference = module(x)

    assert torch.allclose(shifted, reference, atol=1e-5)


def test_rmsnorm_rms_moment():
    """With scale=1, RMSNorm output satisfies RMS(y) ~= 1 along the last dim."""
    module = RMSNorm(C)
    with torch.no_grad():
        module.scale.fill_(1.0)

    out = module(_random_input())

    rms = torch.sqrt((out**2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_scale_parameter():
    """Scaling gamma scales the output linearly (RMSNorm is homogeneous in gamma)."""
    module = RMSNorm(C)
    x = _random_input()

    with torch.no_grad():
        module.scale.fill_(1.0)
        base = module(x)
        module.scale.fill_(2.0)
        scaled = module(x)

    assert torch.allclose(scaled, 2.0 * base, atol=1e-6)


def test_norm_numerical_stability():
    """Zero / tiny inputs produce no NaN or Inf thanks to epsilon in the denominator."""
    for module in (LayerNorm(C), RMSNorm(C)):
        out_zero = module(torch.zeros(2, 3, C))
        out_tiny = module(torch.full((2, 3, C), 1e-12))

        assert torch.isfinite(out_zero).all(), f"{type(module).__name__} NaN/Inf on zeros"
        assert torch.isfinite(out_tiny).all(), f"{type(module).__name__} NaN/Inf on tiny input"
