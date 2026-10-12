"""Unit tests for SwiGLU (plan Component 1).

Contract (Shazeer 2020, GLU Variants): projection -> 2*out_dim, split in half,
output = x1 * SiLU(x2).
"""

import pytest
import torch

from src.gpt.activations import SwiGLU

IN_DIM = 8
OUT_DIM = 6


def test_swiglu_shape_contract():
    """Input [B, T, D_in] yields output [B, T, D_out]."""
    module = SwiGLU(IN_DIM, OUT_DIM)
    x = torch.randn(2, 5, IN_DIM)

    out = module(x)

    assert out.shape == (2, 5, OUT_DIM)


def test_swiglu_zero_input():
    """SwiGLU(0) = 0 * SiLU(0) = 0 exactly."""
    module = SwiGLU(IN_DIM, OUT_DIM)
    x = torch.zeros(3, 4, IN_DIM)

    out = module(x)

    assert torch.all(out == 0)


def test_swiglu_mathematical_formulation():
    """Forward equals the plan's bias-free reference (xW1) * SiLU(xW2).

    The reference deliberately uses only the projection *weights* per the
    contract in plan section 2.1; an implementation-side bias is a deviation,
    not something the reference should absorb.
    """
    module = SwiGLU(IN_DIM, OUT_DIM)
    x = torch.randn(2, 3, IN_DIM)

    W = module.projection.weight          # [2*OUT_DIM, IN_DIM]
    x1, x2 = (x @ W.T).chunk(2, dim=-1)
    reference = x1 * (x2 * torch.sigmoid(x2))

    assert torch.allclose(module(x), reference, atol=1e-6)


def test_swiglu_gradient_flow():
    """Backprop produces non-zero gradients on the projection weight and bias."""
    module = SwiGLU(IN_DIM, OUT_DIM)
    x = torch.randn(2, 3, IN_DIM)

    module(x).sum().backward()

    assert module.projection.weight.grad is not None
    assert module.projection.bias.grad is not None
    assert module.projection.weight.grad.abs().sum() > 0
    assert module.projection.bias.grad.abs().sum() > 0
