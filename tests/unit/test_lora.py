"""Unit tests for LoRALinear (plan Component 10).

Contract (Hu et al. 2021): frozen base W0, Delta-W = (alpha/r) * B @ A,
B zero-initialised so the adapter is the identity map at construction.
"""

import torch

from src.gpt.LoRA import LoRALinear

IN_FEATURES = 8
OUT_FEATURES = 6
RANK = 4


def _base() -> torch.nn.Linear:
    return torch.nn.Linear(IN_FEATURES, OUT_FEATURES)


def test_lora_initial_forward_identity():
    """At init (B = 0), LoRALinear(x) == base(x) exactly."""
    base = _base()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)

    x = torch.randn(3, IN_FEATURES)

    assert torch.all(lora.B == 0)
    assert torch.equal(lora(x), base(x)), "adapter must be identity before training"


def test_lora_base_weights_frozen():
    """Base weight and bias have requires_grad = False."""
    lora = LoRALinear(_base(), rank=RANK, alpha=8.0)

    assert lora.base.weight.requires_grad is False
    assert lora.base.bias.requires_grad is False


def test_lora_adapter_weights_trainable():
    """A and B are trainable parameters."""
    lora = LoRALinear(_base(), rank=RANK, alpha=8.0)

    assert lora.A.requires_grad is True
    assert lora.B.requires_grad is True
    assert lora.A.shape == (RANK, IN_FEATURES)
    assert lora.B.shape == (OUT_FEATURES, RANK)


def test_lora_rank_scaling():
    """Output delta scales proportionally to alpha / r (same A, B, base)."""
    base = _base()
    lora_a = LoRALinear(base, rank=RANK, alpha=8.0)   # scale 2.0
    lora_b = LoRALinear(base, rank=RANK, alpha=2.0)   # scale 0.5

    # Identical adapters: same A, B -> delta ratio must equal the scale ratio.
    with torch.no_grad():
        lora_b.A.copy_(lora_a.A)
        lora_b.B.copy_(lora_a.B)

    assert lora_a.scale == 8.0 / RANK
    assert lora_b.scale == 2.0 / RANK

    x = torch.randn(3, IN_FEATURES)
    delta_a = lora_a(x) - base(x)
    delta_b = lora_b(x) - base(x)

    assert torch.allclose(delta_a, 4.0 * delta_b, atol=1e-6)
    # And the delta is actually non-zero (B must have been moved off zero).
    with torch.no_grad():
        lora_a.B.fill_(0.1)
    assert lora_a(x).abs().sum() > 0
