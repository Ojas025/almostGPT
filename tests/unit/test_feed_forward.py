"""Unit tests for FeedForward (plan Component 6).

Contract: supports "gelu", "relu", "swiglu"; D -> 4D -> D; position-wise.
"""

import pytest
import torch

from src.gpt.config import AttentionConfig, ModelConfig
from src.gpt.feed_forward import FeedForward

N_EMBED = 16


def _config() -> ModelConfig:
    return ModelConfig(
        context_length=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embed=N_EMBED,
        attention=AttentionConfig(n_group=1),
    )


@pytest.mark.parametrize("activation", ["gelu", "relu", "swiglu"])
def test_feedforward_activations(activation):
    """Each supported activation builds and runs a forward pass [B,T,C] -> [B,T,C]."""
    ff = FeedForward(_config(), activation=activation)
    x = torch.randn(2, 5, N_EMBED)

    out = ff(x)

    assert out.shape == (2, 5, N_EMBED)
    assert torch.isfinite(out).all()


def test_feedforward_invalid_activation():
    """Unknown activation strings raise ValueError."""
    with pytest.raises(ValueError, match="unknown activation"):
        FeedForward(_config(), activation="not_a_real_activation")


def test_feedforward_token_independence():
    """Position-wise: changing token t' only changes output at t'."""
    ff = FeedForward(_config(), activation="gelu")
    ff.eval()

    x = torch.randn(1, 6, N_EMBED)
    perturbed = x.clone()
    perturbed[0, 3] += 5.0

    base = ff(x)
    changed = ff(perturbed)

    positions = [i for i in range(6) if i != 3]
    assert torch.equal(base[0, positions], changed[0, positions])
    assert not torch.allclose(base[0, 3], changed[0, 3])
