"""Unit tests for the transformer Block (plan Component 8).

Contract (pre-LN): x_attn = x + Attn(Norm1(x)); out = x_attn + FFN(Norm2(x_attn));
forward returns (output, aux_loss_scalar); cache args pass through to attention.
"""

import pytest
import torch

from src.gpt.kv_cache import KVCache
from src.gpt.transformer import Block


def _zero_linear_weights(module: torch.nn.Module) -> None:
    """Zero every Linear in the subtree: attention output and FFN output become 0."""
    for linear in module.modules():
        if isinstance(linear, torch.nn.Linear):
            torch.nn.init.zeros_(linear.weight)
            if linear.bias is not None:
                torch.nn.init.zeros_(linear.bias)


def test_block_residual_identity(tiny_model_config):
    """With attention and FFN weights zeroed, block output equals input exactly."""
    for moe in (False, True):
        config = tiny_model_config(moe=moe)
        block = Block(config)
        block.eval()

        _zero_linear_weights(block.attention)
        _zero_linear_weights(block.feed_forward)

        x = torch.randn(2, 5, config.n_embed)
        out, aux_loss = block(x)

        assert torch.equal(out, x), f"residual identity broken (moe={moe})"


def test_block_aux_loss_propagation(tiny_model_config):
    """MoE blocks return a scalar aux_loss; dense blocks return a zero scalar."""
    moe_config = tiny_model_config(moe=True)
    moe_block = Block(moe_config)
    x = torch.randn(1, 6, moe_config.n_embed)

    _, aux_moe = moe_block(x)
    assert aux_moe.dim() == 0
    assert torch.isfinite(aux_moe)
    assert aux_moe.item() > 0, "MoE block must surface a positive aux loss"

    dense_config = tiny_model_config(moe=False)
    dense_block = Block(dense_config)

    _, aux_dense = dense_block(x)
    assert aux_dense.dim() == 0
    assert aux_dense.item() == 0.0


def test_block_kv_cache_pass_through(tiny_model_config):
    """Stepped T=1 forwards with a cache match the full-sequence block output."""
    config = tiny_model_config()
    block = Block(config)
    block.eval()

    T = 6
    x = torch.randn(1, T, config.n_embed)

    full, _ = block(x)

    cache = KVCache(
        n_layers=1,
        d_head=config.n_embed // config.n_head,
        window=T,
        sink=config.kv_cache.sink,
        n_group=config.attention.n_group,
    )
    for i in range(T):
        step_out, _ = block(
            x[:, i : i + 1], use_cache=True, cache=cache, layer_index=0, pos=i
        )
        assert step_out.shape == (1, 1, config.n_embed)
        assert torch.allclose(step_out[0, 0], full[0, i], atol=1e-5), (
            f"block cached step {i} does not match full forward"
        )
