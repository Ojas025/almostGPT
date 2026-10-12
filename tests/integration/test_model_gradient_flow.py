"""Integration: end-to-end gradient flow (plan Phase 5, item 1).

Contracts: backprop reaches every parameter class (embeddings, norms, Q/K/V,
output proj, FFN or MoE router+experts, LM head); LoRA freezes the base and
trains only A/B.
"""

import pytest
import torch

from src.gpt.LoRA import LoRALinear
from src.gpt.gpt import GPT


def test_gradient_flow_all_parameters(tiny_model_config):
    """After loss.backward(), every trainable parameter has a finite non-None grad."""
    config = tiny_model_config(moe=True)
    model = GPT(config)
    model.train()

    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    _, loss = model(x, y)
    loss.backward()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Experts that received no tokens in their own block are legitimately
        # grad-free (the expert module is never invoked).
        if ".feed_forward.experts." in name:
            block_idx = int(name.split("blocks.")[1].split(".")[0])
            expert_idx = int(name.split("experts.")[1].split(".")[0])
            load = model.transformer.blocks[block_idx].feed_forward.last_expert_load
            if load[expert_idx] == 0:
                continue

        assert param.grad is not None, f"no grad reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad on {name}"
        assert param.grad.abs().sum() > 0, f"gradient is exactly zero on {name}"


def test_gradient_flow_dense_ffn(tiny_model_config):
    """The dense FeedForward path also receives gradients everywhere."""
    config = tiny_model_config(moe=False)
    model = GPT(config)
    model.train()

    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    _, loss = model(x, y)
    loss.backward()

    embedding_grad = model.transformer.token_embedding.weight.grad
    assert embedding_grad is not None and embedding_grad.abs().sum() > 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no grad reached {name}"
            assert param.grad.abs().sum() > 0, f"zero grad on {name}"


def test_lora_gradient_flow():
    """Base weights stay grad-free while adapters A and B train.

    B is nudged off zero first: at B=0 the gradient of A is mathematically
    zero (it flows through B) -- that is LoRA's cold-start property, covered
    by the identity-init unit test, not a flow failure.
    """
    base = torch.nn.Linear(8, 6)
    lora = LoRALinear(base, rank=4, alpha=8.0)

    with torch.no_grad():
        lora.B.fill_(0.1)  # simulate a slightly-trained adapter

    x = torch.randn(5, 8)
    lora(x).sum().backward()

    assert base.weight.grad is None, "frozen base must not accumulate gradients"
    assert base.bias.grad is None, "frozen bias must not accumulate gradients"
    assert lora.A.grad is not None and lora.A.grad.abs().sum() > 0
    assert lora.B.grad is not None and lora.B.grad.abs().sum() > 0
