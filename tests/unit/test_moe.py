"""Unit tests for MixtureOfExperts / Router / Expert (plan Component 7).

Contracts: Fedus et al. 2022 (Switch Transformer: top-1 routing, capacity,
aux load-balancing loss, router z-loss). Tests drive the router with inputs
crafted to force known routing outcomes (balanced vs collapsed).
"""

import pytest
import torch

from src.gpt.config import AttentionConfig, ModelConfig, MoEConfig
from src.gpt.mixture_of_experts import MixtureOfExperts, Router

D_MODEL = 8
N_EXPERTS = 4
TOKENS_PER_EXPERT = 3
NUM_TOKENS = N_EXPERTS * TOKENS_PER_EXPERT  # 12 -> capacity = 3 per expert


def _router(**overrides) -> Router:
    defaults = dict(
        d_model=D_MODEL,
        num_experts=N_EXPERTS,
        z_loss_coef=1e-3,
        aux_loss_coef=1e-2,
        capacity_factor=1.0,
        min_capacity=1,
    )
    defaults.update(overrides)
    return Router(**defaults)


def _balanced_inputs() -> torch.Tensor:
    """Tokens that the identity gate routes exactly TOKENS_PER_EXPERT per expert."""
    router = _router()
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight.copy_(torch.eye(N_EXPERTS, D_MODEL))

    # One-hot tokens scaled so the argmax is unambiguous.
    tokens = torch.eye(D_MODEL)[
        torch.arange(NUM_TOKENS) // TOKENS_PER_EXPERT
    ] * 10.0
    return tokens.unsqueeze(0), router


def test_router_output_shapes():
    """Routing dict has expert_index [B*T], expert_prob, counts, capacity; aux scalar."""
    router = _router()
    x = torch.randn(2, 5, D_MODEL)

    routing, aux_loss = router(x)

    assert routing["expert_index"].shape == (10,)
    assert routing["expert_prob"].shape == (10,)
    assert routing["counts"].shape == (N_EXPERTS,)
    assert routing["counts"].sum().item() == 10
    assert isinstance(routing["capacity"], int)
    assert aux_loss.dim() == 0
    assert torch.isfinite(aux_loss)


def test_router_load_balancing_loss():
    """Perfectly balanced routing yields far lower aux loss than collapsed routing.

    aux = coef * E * sum(f_i * p_i) (+ z-loss). Balanced one-hot routing gives
    E*sum(f*p) ~= 1; collapsing all tokens into one expert with confident gate
    probabilities gives ~= E. A zeroed gate would NOT work here: uniform p makes
    both cases equal, which is a known property of the Switch loss. z_loss_coef
    is zeroed to isolate the load-balancing term from the (dominant) z-loss.
    """
    balanced_x, router = _balanced_inputs()
    with torch.no_grad():
        router.z_loss_coef = 0.0

    routing_b, aux_balanced = router(balanced_x)
    assert routing_b["counts"].tolist() == [TOKENS_PER_EXPERT] * N_EXPERTS

    # Collapse: every token is expert-0's one-hot, so the identity gate routes
    # all of them confidently to expert 0.
    collapsed_x = (
        torch.eye(D_MODEL)[torch.zeros(NUM_TOKENS, dtype=torch.long)] * 10.0
    ).unsqueeze(0)
    routing_c, aux_collapsed = router(collapsed_x)
    assert routing_c["counts"].tolist() == [NUM_TOKENS, 0, 0, 0]

    assert aux_collapsed > aux_balanced * 2, (
        "collapsed routing must be penalised substantially more than balanced routing"
    )


def test_router_z_loss():
    """The z-loss grows quadratically with gate logit magnitude.

    z = beta * mean(logsumexp(logits)^2); for large logits logsumexp scales
    linearly with the input scale, so doubling the inputs must roughly
    quadruple the z-loss.
    """
    router = _router(z_loss_coef=1.0, aux_loss_coef=0.0)
    x = torch.randn(1, 8, D_MODEL)

    _, aux_small = router(x)
    _, aux_large = router(x * 50.0)
    assert aux_large > aux_small, "z-loss must grow with gate logit magnitude"

    # Deep in the linear-logits regime logsumexp scales with the input scale,
    # so doubling the input must quadruple the squared term.
    _, aux_500 = router(x * 500.0)
    _, aux_1000 = router(x * 1000.0)

    quadratic_ratio = (aux_1000 / aux_500).item()
    assert quadratic_ratio == pytest.approx(4.0, rel=0.1), (
        f"doubling large gate logits must ~4x the z-loss, got ratio "
        f"{quadratic_ratio:.3f}"
    )


def test_expert_capacity_and_dropping():
    """Tokens beyond an expert's capacity are counted, not crashed on."""
    config = ModelConfig(
        context_length=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embed=D_MODEL,
        attention=AttentionConfig(n_group=1),
        moe=MoEConfig(
            enabled=True,
            num_experts=2,
            d_hidden_multiplier=2,
            capacity_factor=1.0,
            min_capacity=4,
            dropout=0.0,
        ),
    )
    moe = MixtureOfExperts(config)

    # Force every token onto expert 0 by zeroing the gate (argmax tie -> index 0).
    with torch.no_grad():
        moe.router.gate.weight.zero_()

    num_tokens = 16  # capacity = max(4, ceil(1.0 * 16 / 2)) = 8
    x = torch.randn(1, num_tokens, D_MODEL)

    out, aux_loss = moe(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all(), "dropped tokens must not produce NaN/Inf"
    assert torch.isfinite(aux_loss)

    expected_capacity = 8
    assert moe.last_capacity == expected_capacity
    assert moe.last_dropped_tokens == num_tokens - expected_capacity
    assert moe.last_drop_rate == pytest.approx(moe.last_dropped_tokens / num_tokens)
    assert moe.drop_rate > 0


def test_moe_forward_backward():
    """Gradients reach both the router gate and the active experts."""
    config = ModelConfig(
        context_length=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embed=D_MODEL,
        attention=AttentionConfig(n_group=1),
        moe=MoEConfig(enabled=True, num_experts=4, d_hidden_multiplier=2, dropout=0.0),
    )
    moe = MixtureOfExperts(config)
    x = torch.randn(1, 8, D_MODEL, requires_grad=True)

    out, aux_loss = moe(x)
    (out.sum() + aux_loss).backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.router.gate.weight.grad is not None
    assert moe.router.gate.weight.grad.abs().sum() > 0, "router gate must receive gradient"

    # Every expert that received tokens must have non-None, non-zero grads.
    counts = moe.last_expert_load
    for i, expert in enumerate(moe.experts):
        for name, param in expert.named_parameters():
            assert param.grad is not None, f"expert {i} param {name} has no grad"
            if counts[i] > 0:
                assert param.grad.abs().sum() > 0, (
                    f"active expert {i} param {name} received a zero gradient"
                )
