"""Unit tests for MaskedSelfAttention (plan Component 4).

Contracts: Vaswani 2017 (causal masking), Ainslie 2023 (GQA), Xiao 2023
(attention sinks / sliding window). Tests exercise the module directly with
tiny configs; causality and fallback-equivalence are asserted per the plan's
"no code muting" rule -- a failing assertion highlights a real bug.
"""

import copy

import pytest
import torch

from src.gpt.attention import MaskedSelfAttention
from src.gpt.config import AttentionConfig, ModelConfig
from src.gpt.kv_cache import KVCache

D_MODEL = 32
VOCAB = 64


def _config(n_head: int = 4, n_group: int = 1, **attn_overrides) -> ModelConfig:
    return ModelConfig(
        context_length=64,
        vocab_size=VOCAB,
        n_layer=1,
        n_head=n_head,
        n_embed=D_MODEL,
        attention=AttentionConfig(n_group=n_group, dropout=0.0, **attn_overrides),
    )


def test_attention_shape_contract():
    """Output is [B, T, C] across head/group configurations."""
    for n_head, n_group in [(4, 1), (4, 2), (4, 4), (8, 2)]:
        config = _config(n_head=n_head, n_group=n_group)
        attn = MaskedSelfAttention(config, n_group=n_group, dropout=0.0)
        x = torch.randn(2, 7, D_MODEL)

        out = attn(x)

        assert out.shape == (2, 7, D_MODEL), f"n_head={n_head}, n_group={n_group}"


def test_gqa_group_broadcasting():
    """With n_head=8, n_group=2, each KV head serves 4 query heads.

    Verified against an independently computed grouped-attention reference at
    T=5 on the causal path, so both the K grouping (it changes the softmax
    scores) and the V grouping are exercised. A wrong mapping (e.g. `repeat`
    instead of group-major `repeat_interleave`) fails this comparison.
    """
    n_head, n_group = 8, 2
    group_size = n_head // n_group
    config = _config(n_head=n_head, n_group=n_group)
    attn = MaskedSelfAttention(config, n_group=n_group, dropout=0.0)
    attn.eval()

    B, T = 1, 5
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)

    # Independent reference computation (RoPE reused: it has its own contract
    # tests; this one isolates the GQA broadcast itself).
    d_head = D_MODEL // n_head
    Q = attn.q_proj(x).view(B, T, n_head, d_head)
    K = attn.k_proj(x).view(B, T, n_group, d_head)
    V = attn.v_proj(x).view(B, T, n_group, d_head)

    Q = attn.RoPE(Q, 0).transpose(1, 2)  # [B, n_head, T, d_head]
    K = attn.RoPE(K, 0).transpose(1, 2).repeat_interleave(group_size, dim=1)
    V = V.transpose(1, 2).repeat_interleave(group_size, dim=1)

    scores = (Q @ K.transpose(-2, -1)) * attn.scale
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    reference = attn.output_proj(
        (weights @ V).transpose(1, 2).reshape(B, T, D_MODEL)
    )

    assert torch.allclose(out, reference, atol=1e-4), (
        "GQA KV heads are not broadcast to their query heads correctly"
    )


@pytest.mark.parametrize("is_flash_attention", [True, False])
def test_causal_masking_strictness(is_flash_attention):
    """Perturbing a future token must have exactly zero impact on earlier outputs.

    Contract: d(output[t1]) / d(x[t2]) == 0 for all t2 > t1.
    """
    config = _config(is_flash_attention=is_flash_attention)
    attn = MaskedSelfAttention(
        config, n_group=1, dropout=0.0, is_flash_attention=is_flash_attention
    )
    attn.eval()

    x = torch.randn(1, 6, D_MODEL)
    perturbed = x.clone()
    perturbed[0, 4] += 100.0  # perturb position t2=4

    base = attn(x)
    changed = attn(perturbed)

    t1 = 2
    assert torch.equal(base[0, t1], changed[0, t1]), (
        f"output at position {t1} changed when perturbing future position 4 "
        f"(is_flash_attention={is_flash_attention}): causality violated"
    )


def test_sliding_window_mask_structure():
    """Query i attends to j iff j <= i AND (j < sink OR j >= i - window)."""
    attn = MaskedSelfAttention(
        _config(sliding_window=4, attention_sink=2),
        n_group=1,
        dropout=0.0,
        sliding_window=4,
        attention_sink=2,
    )
    q_len, kv_len = 10, 10
    mask = attn.build_sliding_window_mask(
        q_len, kv_len, sliding_window=4, attention_sink=2, device=torch.device("cpu")
    )

    for i in range(q_len):
        for j in range(kv_len):
            expected = (j <= i) and (j < 2 or j >= i - 4)
            assert bool(mask[i, j]) == expected, f"mask[{i},{j}] != {expected}"

    # The middle region between sink and local window must actually be dropped.
    # For i=9, sink=2, window=4: visible = {0,1} U {5..9}, so {2,3,4} is dropped
    # and the window boundary j = i - window = 5 stays visible (inclusive).
    i = 9
    assert not mask[i, 3], "token between sink and window should be masked out"
    assert not mask[i, 4], "token between sink and window should be masked out"
    assert mask[i, 5], "window boundary j = i - window must remain visible"


def test_attention_sink_preservation():
    """Sink tokens [0, sink) stay visible to every query, however long T grows."""
    attn = MaskedSelfAttention(
        _config(sliding_window=4, attention_sink=2),
        n_group=1,
        dropout=0.0,
        sliding_window=4,
        attention_sink=2,
    )
    sink = 2
    for T in (8, 32, 128):
        mask = attn.build_sliding_window_mask(
            T, T, sliding_window=4, attention_sink=sink, device=torch.device("cpu")
        )
        # Every query at or past the sink region can still see all sink tokens,
        # which is what keeps softmax weight on them non-zero.
        assert mask[sink:, :sink].all(), f"sink tokens dropped at T={T}"


def test_flash_attention_manual_fallback_equivalence():
    """is_flash_attention=True and False must agree within float tolerance."""
    config = _config()
    flash = MaskedSelfAttention(config, n_group=1, dropout=0.0, is_flash_attention=True)
    flash.eval()
    manual = copy.deepcopy(flash)
    manual.is_flash_attention = False

    x = torch.randn(2, 6, D_MODEL)

    out_flash = flash(x)
    out_manual = manual(x)

    assert torch.allclose(out_flash, out_manual, atol=1e-4), (
        "Flash and manual attention paths disagree -- the manual fallback is "
        "expected to be numerically equivalent (causally masked) softmax attention"
    )


def test_attention_with_kv_cache_step():
    """Stepped T=1 forwards through a KVCache match the full-sequence outputs."""
    config = _config()
    attn = MaskedSelfAttention(config, n_group=1, dropout=0.0)
    attn.eval()

    T = 6
    x = torch.randn(1, T, D_MODEL)

    full = attn(x)

    cache = KVCache(
        n_layers=1,
        d_head=D_MODEL // config.n_head,
        window=T,          # no eviction during this test
        sink=2,
        n_group=1,
    )
    for i in range(T):
        step_out = attn(
            x[:, i : i + 1], use_cache=True, cache=cache, layer_index=0, pos=i
        )
        assert torch.allclose(step_out[0, 0], full[0, i], atol=1e-5), (
            f"cached step {i} does not match full forward at position {i}"
        )


def test_attention_with_kv_cache_step_sliding_window():
    """StreamingLLM contract: stepped decoding under sliding_window + sink must
    match the full forward at every position (plan 2.4 KV-cache support).

    The cache window here exceeds the test length, so eviction is disabled and
    any divergence can only come from the mask applied during stepped decoding.
    """
    window, sink = 4, 2
    config = _config(sliding_window=window, attention_sink=sink)
    attn = MaskedSelfAttention(
        config, n_group=1, dropout=0.0, sliding_window=window, attention_sink=sink
    )
    attn.eval()

    T = 8
    x = torch.randn(1, T, D_MODEL)

    full = attn(x)

    cache = KVCache(
        n_layers=1,
        d_head=D_MODEL // config.n_head,
        window=T,  # >= T: no eviction, isolates the stepped-mask behaviour
        sink=sink,
        n_group=1,
    )
    for i in range(T):
        step_out = attn(
            x[:, i : i + 1], use_cache=True, cache=cache, layer_index=0, pos=i
        )
        assert torch.allclose(step_out[0, 0], full[0, i], atol=1e-5), (
            f"sliding-window cached step {i} diverges from full forward at "
            f"position {i}: the stepped mask must use the real query position "
            f"(pos={i}), not q_len-relative index 0"
        )
