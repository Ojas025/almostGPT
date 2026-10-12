"""Integration: KV-cache vs full-recompute generation (plan Phase 5, item 2).

Contract (plan 2.9): greedy (top_k=1) generation with use_cache=True must
match token-for-token a reference loop that recomputes the full forward pass
without any cache.
"""

import pytest
import torch

from src.gpt.config import GenerateConfig
from src.gpt.gpt import GPT


def _greedy_reference(model, prompt: torch.Tensor, steps: int) -> torch.Tensor:
    """Full-forward-only greedy decoding: argmax at every step, no cache."""
    tokens = prompt.clone().unsqueeze(0)
    for _ in range(steps):
        logits, _ = model(tokens)  # recomputes the whole prefix each step
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


def test_kv_cache_generation_matches_full_recompute(tiny_model_config):
    """Cached greedy decoding == uncached greedy decoding, token for token."""
    config = tiny_model_config()
    assert config.kv_cache.enabled, "test exercises the cache-enabled path"
    model = GPT(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    steps = 10
    gen_cfg = GenerateConfig(
        max_new_tokens=steps,
        num_return_sequences=1,
        temperature=1.0,
        top_k=1,      # greedy: deterministic regardless of RNG state
        top_p=0.95,
    )

    with torch.no_grad():
        cached = model.generate(prompt, gen_cfg, device="cpu")
        reference = _greedy_reference(model, prompt[0], steps)

    assert torch.equal(cached, reference), (
        "KV-cache generation diverged from full recomputation:\n"
        f"cached={cached.tolist()}\nreference={reference.tolist()}"
    )


def test_kv_cache_generation_with_multiple_sequences(tiny_model_config):
    """The same equivalence holds across a batch of identical prompts."""
    config = tiny_model_config()
    model = GPT(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 4))
    gen_cfg = GenerateConfig(
        max_new_tokens=6,
        num_return_sequences=3,
        temperature=1.0,
        top_k=1,
        top_p=0.95,
    )

    with torch.no_grad():
        cached = model.generate(prompt, gen_cfg, device="cpu")
        reference = _greedy_reference(model, prompt[0], 6)

    # All return sequences are identical under greedy decoding and match the
    # uncached reference exactly.
    assert torch.equal(cached[0], reference[0])
    assert all(torch.equal(cached[i], cached[0]) for i in range(1, 3))
