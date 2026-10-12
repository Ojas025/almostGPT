"""Unit tests for the core GPT model (plan Component 9).

Contracts: logits shape [B,T,V], context-limit ValueError, weight tying,
decay/no-decay optimizer groups, top-k / top-p filtering, generate length.
"""

import pytest
import torch

from src.gpt.config import GenerateConfig
from src.gpt.gpt import GPT

B, T, VOCAB = 2, 8, 64


def _tokens(B=B, T=T, vocab=VOCAB):
    return torch.randint(0, vocab, (B, T))


def test_gpt_forward_without_targets(tiny_model_config):
    """Returns (logits [B,T,vocab_size], aux_loss scalar)."""
    config = tiny_model_config()
    model = GPT(config)

    logits, aux_loss = model(_tokens())

    assert logits.shape == (B, T, config.vocab_size)
    assert aux_loss.dim() == 0
    assert torch.isfinite(logits).all()


def test_gpt_forward_with_targets(tiny_model_config):
    """Returns (logits, CE loss + aux_loss) when targets are provided."""
    config = tiny_model_config(moe=True)
    model = GPT(config)
    model.eval()

    x = _tokens()
    y = _tokens()

    logits, loss = model(x, y)

    with torch.no_grad():
        reference_ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        _, aux_only = model(x)

    assert loss.dim() == 0
    assert torch.allclose(loss, reference_ce + aux_only, atol=1e-6)


def test_gpt_context_length_exceeded(tiny_model_config):
    """T > context_length raises ValueError, not a downstream crash."""
    config = tiny_model_config()
    model = GPT(config)

    too_long = _tokens(T=config.context_length + 1)

    with pytest.raises(ValueError, match="context length"):
        model(too_long)


def test_weight_tying(tiny_model_config):
    """tie_embeddings=True shares the exact parameter tensor; False does not."""
    tied = GPT(tiny_model_config(tie_embeddings=True))
    assert tied.transformer.token_embedding.weight is tied.lm_head.weight

    untied = GPT(tiny_model_config(tie_embeddings=False))
    assert untied.transformer.token_embedding.weight is not untied.lm_head.weight


def test_optimizer_parameter_groups(tiny_model_config):
    """2D+ weights get weight_decay; 1D vectors (biases/norms) get 0.0."""
    config = tiny_model_config()
    model = GPT(config)
    wd = 0.42

    optimizer = model.configure_optimizers(
        weight_decay=wd, learning_rate=1e-3, device="cpu"
    )

    assert len(optimizer.param_groups) == 2
    decay_group, no_decay_group = optimizer.param_groups

    assert decay_group["weight_decay"] == wd
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])

    # Every trainable parameter lands in exactly one group.
    total = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == trainable


def test_top_k_filtering(tiny_model_config):
    """Only the k highest logits survive; the rest become -inf."""
    model = GPT(tiny_model_config())
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])

    filtered = model.top_k(logits, k=2)

    kept = filtered[0][torch.isfinite(filtered[0])].sort(descending=True).values
    assert torch.equal(kept, torch.tensor([5.0, 4.0]))
    assert (filtered[0][~torch.isfinite(filtered[0])] == float("-inf")).all()
    # Surviving values are unmodified.
    assert filtered[0, 1] == 5.0 and filtered[0, 4] == 4.0


def test_top_p_nucleus_filtering(tiny_model_config):
    """Nucleus keeps the smallest prefix of sorted tokens with cumprob >= p.

    The crossing token -- the one whose inclusion pushes the cumulative mass
    over p -- must be retained, so the kept probability mass is never below
    the requested threshold (HF / nanoGPT convention).
    """
    model = GPT(tiny_model_config())
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    p = 0.9

    filtered = model.top_p(logits, p=p)

    probs = torch.softmax(logits, dim=-1)
    # cumsum = [0.636, 0.871, 0.957, 0.988, 1.0] -> smallest prefix >= 0.9
    # is {0, 1, 2}; index 2 is the crossing token.
    kept = torch.isfinite(filtered[0])
    assert kept[0] and kept[1] and kept[2], (
        f"nucleus must include the crossing token; kept={kept.tolist()}, "
        f"cumsum={torch.cumsum(probs, dim=-1).tolist()}"
    )
    assert not kept[3] and not kept[4], "tail tokens must be masked to -inf"

    kept_mass = probs[0][kept].sum()
    assert kept_mass >= p, (
        f"kept probability mass {kept_mass:.4f} is below the requested p={p} -- "
        "nucleus sampling must retain at least the threshold mass"
    )
    # At least one token is always kept.
    assert torch.isfinite(model.top_p(logits, p=0.01)).any()


def test_generate_sequence_length(tiny_model_config):
    """generate() returns [num_return_sequences, prompt_len + max_new_tokens]."""
    config = tiny_model_config()
    model = GPT(config)

    prompt_len, max_new, n_return = 5, 7, 3
    prompt = _tokens(B=1, T=prompt_len)
    gen_cfg = GenerateConfig(
        max_new_tokens=max_new,
        num_return_sequences=n_return,
        temperature=1.0,
        top_k=8,
        top_p=0.95,
    )

    out = model.generate(prompt, gen_cfg, device="cpu")

    assert out.shape == (n_return, prompt_len + max_new)
    # The prompt prefix is preserved verbatim in every return sequence.
    assert torch.equal(out[:, :prompt_len], prompt.repeat(n_return, 1))
    assert (out >= 0).all() and (out < config.vocab_size).all()
