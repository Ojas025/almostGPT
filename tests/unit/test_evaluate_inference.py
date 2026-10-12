"""Unit tests for evaluation & inference helpers (plan Component 14).

Contracts: evaluate_model runs no-grad eval and resets the loader;
get_predicted_row scores HellaSwag continuations with a masked, clamped,
length-normalised loss; completion returns only newly generated tokens.
"""

import pytest
import torch

from src.gpt.config import GenerateConfig
from src.gpt.evaluate import evaluate_model
from src.gpt.gpt import GPT
from src.gpt.inference import completion, get_predicted_row
from src.gpt.tokenizer import Tokenizer

VOCAB = 12


def test_evaluate_model_no_grad_and_eval_mode(tiny_model_config, mock_loader):
    """evaluate_model resets the loader, switches to eval, runs with grad disabled."""
    model = GPT(tiny_model_config())
    model.train()  # deliberately in train mode beforehand
    loader = mock_loader(num_batches=3)

    # Direct observation of the grad context *inside* forward: `param.grad is
    # None` alone would pass even without @torch.no_grad (grads only appear
    # after backward), so the hook is what actually verifies the contract.
    grad_enabled_during_forward: list[bool] = []
    handle = model.register_forward_pre_hook(
        lambda module, args: grad_enabled_during_forward.append(torch.is_grad_enabled())
    )
    try:
        loss = evaluate_model(model, loader, step=0, device="cpu", max_batches=3)
    finally:
        handle.remove()

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert model.training is False, "evaluate_model must put the model in eval mode"
    assert loader.reset_calls >= 1, "evaluate_model must reset the val loader"
    assert grad_enabled_during_forward, "forward hook never fired"
    assert not any(grad_enabled_during_forward), (
        "evaluate_model forward ran with grad mode enabled"
    )
    for name, param in model.named_parameters():
        assert param.grad is None, f"gradient stored on {name} during no-grad eval"


def _hellaswag_setup():
    """3 choices, context = positions 0-2, continuation = positions 3-4.

    Choice 2's logits confidently predict its own continuation tokens; choices
    0 and 1 confidently predict WRONG tokens, so choice 2 must win.
    """
    T = 5
    context = [7, 8, 9]
    continuations = [(1, 2), (3, 4), (5, 6)]

    tokens = torch.tensor([context + list(c) for c in continuations])
    mask = torch.tensor([[0, 0, 0, 1, 1]] * 3)

    logits = torch.full((3, T, VOCAB), -10.0)
    for c, (t3, t4) in enumerate(continuations):
        # logits at position p score the token at p+1
        if c == 2:
            logits[c, 2, t3] = 10.0  # correct continuation token
            logits[c, 3, t4] = 10.0
        else:
            wrong = (t3 + 1) % VOCAB
            logits[c, 2, wrong] = 10.0  # confidently wrong
            logits[c, 3, wrong] = 10.0
    return tokens, mask, logits


def test_hellaswag_get_predicted_row_scoring():
    """The choice with the lowest masked continuation loss is selected."""
    tokens, mask, logits = _hellaswag_setup()

    prediction = get_predicted_row(tokens, mask, logits)

    assert prediction == 2


def test_hellaswag_masking_context_exclusion():
    """Perturbing context-region logits cannot change the ranking."""
    tokens, mask, logits = _hellaswag_setup()

    # Positions 0-1 only score masked-out (context) targets.
    logits[:, 0:2] = torch.randn(3, 2, VOCAB) * 100.0

    assert get_predicted_row(tokens, mask, logits) == 2


def test_hellaswag_all_zero_mask_safety():
    """An all-zero continuation mask yields a valid, finite index.

    NOTE: the clamp's NaN-prevention itself is not observable through this
    return value -- argmin maps an all-NaN row to index 0 exactly as an
    all-zero row does -- so what is asserted here is the observable contract:
    a deterministic, in-range integer result with no exception on the
    degenerate mask.
    """
    tokens, mask, logits = _hellaswag_setup()
    zero_mask = torch.zeros_like(mask)

    prediction = get_predicted_row(tokens, zero_mask, logits)

    assert isinstance(prediction, int)
    assert 0 <= prediction < tokens.size(0)


def test_completion_prompt_stripping(tiny_model_config):
    """completion() returns only newly generated tokens, prompt excluded."""
    # The fixed HellaSwag-style prompt contains real GPT-2 ids, so the model
    # needs a full-size vocabulary to embed them.
    model = GPT(tiny_model_config(vocab_size=50304))
    gen_cfg = GenerateConfig(
        max_new_tokens=4, num_return_sequences=2, temperature=1.0, top_k=8, top_p=0.95
    )
    prompt = "Hello, I'm a language model"
    prompt_len = Tokenizer().encode(prompt).size(0)

    torch.manual_seed(7)
    decoded = completion(model, gen_cfg, device="cpu")

    torch.manual_seed(7)
    generated = model.generate(Tokenizer().encode(prompt), gen_cfg, device="cpu")

    expected = [
        Tokenizer().decode(generated[i, prompt_len:].tolist())
        for i in range(gen_cfg.num_return_sequences)
    ]

    assert decoded == expected, "completion must strip the prompt prefix"
    # Sanity: the returned strings really are continuations of differing content.
    assert len(decoded) == gen_cfg.num_return_sequences
    assert all(isinstance(s, str) for s in decoded)
