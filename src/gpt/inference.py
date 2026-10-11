import logging

import torch
import torch.nn as nn
from .config import GenerateConfig
from .console import console
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def get_predicted_row(
    tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor
) -> int:
    """Pick the HellaSwag ending with the lowest masked continuation loss.

    tokens/mask: [num_choices, T] -- mask marks continuation tokens with 1.
    logits: [num_choices, T, vocab_size]. Returns the index of the best choice.
    """
    num_choices = tokens.size(0)

    # Time is dim 1, so shift with `[..., 1:]`. The previous `tokens[..., 1:, :]` applied
    # to a 2-D [num_choices, T] tensor did not raise -- the ellipsis absorbed a
    # dimension and it silently returned [num_choices - 1, T], dropping a whole choice.
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_tokens = tokens[:, 1:].contiguous()
    shifted_mask = mask[:, 1:].contiguous()

    flattened_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flattened_tokens = shifted_tokens.view(-1)

    # `reduction` belongs to the constructor; passing it to forward is a TypeError.
    losses = nn.CrossEntropyLoss(reduction="none")(
        flattened_logits, flattened_tokens
    )
    losses = losses.view(num_choices, -1)

    masked_loss = losses * shifted_mask

    sum_loss = masked_loss.sum(dim=1)
    # A row whose mask is all zeros has no continuation tokens; clamp so it cannot
    # divide by zero and produce NaN that would silently win/lose the argmin.
    avg_loss = sum_loss / shifted_mask.sum(dim=1).clamp(min=1)

    prediction = avg_loss.argmin().item()

    return prediction


def completion(
    model, generate_config: GenerateConfig, device: str = "cpu"
) -> list[str]:
    """Generate completions for a fixed prompt and print them. Returns the decoded strings."""
    tokenizer = Tokenizer()

    prompt = "Hello, I'm a language model"
    input_tokens = tokenizer.encode(prompt)

    generated_tokens = model.generate(
        tokens=input_tokens,
        generate_config=generate_config,
        device=device,
    )

    prompt_len = input_tokens.size(0)
    decoded: list[str] = []

    print()
    for i in range(generate_config.num_return_sequences):
        # Skip the prompt; return only the newly generated continuation.
        new_tokens = generated_tokens[i, prompt_len:].tolist()
        text = tokenizer.decode(new_tokens)
        decoded.append(text)
        console.print(f"{i}: {text}")
    print()

    return decoded