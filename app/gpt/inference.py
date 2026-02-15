import logging

import torch.nn as nn
from tokenizer import Tokenizer
from train import console, model

logger = logging.getLogger(__name__)


def get_predicted_row(tokens, mask, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_tokens = tokens[..., 1:, :].contiguous()

    flattened_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flattened_tokens = shifted_tokens.view(-1)

    losses = nn.CrossEntropyLoss()(flattened_logits, flattened_tokens, reduction="none")
    losses = losses.view(tokens.size(0), -1)

    shifted_mask = mask[..., 1:, :].contiguous()
    masked_loss = losses * shifted_mask

    sum_loss = masked_loss.sum(dim=1)
    avg_loss = sum_loss / shifted_mask.sum(dim=1)

    prediction = avg_loss.argmin().item()

    return prediction


def completion():
    num_return_sequences = 3
    max_length = 32

    tokenizer = Tokenizer()

    prompt = "Hello, I'm a language model"
    input_tokens = tokenizer.encode(prompt)

    generated_tokens = model.generate(
        tokens=input_tokens,
        max_output_tokens=max_length,
        num_return_sequences=num_return_sequences,
    )

    print()
    for i in range(num_return_sequences):
        tokens = generated_tokens[i, :max_length].tolist()
        decoded_tokens = tokenizer.decode(tokens)
        console.print(f"{i}: {decoded_tokens}")
    print()
