import logging
import math

logger = logging.getLogger(__name__)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 19073


def get_learning_rate(step):
    """
    Cosine decay learning rate schedule with linear warmup
    """

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    # progress after warmup
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    # maps -1 -> 0, 1 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)
