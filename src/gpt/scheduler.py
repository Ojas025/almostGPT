import logging
import math

from .config import TrainConfig

logger = logging.getLogger(__name__)


def get_learning_rate(step: int, train_config: TrainConfig) -> float:
    """Cosine-decay learning rate with linear warmup.

    Every schedule value comes from config; this module used to own `max_lr`,
    `warmup_steps` and `max_steps` as module constants, which made them
    unreachable from config.yml.
    """
    if step < train_config.warmup_steps:
        return train_config.learning_rate * (step + 1) / train_config.warmup_steps

    if step > train_config.max_steps:
        return train_config.learning_rate * train_config.min_lr_ratio

    # progress after warmup
    decay_ratio = (step - train_config.warmup_steps) / (
        train_config.max_steps - train_config.warmup_steps
    )
    assert 0 <= decay_ratio <= 1

    # maps -1 -> 0, 1 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return train_config.learning_rate * train_config.min_lr_ratio + coeff * (train_config.learning_rate - train_config.learning_rate * train_config.min_lr_ratio)