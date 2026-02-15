import logging

import torch.nn as nn
from activations import SwiGLU
from train import GPTConfig

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """Feed forward neural networm"""

    def __init__(self, config: GPTConfig, approximate=False, activation="swiglu"):
        super().__init__()

        self.config = config

        if activation == "gelu":
            self.network = nn.Sequential(
                nn.Linear(config.n_embed, 4 * config.n_embed),
                nn.GELU(approximate="tanh") if approximate else nn.GELU(),
                nn.Linear(4 * config.n_embed, config.n_embed),
            )
        elif activation == "relu":
            self.network = nn.Sequential(
                nn.Linear(config.n_embed, 4 * config.n_embed),
                nn.ReLU(),
                nn.Linear(4 * config.n_embed, config.n_embed),
            )
        elif activation == "swiglu":
            self.network = nn.Sequential(
                SwiGLU(config.n_embed, 4 * config.n_embed),
                nn.Linear(4 * config.n_embed, config.n_embed),
            )

    def forward(self, x):
        return self.network(x)
