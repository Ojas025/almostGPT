import logging

import torch
import torch.nn as nn
from .activations import SwiGLU
from .config import ModelConfig

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        config: ModelConfig,
        approximate: bool = False,
        activation: str = "swiglu",
    ) -> None:
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
            # SwiGLU doubles its own projection internally (x1, x2 = chunk), so the
            # trailing Linear projects 4*n_embed back down to n_embed.
            self.network = nn.Sequential(
                SwiGLU(config.n_embed, 4 * config.n_embed),
                nn.Linear(4 * config.n_embed, config.n_embed),
            )
        else:
            raise ValueError(f"unknown activation: {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)