import logging

import torch.nn as nn
from attention import MaskedSelfAttention
from mixture_of_experts import MixtureOfExperts
from normalization import LayerNorm, RMSNorm
from train import GPTConfig

logger = logging.getLogger(__name__)


class Block(nn.Module):
    """Transformer decoder block"""

    def __init__(self, config: GPTConfig, norm: str = "rms"):
        super().__init__()

        self.config = config

        if norm == "layer":
            self.norm_1 = LayerNorm(config.n_embed)
            self.norm_2 = LayerNorm(config.n_embed)
        elif norm == "rms":
            self.norm_1 = RMSNorm(config.n_embed)
            self.norm_2 = RMSNorm(config.n_embed)

        self.attention = MaskedSelfAttention(config, n_group=6)

        # self.feed_forward = FeedForward(config)
        self.MoE = MixtureOfExperts(config.n_embed, 4 * config.n_embed, 32)

    def forward(self, x, use_cache=False, cache=None, layer_index=None, pos: int = 0):
        shortcut = x
        x = self.norm_1(x)
        x = self.attention(x, use_cache, cache, layer_index, pos)
        x = x + shortcut

        shortcut = x
        x = self.norm_2(x)
        # x = self.feed_forward(x)
        x, aux_loss = self.MoE(x)
        x = x + shortcut

        return x, aux_loss
