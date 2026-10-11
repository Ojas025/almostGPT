import logging

import torch
import torch.nn as nn
from .attention import MaskedSelfAttention
from .config import ModelConfig
from .feed_forward import FeedForward
from .mixture_of_experts import MixtureOfExperts
from .normalization import LayerNorm, RMSNorm

logger = logging.getLogger(__name__)


class Block(nn.Module):
    """Transformer decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        if config.norm_type == "layer":
            self.norm_1 = LayerNorm(config.n_embed)
            self.norm_2 = LayerNorm(config.n_embed)
        else:
            self.norm_1 = RMSNorm(config.n_embed)
            self.norm_2 = RMSNorm(config.n_embed)

        # n_group comes from config and is validated against n_head by ModelConfig, so
        # an incompatible grouping fails at config load rather than here.
        self.attention = MaskedSelfAttention(
            config,
            n_group=config.attention.n_group,
            is_flash_attention=config.attention.is_flash_attention,
            sliding_window=config.attention.sliding_window,
            attention_sink=config.attention.attention_sink,
            dropout=config.attention.dropout,
        )

        if config.moe.enabled:
            self.feed_forward: nn.Module = MixtureOfExperts(config)
        else:
            self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache=None,
        layer_index: int | None = None,
        pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (block output [B,T,n_embed], auxiliary loss scalar)."""
        shortcut = x
        x = self.norm_1(x)
        x = self.attention(x, use_cache, cache, layer_index, pos)
        x = x + shortcut

        shortcut = x
        x = self.norm_2(x)

        # FeedForward returns a bare tensor; MixtureOfExperts returns (output, aux_loss).
        # Normalise here so the rest of the block sees one shape either way.
        ff_out = self.feed_forward(x)
        if isinstance(ff_out, tuple):
            x, aux_loss = ff_out
        else:
            x, aux_loss = ff_out, torch.zeros((), device=x.device, dtype=x.dtype)

        x = x + shortcut

        return x, aux_loss