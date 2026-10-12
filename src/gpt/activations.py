import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed Forward Network
    - (x.w1) x swish(x.w2) . w3
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim * 2)
        nn.init.zeros_(self.projection.bias)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.projection(x).chunk(2, dim=-1)  # split
        return x1 * self.silu(x2)
