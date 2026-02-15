import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """Layern Normalization"""

    def __init__(self, n_embed, eps=1e-5):
        super().__init__()

        self.eps = eps

        # parameters
        self.gamma = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x - x_mean) / torch.sqrt(x_var + self.eps)

        output = self.gamma * x + self.beta

        return output


class RMSNorm(nn.Module):
    """
        Root Mean Square Layer Normalization

    :param dim: model dimensions
    :param eps: epsilon vale, default 1e-8
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()

        self.dim = dim
        self.eps = eps

        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        return (x / rms) * self.scale
