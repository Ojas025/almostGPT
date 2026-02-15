import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.base = base
        self.base.weight.requires_grad_(False)

        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.empty(base.out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        base_output = self.base(x)
        lora_output = (x @ self.A.T) @ self.B.T

        return base_output + lora_output * self.scale
