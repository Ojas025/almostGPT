import logging

import torch
import torch.nn as nn
from train import DEVICE

logger = logging.getLogger(__name__)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, context_length, n_embed):
        super().__init__()
        self.embedding = nn.Embedding(context_length, n_embed)

    def forward(self, x, pos: int = 0):
        _, T, _ = x.shape

        # During inference, the sequence length is not always T
        # In case of a partially completed sequence, T will always be 1
        # Instead add the amount of tokens generated so far to account for positional embedding correctness
        pos = torch.arange(pos, pos + T, device=DEVICE, dtype=torch.long)
        pos_emb = self.embedding(pos).unsqueeze(0)  # [B,T,n_embed]

        return pos_emb


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, T, n_embed):
        super().__init__()

        positions = torch.arange(T, device=DEVICE)

        base = 10000.0
        denominator = torch.pow(
            base, 2 * torch.arange(n_embed // 2) / n_embed
        )  # Generate index 'i' for each timestep

        # For each timestep, one positional embedding
        embeddings = torch.zeros(T, n_embed, dtype=torch.float16)

        angles = positions / denominator
        embeddings[:, 0::2] = torch.sin(angles)  # sine for even positions

        embeddings[:, 1::2] = torch.cos(angles)  # cosine for odd positions

        self.register_buffer("PE", embeddings)

    def forward(self, x):
        _, T, _ = x.shape
        return self.PE[:T].unsqueeze(0)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, n_head: int, base: int = 10000):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.base = base

        self.sin_cache = None
        self.cos_cache = None

    @torch.no_grad()
    def _build_cache(self, x: torch.Tensor, pos: int = 0):
        if self.sin_cache is not None and pos + x.shape[1] <= self.sin_cache.shape[0]:
            return  # use the existing cache, no need to create or grow

        sequence_length = pos + x.shape[1]
        d_head = self.d_model // self.n_head

        w = 1 / (self.base ** (torch.arange(0, d_head, 2).float() / d_head)).to(
            x.device
        )  # 1 / 10000 ^ (2i / d_model)
        positions = torch.arange(sequence_length, device=x.device)

        theta = positions.unsqueeze(1) * w.unsqueeze(
            0
        )  # [sequence_length, 1] * [1, d_model/2] -> [sequence_length, d_model/2]

        theta = torch.repeat_interleave(theta, 2, dim=1)  # [sequence_length, d_model]

        self.sin_cache = theta.sin()
        self.cos_cache = theta.cos()

    def forward(self, x: torch.Tensor, pos: int = 0):
        # x -> [B, T, n_head or n_group, d_head]

        self._build_cache(x, pos)

        x_odd = x[..., 1::2]  # [B, T, n_dim, d_head/2]
        x_even = x[..., 0::2]  # [B, T, n_dim, d_head/2]

        cos = self.cos_cache[pos : pos + x.shape[1], 0::2]  # [T, d_head/2]
        sin = self.sin_cache[pos : pos + x.shape[1], 0::2]  # [T, d_head/2]

        rot_dim = x_even.shape[-1]
        cos = cos[..., :rot_dim]
        sin = sin[..., :rot_dim]

        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, d_head/2]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, T, 1, d_head/2]

        rotated_even = x_even * cos - x_odd * sin  # [B, T, n_dim, d_head/2]
        rotated_odd = x_even * sin + x_odd * cos  # [B, T, n_dim, d_head/2]

        rotated_x = torch.stack(
            [rotated_even, rotated_odd], dim=-1
        )  # [B, T, n_dim, d_head/2, 2]
        rotated_x = torch.flatten(
            rotated_x, 2, 3
        )  # handles interleaving odd,even indices -> [B, T, n_dim, d_head]

        return rotated_x
