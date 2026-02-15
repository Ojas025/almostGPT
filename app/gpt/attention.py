import logging

import torch
import torch.nn as nn
from embeddings import RotaryPositionalEmbedding
from kv_cache import KVCache
from train import DEVICE, GPTConfig

logger = logging.getLogger(__name__)


class MaskedSelfAttention(nn.Module):
    """Masked/Causal Self Attention"""

    def __init__(
        self,
        config: GPTConfig,
        n_group,
        dropout=0.2,
        is_flash_attention=True,
        sliding_window: int = None,
        attention_sink: int = None,
    ):
        super().__init__()
        assert config.n_embed % config.n_head == 0, (
            "Embedding size must be divisible by number of heads"
        )
        assert config.n_head % n_group == 0, (
            "Number of heads must be divisible by number of groups"
        )

        self.is_flash_attention = is_flash_attention
        self.config = config
        self.d_head = config.n_embed // config.n_head
        self.scale = self.d_head**-0.5
        self.groups = n_group
        self.group_size = config.n_head // n_group

        # self.qkv_proj = nn.Linear(config.n_embed, 3*config.n_embed) # For Multi-head Attention

        # Sliding window attention
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink

        # For Grouped Multi-Query Attention
        self.q_proj = nn.Linear(config.n_embed, config.n_head * self.d_head)
        self.k_proj = nn.Linear(config.n_embed, n_group * self.d_head)
        self.v_proj = nn.Linear(config.n_embed, n_group * self.d_head)

        self.RoPE = RotaryPositionalEmbedding(self.d_head, config.n_head)

        # self.register_buffer("tril", torch.tril(torch.ones(config.context_length, config.context_length)))

        self.output_proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(dropout)

    def build_sliding_window_mask(
        self, q_len, kv_len, sliding_window, attention_sink
    ) -> torch.Tensor:
        i = torch.arange(q_len, device=DEVICE)[:, None]
        j = torch.arange(kv_len, device=DEVICE)[None, :]

        causal = i >= j
        sink = j < attention_sink
        local = j >= (i - sliding_window)

        return causal & (local | sink)

    def forward(
        self, x, use_cache=False, cache: KVCache = None, layer_index=None, pos: int = 0
    ):

        # Q = self.query(x) # [B,T,d_head]
        # K = self.key(x) # [B,T,d_head]
        # V = self.value(x) # [B,T,d_head]

        # If using cache and not the initial prompt, slice for sequence length = 1
        # if use_cache and not cache.is_empty():
        #     x = x[:,-1,:].unsqueeze(1)

        B, T, C = x.shape

        Q = self.q_proj(x)  # [B, T, n_head, d_head]
        K = self.k_proj(x)  # [B, T, n_group, d_head]
        V = self.v_proj(x)  # [B, T, n_group, d_head]

        # qkv = self.qkv_proj(x) # [B,T,3*C]
        # qkv = qkv.view(B, T,self.config.n_head, 3, self.d_head) # [B,T,n_head,3,d_head]
        # Q,K,V = qkv[...,0,:], qkv[...,1,:], qkv[...,2,:] # [B,T,n_head,d_head]

        # K = K.view(B, T, self.groups, self.config.d_head).transpose(1, 2)
        # V = V.view(B, T, self.groups, self.config.d_head).transpose(1, 2)

        Q = self.RoPE(Q, pos)  # [B, T, n_head, d_head]
        K = self.RoPE(K, pos)  # [B, T, n_group, d_head]

        Q = Q.transpose(1, 2)  # [B, n_head, T, d_head]
        K_new = K.transpose(1, 2)  # [B, n_group, T, d_head]
        V_new = V.transpose(1, 2)  # [B, n_group, T, d_head]
        # Q = Q.transpose(1, 2) # [B, n_head, T, d_head]
        # K_new = K.transpose(1, 2) # [B, n_head, T, d_head] for training, [B, n_head, 1, d_head] for inference
        # V_new = V.transpose(1, 2) # [B, n_head, T, d_head]

        if use_cache:
            assert cache is not None and layer_index is not None
            K, V = cache.update(layer_index, K_new, V_new)

            if T == 1:
                sliding_window_mask = None
        else:
            K, V = K_new, V_new

        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        if self.sliding_window is not None:
            sliding_window_mask = self.build_sliding_window_mask(
                T, K.size(2), self.sliding_window, self.attention_sink
            )
            sliding_window_mask = sliding_window_mask[None, None, :, :]

        if self.is_flash_attention:
            is_causal = (True if cache is None else False) and (
                sliding_window_mask is None
            )
            output = nn.functional.scaled_dot_product_attention(
                Q,
                K,
                V,
                is_causal=is_causal,
                scale=self.scale,
                attn_mask=sliding_window_mask,
            )
        else:
            # matrix multiplication happens on last 2 dimensions
            # [B, n_head, T, d_head] @ [B, n_head, d_head, T]
            W = (Q @ K.transpose(-2, -1)) * self.scale  # [B, n_head, T, T]

            # if T > 1:
            # mask = self.tril[:T, :K.size(2)].unsqueeze(0).unsqueeze(0)
            # W = W.masked_fill(mask == 0, torch.finfo(W.dtype).min)

            if sliding_window_mask is not None:
                W = W.masked_fill(~sliding_window_mask, torch.finfo(W.dtype).min)

            W = torch.softmax(W, dim=-1)  # apply row-wise

            # [B, n_head, T, T] @ [B, n_head, T, d_head]
            output = W @ V  # [B, n_head, T, d_head]

        output = output.transpose(1, 2)  # [B, T, n_head, d_head]
        output = output.contiguous().view(B, T, C)  # n_head * d_head -> C

        # [B, T, C]
        return self.output_proj(output)
