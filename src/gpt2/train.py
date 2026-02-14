"""
Changes in GPT-2 from the original transformer paper:
    - Layer Normalization moved to the input of each sub-block
    - Additional LayerNorm after final self-attention block
"""

"""
Stuff implemented:
    - Use of GeLU instead of ReLU to deal with dead activations, as GeLU always contributes a local gradient
    - Use SwiGLU over GeLU, uses Swish activation (x.sigmoid(x)) with Gated Linear Units
    - Includes ugly number optimization
    - Gradient clipping to a norm -> 1.0
    - Learning rate schedule using cosine decay
    - Add parameter decay to introduce regularization
    - Implement gradient accumulation to increase batch_size
    - Implement Learned Positional Embedding
    - Implement Sinusoidal Positional Embedding 
    - Added Gradient Scaling to avoid gradient underflow due to Automatic Mixed Precision (float16)
    - Added top_k and top_p filtering
    - Add RMS Norm over LayerNorm to reduce computations
    - Added checkpointing to resume training state
    - Implement Rotary Positional Embeddings (RoPE)
    - Implement Grouped Multi-Query Attention
    - Implement Sliding Window Attention
    - Implement Rolling KV Cache
    
TODO:
    - Add Mixture-of-Experts
    - LoRA Fine Tuning
    - Custom Flash Attention? (Cuda)
"""

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import time

import tiktoken
import torch
import torch.nn as nn
from logger import configure_logging
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader, Dataset

configure_logging()

logger = logging.getLogger(__name__)

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = Path("../data/shakespeare.txt").resolve()

panel = Panel(
    "An implementation of GPT-like decoder-only transformer", title="AlmostGPT"
)
console.print(panel)
console.print(f"Using device: [yellow]{DEVICE}[/yellow]")


def load_data(path: str):
    with open(path, "r") as file:
        text = file.read()

    return text


@dataclass
class GPTConfig:
    """GPT architecture configurations"""

    context_length: int = 1024  # max context length
    vocab_size: int = 50257  # number of tokens: 50000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension


# flash_attention_kernel = load(name="flash_attention", sources=["flash_attention_kernel.cu"])

# def flash_attention(q,k,v,scale):
#     return flash_attention_kernel(q,k,v,scale)

# class FlashAttention(torch.autograd.Function):
#     """Flash attention implementation"""
#     # TODO: implement CUDA kernel

#     @staticmethod
#     def forward(ctx, q, k, v, scaling):
#         ctx.save_for_backward(q,k,v,scaling)
#         output = flash_attention(q,k,v,scaling)
#         return output

#     @staticmethod
#     def backward(ctx, grad_outputs):
#         q, k, v, scale = ctx.saved_tensors

#         grad_q, grad_k, grad_v = flash_attention_kernel(grad_outputs, q, k, v, scale)

#         return grad_q,grad_k,grad_v,None


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


class Expert(nn.Module):
    """Simple feed-forward expert"""

    def __init__(self, d_model, d_hidden, dropout: float = 0.0):
        super().__init__()

        self.expert = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return self.expert(x)


class Router(nn.Module):
    """Top-1 Router with capacity and load-balancing loss."""

    def __init__(
        self,
        d_model,
        num_experts,
        z_loss_coef: float,
        aux_loss_coef: float,
        k=2,
        capacity_factor=1.0,
        epsilon=1e-6,
        min_capacity: int = 4,
        training=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.min_capacity = min_capacity
        self.training = training
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal(self.gate, std=0.02)

    def _compute_capacity(self, num_tokens: int) -> int:
        return math.ceil(self.capacity_factor * num_tokens / self.num_experts)

    def forward(self, x: Tensor):
        B, T, D = x.shape

        num_tokens = B * T
        E = self.num_experts

        # Compute logits
        logits = self.gate(x)  # [B, T, E]

        # Calc probabilities
        probs = torch.softmax(logits, dim=-1)  # [B, T, E]

        capacity = self._compute_capacity(num_tokens)

        # Compute z-loss
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()

        if self.training:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u))  # gumbel noise

            noisy_logits = logits + g
            expert_index = noisy_logits.argmax(dim=-1)

            routing_weights = probs[torch.arange(num_tokens), expert_index]
        else:
            expert_index = logits.argmax(dim=-1)
            routing_weights = probs[torch.arange(num_tokens), expert_index]

        # Auxilliary loss
        tokens_per_expert = torch.zeros(E, device=x.device)
        tokens_per_expert.scatter_add_(
            0, expert_index, torch.ones(num_tokens, device=x.device)
        )

        f_i = (
            tokens_per_expert / num_tokens
        )  # fraction of tokens dispatched to expert i
        p_i = probs.mean(
            dim=0
        )  # fraction of the router probability allocated to expert i

        aux_loss = (
            self.aux_loss_coef * E * (f_i * p_i).sum() + z_loss * self.z_loss_coef
        )

        dispatch_mask = torch.zeros(
            num_tokens, E, capacity, dtype=torch.bool, device=x.device
        )
        combine_weights = torch.zeros(
            num_tokens, E, capacity, dtype=probs.dtype, device=x.device
        )

        for e in range(E):
            token_indices = torch.where(expert_index == e)

            if token_indices.numel() == 0:
                continue

            token_indices = token_indices[:capacity]

            slots = torch.arange(token_indices.numel(), device=x.device)

            dispatch_mask[num_tokens, e, slots] = True
            combine_weights[num_tokens, e, slots] = routing_weights[token_indices].to(
                combine_weights.dtype
            )

        return dispatch_mask, combine_weights, aux_loss, probs.detach()


class MixtureOfExperts(nn.Module):
    """Top-1 Mixture of Experts Layer"""

    def __init__(self, d_model, d_hidden, n_experts):
        super().__init__()

        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(n_experts)]
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def _dispatch(self):
        # expert input [C, D]
        # Extract token_idx, slot_idx from dispatch mask
        # Slice selected tokens from x
        # add to expert input
        pass

    def _combine(self):
        pass

    def forward(self, x):
        pass


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
        n_dim = x.shape[2]

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


class KVCache(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_head: int,
        d_head: int,
        window: int,
        sink: int,
        device="cuda",
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_head = n_head
        self.d_head = d_head
        self.device = device
        self.window = window
        self.sink = sink

        self.cache = [(None, None) for _ in range(n_layers)]

    def is_empty(self):
        return self.cache[0][0] is None

    def reset(self):
        self.cache = [(None, None) for _ in range(self.n_layers)]

    def update(self, index: int, K_new: torch.Tensor, V_new: torch.Tensor):
        K, V = self.cache[index]

        if K is None:
            K, V = K_new, V_new  # [B, n_head, T, d_head]
        else:
            K = torch.cat([K, K_new], dim=2)
            V = torch.cat([V, V_new], dim=2)

        if self.window is not None and K.size(2) > self.window + self.sink:
            K = torch.cat([K[:, :, : self.sink, :], K[:, :, -self.window :, :]], dim=2)
            V = torch.cat([V[:, :, : self.sink, :], V[:, :, -self.window :, :]], dim=2)

        self.cache[index] = (K, V)

        return K.clone(), V.clone()

    def get(self, index):
        K, V = self.cache[index]
        return K.clone(), V.clone()


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


class SwiGLU(nn.Module):
    """
    SwiGLU Feed Forward Network
    - (x.w1) x swish(x.w2) . w3
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim * 2)
        self.silu = nn.SiLU()

    def forward(self, x):
        x1, x2 = self.projection(x).chunk(2, dim=-1)  # split
        return x1 * self.silu(x2)


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
        self.feed_forward = FeedForward(config)

    def forward(self, x, use_cache=False, cache=None, layer_index=None, pos: int = 0):
        shortcut = x
        x = self.norm_1(x)
        x = self.attention(x, use_cache, cache, layer_index, pos)
        x = x + shortcut

        shortcut = x
        x = self.norm_2(x)
        x = self.feed_forward(x)
        x = x + shortcut

        return x


class GPT(nn.Module):
    """Generative Pretrained Transformer"""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.tokenizer = Tokenizer()

        self.transformer = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(
                    config.vocab_size, config.n_embed
                ),  # contextual embeddings
                # positional_embedding = LearnedPositionalEmbedding(config.context_length, config.n_embed), # learned positional embeddings
                # positional_embedding = SinusoidalPositionalEmbedding(config.context_length, config.n_embed) # sinusoidal positional embedding
                blocks=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # decoder blocks
                layer_norm_f=LayerNorm(config.n_embed),  # additional layernorm
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing between token_embedding and final linear layer
        self.transformer.token_embedding.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 * (2 * self.config.n_layer) ** -0.5  # scale residuals
            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # get params that require grad
        grad_params = {name: param for name, param in self.named_parameters()}
        grad_params = {
            name: param for name, param in grad_params.items() if param.requires_grad
        }

        # create param groups based on whether the parameter is to be decayed
        # decay is usually not applied on params with dim < 2 (biases, layerNorm, scale, etc)
        decay_params = [p for n, p in grad_params.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in grad_params.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)

        console.print(f"\nDecayed parameters: {n_decay}")
        console.print(f"Non-Decayed parameters: {n_no_decay}\n")

        fused = False
        if "cuda" in device:
            fused = True

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=[0.9, 0.95], eps=1e-8, fused=fused
        )

        return optimizer

    def top_k(self, logits, k):
        filtered = logits.clone()
        v, _ = torch.topk(filtered, k, dim=-1)
        mask = filtered < v[..., -1, None]
        filtered[mask] = float("-inf")

        return filtered

    def top_p(self, logits, p):
        assert 0.0 < p < 1.0, "p must be within [0,1]"

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probabilities, dim=-1)

        # mask all tokens with cumulative probability > p
        mask = cumsum > p
        mask[..., 0] = False  # keep at least the first token

        sorted_logits[mask] = float("-inf")

        # unsort
        filtered = torch.full_like(sorted_logits, float("-inf"))
        filtered.scatter_(1, sorted_indices, sorted_logits)

        return filtered

    def forward(self, index, targets=None, use_cache=False, cache=None, pos: int = 0):
        # index: [B,T]
        _, T = index.size()
        # H = self.config.n_head
        # D = self.config.n_embed // H

        if T > self.config.context_length:
            console.print(f"[red]Cannot forward context of length {T}[/red]")
            return

        x = self.transformer.token_embedding(index)  # [B,T,n_embed]

        # pos_emb = self.transformer.positional_embedding(x, pos)

        for layer_index, block in enumerate(self.transformer.blocks):
            x = block(x, use_cache, cache, layer_index, pos)

        x = self.transformer.layer_norm_f(x)  # [B,T,n_embed]

        loss = None
        logits = self.lm_head(x)  # [B,T,vocab_size]

        if targets is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_output_tokens: int = 200,
        num_return_sequences: int = 3,
        temperature: float = 1.0,
        k: int = 50,
        p: float = 0.95,
    ):
        self.eval()

        tokens = tokens.unsqueeze(0)  # [1, T]
        tokens = tokens.repeat(num_return_sequences, 1)  # [num_return_sequences, T]
        tokens = tokens.to(DEVICE)

        cache = KVCache(
            n_layers=model.config.n_layer,
            n_head=model.config.n_head,
            d_head=model.config.n_embed // model.config.n_head,
            window_size=model.config.context_length,
            device=DEVICE,
        )

        pos = 0

        while tokens.size(1) < max_output_tokens:
            tokens_curr = tokens if cache.is_empty() else tokens[:, -1:]
            logits, _ = self(tokens_curr, use_cache=True, cache=cache, pos=pos)
            logits = logits[:, -1, :]  # logits at the last position

            logits = logits / max(temperature, 1e-6)
            logits = self.top_k(logits, k)
            logits = self.top_p(logits, p)

            probabilities = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probabilities, 1)

            # append new col to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)
            pos += tokens_curr.size(1)

        return tokens


# --------------------------------------------------------------------


class Tokenizer:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, data):
        tokens = self.encoder.encode(data)
        tokens = torch.tensor(tokens, device=DEVICE)

        return tokens

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        return self.encoder.decode(ids)


import os

import numpy as np


def load_tokens(filepath: str):
    tokens = np.load(filepath)
    tokens = torch.tensor(tokens, dtype=torch.long)

    return tokens


from hellaswag import get_formatted_example, iterate_examples


class HellaSwagDataset(Dataset):
    def __init__(self, split="train"):
        self.examples = list(iterate_examples(split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        data, tokens, mask, label = get_formatted_example(example)

        return tokens, mask, label


class CustomDataLoader:
    """Data loader class implementation"""

    def __init__(self, B, T, split="train", dataset="shakespeare"):
        self.B = B
        self.T = T
        self.dataset = dataset

        if dataset == "shakespeare":
            data = load_data(DATASET_PATH)
            self.tokenizer = Tokenizer()
            self.tokens = self.tokenizer.encode(data)
        elif dataset == "fineweb":
            dataset_dir = Path(__file__).parent / "edu_fineweb10B"
            shards = os.listdir(dataset_dir)
            self.shards = [f"{dataset_dir}/{s}" for s in shards if split in s]
            # self.current_shard = 0
            # self.tokens = load_tokens(self.shards[self.current_shard])

            assert len(self.shards) > 0, f"No shards found for split {split}"

        assert split in {"train", "val"}

        self.reset()
        console.print(f"Loaded {len(self.tokens)} tokens")

    def reset(self):
        if self.dataset == "fineweb":
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
        self.index = 0

    def next_batch(self):
        buffer = self.tokens[self.index : self.index + self.B * self.T + 1]

        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)

        self.index += self.B * self.T

        if self.index + (self.B * self.T + 1) > len(self.tokens):
            if self.dataset == "fineweb":
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
            self.index = 0

        return x, y


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 19073


def get_learning_rate(step):
    """
    Cosine decay learning rate schedule with linear warmup
    """

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    # progress after warmup
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    # maps -1 -> 0, 1 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def train_model(step: int):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(gradient_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = model(x, y)

        # scale gradient using MEAN to account for gradient accumulation
        loss /= gradient_accumulation_steps
        loss_accum += loss.detach()

        scaler.scale(loss).backward()

    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # determine learning rate for this iteration
    lr = get_learning_rate(step)

    # update learning rate in params
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # without scaling
    # optimizer.step()

    # with scaling
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()  # wait for GPU to finish

    return loss_accum, lr, norm


@torch.no_grad()
def evaluate_model():
    model.eval()

    # reset data loader
    val_loader.reset()
    val_loss_accum = 0.0
    val_loss_steps = 20

    for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.autocast(dtype=torch.float16, device_type="cuda"):
            _, loss = model(x, y)

        val_loss_accum += loss.detach()
        loss /= val_loss_steps

    console.print(f"\n[green]Validation Loss: [/green]{val_loss_accum.item():.4f}\n")
    logger.info(f"Step {step}, Validation Loss {val_loss_accum.item():.4f}")


def save_checkpoint(step, model, optimizer, scaler, train_loader, path="checkpoints"):
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        # Training progress
        "step": step,
        # Model state
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        # Dataloader state
        "dataloader_state": {
            "index": train_loader.index,
            "current_shard": train_loader.current_shard,
        },
        # RNG state
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }

    temp_path = f"{path}/latest.pt.temp"
    final_path = f"{path}/latest.pt"

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, final_path)

    console.print("[green]Checkpoint saved[/green]")


def load_checkpoint(path, model, optimizer, scaler, train_loader):
    checkpoint = torch.load(path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    train_loader.index = checkpoint["dataloader_state"]["index"]
    # If the dataset is not fineweb
    if checkpoint["dataloader_state"]["current_shard"] is not None:
        train_loader.current_shard = checkpoint["dataloader_state"]["current_shard"]
        train_loader.tokens = load_tokens(
            train_loader.shards[train_loader.current_shard]
        )

    torch.set_rng_state(checkpoint["rng_state"]["torch"])
    np.random.set_state(checkpoint["rng_state"]["numpy"])
    random.setstate(checkpoint["rng_state"]["python"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])

    console.print("[green]Checkpoint loaded[/green]")

    return checkpoint["step"]


def get_predicted_row(tokens, mask, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_tokens = tokens[..., 1:, :].contiguous()

    flattened_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flattened_tokens = shifted_tokens.view(-1)

    losses = nn.CrossEntropyLoss()(flattened_logits, flattened_tokens, reduction="none")
    losses = losses.view(tokens.size(0), -1)

    shifted_mask = mask[..., 1:, :].contiguous()
    masked_loss = losses * shifted_mask

    sum_loss = masked_loss.sum(dim=1)
    avg_loss = sum_loss / shifted_mask.sum(dim=1)

    prediction = avg_loss.argmin().item()

    return prediction


def completion():
    num_return_sequences = 3
    max_length = 32

    tokenizer = Tokenizer()

    prompt = "Hello, I'm a language model"
    input_tokens = tokenizer.encode(prompt)

    generated_tokens = model.generate(
        tokens=input_tokens,
        max_output_tokens=max_length,
        num_return_sequences=num_return_sequences,
    )

    print()
    for i in range(num_return_sequences):
        tokens = generated_tokens[i, :max_length].tolist()
        decoded_tokens = tokenizer.decode(tokens)
        console.print(f"{i}: {decoded_tokens}")
    print()


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")

    total_batch_size = 524288  # ~0.5M in number of tokens
    B = 4
    T = 1024

    assert total_batch_size % (B * T) == 0, "total_batch_size is divisible by B*T"

    gradient_accumulation_steps = total_batch_size // (B * T)

    console.print(f"\nTotal desired batch size: {total_batch_size}")
    console.print(f"Gradient accumulation steps: {gradient_accumulation_steps}\n")

    # init model
    model = GPT(GPTConfig(vocab_size=50304))  # remove ugly number 50257 -> 50304
    model.eval()
    model.to(DEVICE)

    if hasattr(torch, "compile"):
        model.compile()

    console.print("Initialized model")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=[0.9, 0.95], eps=1e-8)
    train_loader = CustomDataLoader(B, T, split="train", dataset="shakespeare")
    val_loader = CustomDataLoader(B, T, split="val", dataset="shakespeare")

    # Hellaswag
    hellaswag_dataset = HellaSwagDataset(split="val")
    hellaswag_dataloader = DataLoader(hellaswag_dataset, batch_size=1, shuffle=True)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=DEVICE
    )
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))

    resume_checkpoint_path = None

    if resume_checkpoint_path is not None:
        step = load_checkpoint(
            resume_checkpoint_path, model, optimizer, scaler, train_loader
        )
    else:
        step = 0

    try:
        while step < max_steps:
            is_last_step = step == (max_steps - 1)

            # Evaluate
            if step % 250 == 0:
                evaluate_model()

            # Save checkpoint
            if step > 0 and (step % 5000 == 0 or is_last_step):
                if is_last_step:
                    console.print("\n[green]Saving final model...[/green]")

                save_checkpoint(step, model, optimizer, scaler)

            # Hellaswag
            if step > 0 and (step % 250 == 0 or is_last_step):
                num_correct_norm = 0
                num_total = 0

                with torch.no_grad():
                    for tokens, mask, label in hellaswag_dataloader:
                        tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)

                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits, loss = model(tokens)

                        predicted_norm = get_predicted_row(tokens, mask, logits)

                        num_correct_norm += int(predicted_norm == label.item())
                        num_total += 1

                accuracy = num_correct_norm / num_total

                console.print(f"\n[green]Hellaswag Accuracy: [/green]{accuracy:.4f}\n")
                logger.info(f"Step {step}, Hellaswag accuracy {accuracy}")

            t0 = time()
            loss_accum, lr, norm = train_model(step)
            t1 = time()

            tokens_processed = (
                train_loader.B * train_loader.T * gradient_accumulation_steps
            )
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt

            print(
                f"Step {step} | Loss: {loss_accum.item():.4f} | lr: {lr:.4e} | Norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}"
            )
            logger.info(f"Step {step}, Loss: {loss_accum.item():.4f}")

            step += 1
    except KeyboardInterrupt:
        console.print("[red]Keyboard interrupt detected[/red]")
        console.print("[red]Saving checkpoint...[/red]")

        save_checkpoint(step, model, optimizer, scaler)
        exit(1)
