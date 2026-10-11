import logging
import math

import torch
import torch.nn as nn
from .config import ModelConfig
from torch import Tensor

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """Simple feed-forward expert"""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.expert = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.expert(x)


class Router(nn.Module):
    """Top-1 Router with capacity and load-balancing loss."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        z_loss_coef: float,
        aux_loss_coef: float,
        k: int = 2,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        min_capacity: int = 4,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.min_capacity = min_capacity
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # Must be `normal_` (trailing underscore); `nn.init.normal` does not exist and
        # raised AttributeError on every model construction.
        nn.init.normal_(self.gate.weight, std=0.02)

    def _compute_capacity(self, num_tokens: int) -> int:
        return max(
            self.min_capacity,
            math.ceil(self.capacity_factor * num_tokens / self.num_experts),
        )

    def forward(self, x: Tensor) -> tuple[dict, Tensor]:
        B, T, D = x.shape

        num_tokens = B * T
        E = self.num_experts

        x_flat = x.view(num_tokens, D)

        # Compute logits
        logits = self.gate(x_flat)  # [num_tokens, E]

        # Calc probabilities
        probs = torch.softmax(logits, dim=-1)  # [num_tokens, E]

        capacity = self._compute_capacity(num_tokens)

        # Compute z-loss
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()

        # if self.training:
        #     eps = 1e-9
        #     u = torch.rand_like(logits)
        #     g = -torch.log(-torch.log(u + eps) + eps)  # gumbel noise

        #     noisy_logits = logits + g
        #     expert_index = noisy_logits.argmax(dim=-1)
        #     expert_index_flat = expert_index.view(-1)

        #     routing_weights = probs_flat[torch.arange(num_tokens), expert_index_flat]
        # else:
        #     expert_index = logits.argmax(dim=-1)
        #     expert_index_flat = expert_index.view(-1)
        #     routing_weights = probs[torch.arange(num_tokens), expert_index_flat]

        expert_index = logits.argmax(dim=-1)
        expert_prob = probs[torch.arange(num_tokens), expert_index]

        # count tokens per expert
        counts = torch.bincount(expert_index, minlength=E)

        # Auxilliary loss
        f_i = counts.float() / num_tokens  # fraction of tokens dispatched to expert i
        p_i = probs.mean(
            dim=0
        )  # fraction of the router probability allocated to expert i

        aux_loss = (
            self.aux_loss_coef * E * (f_i * p_i).sum() + z_loss * self.z_loss_coef
        )

        # dispatch_mask = torch.zeros(
        #     num_tokens, E, capacity, dtype=torch.bool, device=x.device
        # )
        # combine_weights = torch.zeros(
        #     num_tokens, E, capacity, dtype=probs.dtype, device=x.device
        # )

        # for e in range(E):
        #     token_indices = (expert_index_flat == e).nonzero(as_tuple=True)[0]

        #     if token_indices.numel() == 0:
        #         continue

        #     token_indices = token_indices[:capacity]

        #     slots = torch.arange(token_indices.numel(), device=x.device)

        #     dispatch_mask[token_indices, e, slots] = True
        #     combine_weights[token_indices, e, slots] = routing_weights[
        #         token_indices
        #     ].to(combine_weights.dtype)

        routing = {
            "expert_index": expert_index,
            "expert_prob": expert_prob,
            "counts": counts,
            "capacity": capacity,
        }

        return routing, aux_loss


class MixtureOfExperts(nn.Module):
    """Top-1 Mixture of Experts Layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        d_model = config.n_embed
        d_hidden = config.n_embed * config.moe.d_hidden_multiplier
        num_experts = config.moe.num_experts

        self.router = Router(
            d_model,
            num_experts,
            z_loss_coef=config.moe.z_loss_coef,
            aux_loss_coef=config.moe.aux_loss_coef,
            k=config.moe.top_k,
            capacity_factor=config.moe.capacity_factor,
            min_capacity=config.moe.min_capacity,
        )

        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden, dropout=config.moe.dropout) for _ in range(num_experts)]
        )

        # Routing metrics, refreshed on every forward. Initialised here so they are
        # readable before the first call rather than raising AttributeError.
        self.last_dropped_tokens = 0
        self.last_total_tokens = 0
        self.last_drop_rate = 0.0
        self.last_expert_load = torch.zeros(num_experts)
        self.last_capacity = 0

    @property
    def drop_rate(self) -> float:
        """Fraction of tokens discarded by expert capacity on the last forward."""
        return self.last_drop_rate

    # def _dispatch(self, x: Tensor, dispatch: Tensor, e: int, capacity: int):
    #     _, D = x.shape

    #     expert_input = torch.zeros(capacity, D, dtype=x.dtype, device=x.device)

    #     token_idx, slot_idx = dispatch[:, e, :].nonzero(as_tuple=True)  # [T, E, C]

    #     if token_idx.numel() > 0:
    #         input_tokens = x[token_idx]
    #         expert_input[slot_idx] = input_tokens

    #     return expert_input

    # def _combine(
    #     self, expert_output: Tensor, combine_weighs: Tensor, e: int, output: Tensor
    # ):
    #     token_idx, slot_idx = combine_weighs[:, e, :].nonzero(as_tuple=True)

    #     if token_idx.numel() > 0:
    #         weights = combine_weighs[token_idx, e, slot_idx].unsqueeze(-1)  # [k, 1]
    #         contribution = expert_output[slot_idx].to(output.device) * weights.to(
    #             output.device
    #         )
    #         output.index_add_(
    #             0, token_idx.to(output.device), contribution.to(output.dtype)
    #         )

    #     return output

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape

        x_flat = x.reshape(B * T, D)

        routing, aux_loss = self.router(x)

        expert_index = routing["expert_index"]
        expert_prob = routing["expert_prob"]
        counts = routing["counts"]
        capacity = routing["capacity"]

        # output = torch.zeros_like(x_flat)

        # for e, expert in enumerate(self.experts):
        #     expert_input = self._dispatch(x_flat, dispatch_mask, e, capacity)
        #     expert_output = expert(expert_input)
        #     output = self._combine(expert_output, combine_weights, e, output)

        # output = output.reshape(B, T, D)

        # Sort tokens by expert
        sorted_expert, sort_idx = torch.sort(expert_index)

        # tokens assigned to the same expert are contiguous in memory - efficiency
        x_sorted = x_flat[sort_idx]
        prob_sorted = expert_prob[sort_idx]

        # to get token indices per expert, calc cumulative sum of counts
        cum_counts = torch.cumsum(counts, dim=0)

        # start index for each expert
        start_idx = cum_counts - counts

        output_sorted = torch.zeros_like(x_sorted)

        # Tokens past an expert's capacity are left as zeros in `output_sorted` and are
        # then scattered back to their original positions. That is valid Switch-style
        # behaviour, but it was completely invisible -- you could not tell a learning
        # model from one silently discarding most of its tokens. These counters make the
        # drop rate observable. Training math is unchanged.
        dropped_tokens = 0
        expert_load = torch.zeros(self.num_experts)

        for e in range(self.num_experts):
            start = start_idx[e].item()
            end = cum_counts[e].item()

            if end <= start:
                continue

            # currently adding capacity to the num of tokens per expert
            # drops if excess
            end_cap = min(end, start + capacity)

            expert_load[e] = end - start

            if end_cap < end:
                dropped_tokens += end - end_cap

            tokens_e = x_sorted[start:end_cap]
            out_e = self.experts[e](tokens_e)

            # consists of outputs per expert in sorted order
            output_sorted[start:end_cap] = out_e

        output = torch.zeros_like(x_flat)
        output[sort_idx] = output_sorted * prob_sorted.unsqueeze(-1)

        # Observability only -- deliberately not returned, so the (output, aux_loss)
        # signature every caller depends on is untouched.
        self.last_dropped_tokens = dropped_tokens
        self.last_total_tokens = B * T
        self.last_drop_rate = dropped_tokens / (B * T) if B * T else 0.0
        self.last_expert_load = expert_load
        self.last_capacity = capacity

        return output.view(B, T, D), aux_loss
