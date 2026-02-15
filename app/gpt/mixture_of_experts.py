import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


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
    ):
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
        nn.init.normal(self.gate, std=0.02)

    def _compute_capacity(self, num_tokens: int) -> int:
        return max(
            self.min_capacity,
            math.ceil(self.capacity_factor * num_tokens / self.num_experts),
        )

    def forward(self, x: Tensor):
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
    """Top-1 Mixture of Experts Layer"""

    def __init__(self, d_model, d_hidden, num_experts):
        super().__init__()

        self.router = Router(d_model, num_experts, z_loss_coef=1e-3, aux_loss_coef=1e-2)

        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(num_experts)]
        )

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
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
        x_sorted = x_flat[sort_idx]
        prob_sorted = expert_prob[sort_idx]

        cum_counts = torch.cumsum(counts, dim=0)
        start_idx = cum_counts - counts

        output_sorted = torch.zeros_like(x_sorted)

        for e in range(self.num_experts):
            start = start_idx[e].item()
            end = cum_counts[e].item()

            if end <= start:
                continue

            end_cap = min(end, start + capacity)

            tokens_e = x_sorted[start:end_cap]
            out_e = self.experts[e](tokens_e)

            output_sorted[start:end_cap] = out_e

        output = torch.zeros_like(x_flat)
        output[sort_idx] = output_sorted

        output = output * prob_sorted.unsqueeze(-1)

        return output.view(B, T, D), aux_loss
