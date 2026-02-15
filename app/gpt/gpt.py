import logging

import torch
import torch.nn as nn
from normalization import LayerNorm
from train import GPTConfig, Tokenizer, console
from transformer import Block

logger = logging.getLogger(__name__)


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
        aux_loss = 0

        for layer_index, block in enumerate(self.transformer.blocks):
            x, layer_aux_loss = block(x, use_cache, cache, layer_index, pos)
            aux_loss += layer_aux_loss

        aux_loss = aux_loss / self.config.n_layer

        x = self.transformer.layer_norm_f(x)  # [B,T,n_embed]

        loss = None
        logits = self.lm_head(x)  # [B,T,vocab_size]

        if targets is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss + aux_loss
