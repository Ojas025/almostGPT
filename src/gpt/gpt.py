import logging

import torch
import torch.nn as nn
from .config import GenerateConfig, ModelConfig
from .console import console
from .kv_cache import KVCache
from .normalization import LayerNorm, RMSNorm
from .tokenizer import Tokenizer
from .transformer import Block

logger = logging.getLogger(__name__)


class GPT(nn.Module):
    """Generative Pretrained Transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = Tokenizer()

        final_norm = RMSNorm if config.norm_type == "rms" else LayerNorm

        self.transformer = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(config.vocab_size, config.n_embed),
                blocks=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                layer_norm_f=final_norm(config.n_embed),
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Weight sharing between token_embedding and the final linear layer.
            self.transformer.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02 * (2 * self.config.n_layer) ** -0.5  # scale residuals
            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device: str
    ) -> torch.optim.AdamW:
        """Build an AdamW optimizer with standard decay/no-decay parameter groups."""
        grad_params = {name: param for name, param in self.named_parameters()}
        grad_params = {
            name: param for name, param in grad_params.items() if param.requires_grad
        }

        # Decay is not applied to params with dim < 2 (biases, layerNorm, scale, etc).
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

        fused = "cuda" in device

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=[0.9, 0.95], eps=1e-8, fused=fused
        )

        return optimizer

    def top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only the k highest-scoring tokens; set the rest to -inf."""
        # Guard: torch.topk raises if k exceeds the vocabulary size, which a small
        # test model can easily trip.
        k = min(int(k), logits.size(-1))

        if k <= 0:
            return logits

        filtered = logits.clone()
        v, _ = torch.topk(filtered, k, dim=-1)
        mask = filtered < v[..., -1, None]
        filtered[mask] = float("-inf")

        return filtered

    def top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus sampling: keep the smallest set of tokens with cumulative prob >= p."""
        assert 0.0 < p < 1.0, "p must be within (0,1)"

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probabilities, dim=-1)

        mask = cumsum > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False  # keep at least the first token

        sorted_logits[mask] = float("-inf")

        # unsort
        filtered = torch.full_like(sorted_logits, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)

        return filtered

    def forward(
        self,
        index: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        cache: KVCache | None = None,
        pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the decoder stack.

        index: [B, T] input token ids. targets: [B, T] next-token ids, or None.
        Returns (logits [B, T, vocab_size], loss) where loss is None when targets is None.
        """
        _, T = index.size()

        if T > self.config.context_length:
            # Previously returned None here, which made the caller fail much later with
            # an unrelated "'NoneType' is not subscriptable".
            raise ValueError(
                f"context length {T} exceeds model context_length "
                f"{self.config.context_length}"
            )

        x = self.transformer.token_embedding(index)  # [B,T,n_embed]

        aux_loss = torch.zeros((), device=index.device)

        for layer_index, block in enumerate(self.transformer.blocks):
            x, layer_aux_loss = block(x, use_cache, cache, layer_index, pos)
            aux_loss = aux_loss + layer_aux_loss

        aux_loss = aux_loss / self.config.n_layer

        x = self.transformer.layer_norm_f(x)  # [B,T,n_embed]

        logits = self.lm_head(x)  # [B,T,vocab_size]

        if targets is None:
            # No loss without targets. Returning `None + aux_loss` used to raise
            # TypeError, which broke every evaluation-only call.
            return logits, aux_loss

        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss + aux_loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        generate_config: GenerateConfig,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample continuations for a single prompt.

        tokens: [T] prompt token ids. Returns [num_return_sequences, prompt_len + max_new_tokens].
        `max_new_tokens` counts *newly generated* tokens, not the total length.
        """
        self.eval()

        if tokens.dim() == 2:
            if tokens.size(0) != 1:
                raise ValueError(
                    "generate() expects a single prompt; got a batch of "
                    f"{tokens.size(0)}. Use one prompt per call."
                )
            tokens = tokens[0]

        tokens = tokens.unsqueeze(0)  # [1, T]
        tokens = tokens.repeat(generate_config.num_return_sequences, 1)
        tokens = tokens.to(device)

        prompt_len = tokens.size(1)
        target_len = prompt_len + generate_config.max_new_tokens

        cache = None

        if self.config.kv_cache.enabled:
            cache = KVCache(
                n_layers=self.config.n_layer,
                n_head=self.config.n_head,
                d_head=self.config.n_embed // self.config.n_head,
                window=self.config.kv_cache.window or self.config.context_length,
                sink=self.config.kv_cache.sink,
                device=device,
            )

        pos = 0

        while tokens.size(1) < target_len:
            if cache is None or cache.is_empty():
                tokens_curr = tokens  # prefill: the whole prompt
            else:
                tokens_curr = tokens[:, -1:]  # decode: one token per step

            logits, _ = self(tokens_curr, use_cache=cache is not None, cache=cache, pos=pos)
            logits = logits[:, -1, :]  # logits at the last position

            logits = logits / max(generate_config.temperature, 1e-6)
            logits = self.top_k(logits, generate_config.top_k)
            logits = self.top_p(logits, generate_config.top_p)

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)

            tokens = torch.cat((tokens, next_token), dim=1)
            pos += tokens_curr.size(1)

        return tokens