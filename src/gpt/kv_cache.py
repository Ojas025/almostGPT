import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class KVCache(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_head: int,
        window: int,
        sink: int,
        device: str = "cpu",
        n_group: int | None = None,
        n_head: int | None = None,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.d_head = d_head
        self.device = device
        self.window = window
        self.sink = sink

        if n_head:
            self.attention_type = "mha"
            self.n_head = n_head
        elif n_group:
            self.attention_type = "gqa"
            self.n_group = n_group

        self.cache = [(None, None) for _ in range(n_layers)]

    def is_empty(self) -> bool:
        return self.cache[0][0] is None

    def reset(self) -> None:
        self.cache = [(None, None) for _ in range(self.n_layers)]

    def update(
        self, index: int, K_new: torch.Tensor, V_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        return K, V

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        K, V = self.cache[index]
        return K, V
