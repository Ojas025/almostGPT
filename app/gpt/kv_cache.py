import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
