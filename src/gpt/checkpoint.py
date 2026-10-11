import logging
import os
import random

import numpy as np
import torch
from .console import console

from .data import load_tokens

logger = logging.getLogger(__name__)


def save_checkpoint(
    step: int,
    model,
    optimizer,
    scaler,
    train_loader,
    path: str = "checkpoints",
) -> None:
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        # Training progress
        "step": step,
        # Model state
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        # Dataloader state. `current_shard` is set only by the fineweb path, so it is read
        # defensively -- it does not exist for shakespeare.
        "dataloader_state": {
            "index": train_loader.index,
            "current_shard": getattr(train_loader, "current_shard", None),
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


def load_checkpoint(
    path: str,
    model,
    optimizer,
    scaler,
    train_loader,
    device: str = "cpu",
) -> int:
    # weights_only=False is deliberate: the checkpoint stores RNG state (numpy / python
    # random / torch generators) so a resumed run continues from the same stochastic
    # point. Those objects are not on torch's default allowlist, and torch >= 2.6
    # defaults weights_only=True, which made every resume fail with an
    # UnpicklingError. Safe here because we wrote this file ourselves.
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    train_loader.index = checkpoint["dataloader_state"]["index"]
    # If the dataset is not fineweb
    # Only re-open a shard if this loader actually has shards; shakespeare has none.
    if checkpoint["dataloader_state"]["current_shard"] is not None and getattr(
        train_loader, "shards", None
    ):
        train_loader.current_shard = (
            checkpoint["dataloader_state"]["current_shard"]
            if checkpoint["dataloader_state"]["current_shard"] is not None
            else None
        )
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
