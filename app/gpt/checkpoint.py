import logging
import os
import random

import numpy as np
import torch
from train import DEVICE, console

from data import load_tokens

logger = logging.getLogger(__name__)


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
