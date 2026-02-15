import logging

import torch
from train import DEVICE, console, model, step, val_loader

logger = logging.getLogger(__name__)


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
