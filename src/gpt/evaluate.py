import logging

import torch
from .console import console

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model,
    val_loader,
    step: int,
    device: str = "cpu",
    use_amp: bool = False,
    max_batches: int = 20,
) -> float:
    """Average validation loss over `max_batches` batches. Returns a Python float."""
    model.eval()

    # reset data loader so every eval pass sees the same tokens
    val_loader.reset()
    val_loss_accum = 0.0

    for _ in range(max_batches):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # BLOCKER: autocast was hardcoded to CUDA. float16 autocast on CPU is
        # unsupported and this fired on every CPU run. `use_amp` is resolved once
        # from config (which already requires CUDA) rather than per-call here.
        with torch.autocast(
            device_type=device, dtype=torch.float16, enabled=use_amp and device == "cuda"
        ):
            _, loss = model(x, y)

        val_loss_accum += loss.detach()

    mean_loss = (val_loss_accum / max_batches).item()

    console.print(f"\n[green]Validation Loss: [/green]{mean_loss:.4f}\n")
    logger.info(f"Step {step}, Validation Loss {mean_loss:.4f}")

    return mean_loss