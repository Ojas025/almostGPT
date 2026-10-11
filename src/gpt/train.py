"""Training entrypoint for almostGPT.

Everything tunable comes from `config.yml` via `AppConfig`. Pick a run with:

    python train.py                 # uses the `defaults:` block
    python train.py --profile smoke # tiny CPU run for verification

This module owns no configuration of its own; it is the only place that wires the
model, data, optimizer and loop together.
"""

import argparse
import logging
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
from .checkpoint import load_checkpoint, save_checkpoint
from .config import AppConfig
from .console import console
from .data import CustomDataLoader, HellaSwagDataset
from .evaluate import evaluate_model
from .gpt import GPT
from .inference import completion, get_predicted_row
from ..logger import configure_logging
from rich.panel import Panel
from .scheduler import get_learning_rate
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

"""
Changes in GPT-2 from the original transformer paper:
    - Layer Normalization moved to the input of each sub-block
    - Additional LayerNorm after final self-attention block

Implemented here:
    - GeLU instead of ReLU; SwiGLU variant
    - Ugly-number optimization (vocab padded to a multiple of 64)
    - Gradient clipping
    - Cosine LR schedule with linear warmup
    - Weight decay on matrices only
    - Gradient accumulation
    - Learned / sinusoidal / rotary (RoPE) positional embeddings
    - AMP + GradScaler
    - top-k and top-p sampling
    - RMSNorm
    - Checkpointing with RNG state
    - Grouped multi-query attention
    - Sliding-window attention
    - Rolling KV cache
    - Mixture-of-Experts routing with load-balancing and z-loss
    - LoRA adapters
"""


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training, so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an almostGPT model.")
    parser.add_argument(
        "--config", default=None, help="Path to config.yml (default: src/config.yml)"
    )
    parser.add_argument(
        "--profile", default=None, help="Profile name under `profiles:` in config.yml"
    )
    parser.add_argument(
        "--resume", default=None, help="Checkpoint path to resume from"
    )

    return parser.parse_args()


def train_model(
    step: int,
    model: GPT,
    optimizer,
    scaler,
    train_loader: CustomDataLoader,
    config: AppConfig,
    device: str,
) -> tuple[torch.Tensor, float, float]:
    """Run one optimizer step (over all micro-batches). Returns (loss, lr, grad_norm)."""
    train_config = config.train
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for _ in range((train_config.total_batch_size // (train_config.batch_size * train_config.context))):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # BLOCKER: this was hardcoded to `device_type="cuda"`. float16 autocast is not
        # supported on CPU, so every CPU run crashed here. `use_amp` is False off-CUDA.
        with torch.autocast(
            device_type=device, dtype=torch.float16, enabled=(train_config.amp and device == "cuda")
        ):
            _, loss = model(x, y)

        # scale gradient using MEAN to account for gradient accumulation
        loss = loss / (train_config.total_batch_size // (train_config.batch_size * train_config.context))
        loss_accum = loss_accum + loss.detach()

        scaler.scale(loss).backward()

    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip)

    lr = get_learning_rate(step, train_config)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    scaler.step(optimizer)
    scaler.update()

    if device == "cuda":
        torch.cuda.synchronize()

    return loss_accum, lr, float(norm)


def probe_loss(model: GPT, loader: CustomDataLoader, config: AppConfig, device: str) -> float:
    """Loss on one fixed batch, for a like-for-like comparison across steps."""
    loader.reset()
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)

    return float(loss)


def main() -> None:
    args = parse_args()
    configure_logging()

    config = AppConfig.from_yaml(args.config, profile=args.profile)
    train_config = config.train
    device = train_config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(train_config.seed)

    console.print(
        Panel(
            "An implementation of GPT-like decoder-only transformer", title="AlmostGPT"
        )
    )

    if config.profile:
        console.print(f"Profile: [cyan]{config.profile}[/cyan]")

    console.print(f"Using device: [yellow]{device}[/yellow]")
    console.print(f"Dataset: [cyan]{config.data.dataset}[/cyan] at {config.data.path}")

    model = GPT(config.model)
    console.print(
        f"Initialized model: {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # BLOCKER: `model.compile()` was wrong -- the API is `torch.compile(model)`,
    # which returns a wrapped module. Compilation is also slow to warm up, so it is
    # opt-in and never used for the CPU smoke profile.
    if train_config.compile and device == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    model.to(device)

    train_loader = CustomDataLoader(
        train_config.batch_size, train_config.context, config.data, split="train"
    )
    val_loader = CustomDataLoader(
        train_config.batch_size, train_config.context, config.data, split="val"
    )

    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        device=device,
    )
    scaler = torch.amp.GradScaler(device=device, enabled=(train_config.amp and device == "cuda"))

    hellaswag_dataloader = None
    if train_config.run_hellaswag:
        # Off by default: constructing this downloads ~50MB on first use.
        hellaswag_dataset = HellaSwagDataset(split="val")
        hellaswag_dataloader = DataLoader(hellaswag_dataset, batch_size=1, shuffle=True)

    step = 0
    if args.resume is not None:
        step = load_checkpoint(args.resume, model, optimizer, scaler, train_loader, device)

    # Captured before training starts so the smoke run can prove the loss actually
    # dropped on an identical batch.
    baseline = probe_loss(model, train_loader, config, device)
    console.print(f"Probe loss before training: [yellow]{baseline:.4f}[/yellow]")

    console.print(
        f"\nTotal batch size: {train_config.total_batch_size} | "
        f"Gradient accumulation steps: {(train_config.total_batch_size // (train_config.batch_size * train_config.context))}\n"
    )

    try:
        while step < train_config.max_steps:
            is_last_step = step == (train_config.max_steps - 1)

            if step % train_config.eval_interval == 0:
                evaluate_model(
                    model,
                    val_loader,
                    step,
                    device=device,
                    use_amp=(train_config.amp and device == "cuda"),
                )

            if step > 0 and (
                step % train_config.checkpoint_interval == 0 or is_last_step
            ):
                if is_last_step:
                    console.print("\n[green]Saving final model...[/green]")

                save_checkpoint(step, model, optimizer, scaler, train_loader)

            if hellaswag_dataloader is not None and (
                step % train_config.eval_interval == 0 or is_last_step
            ):
                num_correct = 0
                num_total = 0

                model.eval()
                with torch.no_grad():
                    for tokens, mask, label in hellaswag_dataloader:
                        tokens, mask = tokens.to(device), mask.to(device)

                        with torch.autocast(
                            device_type=device,
                            dtype=torch.float16,
                            enabled=(train_config.amp and device == "cuda"),
                        ):
                            logits, _ = model(tokens)

                        predicted_norm = get_predicted_row(tokens, mask, logits)

                        num_correct += int(predicted_norm == label.item())
                        num_total += 1

                accuracy = num_correct / max(num_total, 1)

                console.print(f"\n[green]Hellaswag Accuracy: [/green]{accuracy:.4f}\n")
                logger.info(f"Step {step}, Hellaswag accuracy {accuracy}")

            t0 = time()
            loss_accum, lr, norm = train_model(
                step, model, optimizer, scaler, train_loader, config, device
            )
            t1 = time()

            tokens_processed = (
                train_loader.B * train_loader.T * (train_config.total_batch_size // (train_config.batch_size * train_config.context))
            )
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt

            if step % train_config.log_interval == 0:
                console.print(
                    f"Step {step} | Loss: {float(loss_accum):.4f} | lr: {lr:.4e} | "
                    f"Norm: {norm:.4f} | dt: {dt * 1000:.0f}ms | "
                    f"tokens/sec: {tokens_per_sec:.0f}"
                )

            logger.info(f"Step {step}, Loss {float(loss_accum):.4f}")

            step += 1

    except KeyboardInterrupt:
        console.print("[red]Keyboard interrupt detected[/red]")
        console.print("[red]Saving checkpoint...[/red]")
        save_checkpoint(step, model, optimizer, scaler, train_loader)
        raise SystemExit(1)

    final_loss = probe_loss(model, train_loader, config, device)
    console.print(
        f"Probe loss after training: [green]{final_loss:.4f}[/green] "
        f"(was {baseline:.4f}, delta {final_loss - baseline:+.4f})"
    )

    console.print("\n[green]Sample generation:[/green]")
    completion(model, config.generate, device=device)


if __name__ == "__main__":
    main()