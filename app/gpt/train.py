import logging
from dataclasses import dataclass
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from checkpoint import load_checkpoint, save_checkpoint
from evaluate import evaluate_model
from gpt import GPT
from inference import get_predicted_row
from kv_cache import KVCache
from logger import configure_logging
from rich.console import Console
from rich.panel import Panel
from scheduler import get_learning_rate, max_steps
from torch.utils.data import DataLoader

from data import CustomDataLoader, HellaSwagDataset

logger = logging.getLogger(__name__)

"""
Changes in GPT-2 from the original transformer paper:
    - Layer Normalization moved to the input of each sub-block
    - Additional LayerNorm after final self-attention block
"""

"""
Stuff implemented:
    - Use of GeLU instead of ReLU to deal with dead activations, as GeLU always contributes a local gradient
    - Use SwiGLU over GeLU, uses Swish activation (x.sigmoid(x)) with Gated Linear Units
    - Includes ugly number optimization
    - Gradient clipping to a norm -> 1.0
    - Learning rate schedule using cosine decay
    - Add parameter decay to introduce regularization
    - Implement gradient accumulation to increase batch_size
    - Implement Learned Positional Embedding
    - Implement Sinusoidal Positional Embedding 
    - Added Gradient Scaling to avoid gradient underflow due to Automatic Mixed Precision (float16)
    - Added top_k and top_p filtering
    - Add RMS Norm over LayerNorm to reduce computations
    - Added checkpointing to resume training state
    - Implement Rotary Positional Embeddings (RoPE)
    - Implement Grouped Multi-Query Attention
    - Implement Sliding Window Attention
    - Implement Rolling KV Cache
    
TODO:
    - Add Mixture-of-Experts
    - LoRA Fine Tuning
    - Custom Flash Attention? (Cuda)
"""


configure_logging()

logger = logging.getLogger(__name__)

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = Path("../data/shakespeare.txt").resolve()

panel = Panel(
    "An implementation of GPT-like decoder-only transformer", title="AlmostGPT"
)
console.print(panel)
console.print(f"Using device: [yellow]{DEVICE}[/yellow]")


def load_data(path: str):
    with open(path, "r") as file:
        text = file.read()

    return text


@dataclass
class GPTConfig:
    """GPT architecture configurations"""

    context_length: int = 1024  # max context length
    vocab_size: int = 50257  # number of tokens: 50000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension

    # flash_attention_kernel = load(name="flash_attention", sources=["flash_attention_kernel.cu"])

    # def flash_attention(q,k,v,scale):
    #     return flash_attention_kernel(q,k,v,scale)

    # class FlashAttention(torch.autograd.Function):
    #     """Flash attention implementation"""
    #     # TODO: implement CUDA kernel

    #     @staticmethod
    #     def forward(ctx, q, k, v, scaling):
    #         ctx.save_for_backward(q,k,v,scaling)
    #         output = flash_attention(q,k,v,scaling)
    #         return output

    #     @staticmethod
    #     def backward(ctx, grad_outputs):
    #         q, k, v, scale = ctx.saved_tensors

    #         grad_q, grad_k, grad_v = flash_attention_kernel(grad_outputs, q, k, v, scale)

    #         return grad_q,grad_k,grad_v,None

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_output_tokens: int = 200,
        num_return_sequences: int = 3,
        temperature: float = 1.0,
        k: int = 50,
        p: float = 0.95,
    ):
        self.eval()

        tokens = tokens.unsqueeze(0)  # [1, T]
        tokens = tokens.repeat(num_return_sequences, 1)  # [num_return_sequences, T]
        tokens = tokens.to(DEVICE)

        cache = KVCache(
            n_layers=model.config.n_layer,
            n_head=model.config.n_head,
            d_head=model.config.n_embed // model.config.n_head,
            window_size=model.config.context_length,
            device=DEVICE,
        )

        pos = 0

        while tokens.size(1) < max_output_tokens:
            tokens_curr = tokens if cache.is_empty() else tokens[:, -1:]
            logits, _ = self(tokens_curr, use_cache=True, cache=cache, pos=pos)
            logits = logits[:, -1, :]  # logits at the last position

            logits = logits / max(temperature, 1e-6)
            logits = self.top_k(logits, k)
            logits = self.top_p(logits, p)

            probabilities = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probabilities, 1)

            # append new col to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)
            pos += tokens_curr.size(1)

        return tokens


# --------------------------------------------------------------------


def train_model(step: int):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(gradient_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = model(x, y)

        # scale gradient using MEAN to account for gradient accumulation
        loss /= gradient_accumulation_steps
        loss_accum += loss.detach()

        scaler.scale(loss).backward()

    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # determine learning rate for this iteration
    lr = get_learning_rate(step)

    # update learning rate in params
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # without scaling
    # optimizer.step()

    # with scaling
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()  # wait for GPU to finish

    return loss_accum, lr, norm


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")

    total_batch_size = 524288  # ~0.5M in number of tokens
    B = 4
    T = 1024

    assert total_batch_size % (B * T) == 0, "total_batch_size is divisible by B*T"

    gradient_accumulation_steps = total_batch_size // (B * T)

    console.print(f"\nTotal desired batch size: {total_batch_size}")
    console.print(f"Gradient accumulation steps: {gradient_accumulation_steps}\n")

    # init model
    model = GPT(GPTConfig(vocab_size=50304))  # remove ugly number 50257 -> 50304
    model.eval()
    model.to(DEVICE)

    if hasattr(torch, "compile"):
        model.compile()

    console.print("Initialized model")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=[0.9, 0.95], eps=1e-8)
    train_loader = CustomDataLoader(B, T, split="train", dataset="shakespeare")
    val_loader = CustomDataLoader(B, T, split="val", dataset="shakespeare")

    # Hellaswag
    hellaswag_dataset = HellaSwagDataset(split="val")
    hellaswag_dataloader = DataLoader(hellaswag_dataset, batch_size=1, shuffle=True)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=DEVICE
    )
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))

    resume_checkpoint_path = None

    if resume_checkpoint_path is not None:
        step = load_checkpoint(
            resume_checkpoint_path, model, optimizer, scaler, train_loader
        )
    else:
        step = 0

    try:
        while step < max_steps:
            is_last_step = step == (max_steps - 1)

            # Evaluate
            if step % 250 == 0:
                evaluate_model()

            # Save checkpoint
            if step > 0 and (step % 5000 == 0 or is_last_step):
                if is_last_step:
                    console.print("\n[green]Saving final model...[/green]")

                save_checkpoint(step, model, optimizer, scaler)

            # Hellaswag
            if step > 0 and (step % 250 == 0 or is_last_step):
                num_correct_norm = 0
                num_total = 0

                with torch.no_grad():
                    for tokens, mask, label in hellaswag_dataloader:
                        tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)

                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits, loss = model(tokens)

                        predicted_norm = get_predicted_row(tokens, mask, logits)

                        num_correct_norm += int(predicted_norm == label.item())
                        num_total += 1

                accuracy = num_correct_norm / num_total

                console.print(f"\n[green]Hellaswag Accuracy: [/green]{accuracy:.4f}\n")
                logger.info(f"Step {step}, Hellaswag accuracy {accuracy}")

            t0 = time()
            loss_accum, lr, norm = train_model(step)
            t1 = time()

            tokens_processed = (
                train_loader.B * train_loader.T * gradient_accumulation_steps
            )
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt

            print(
                f"Step {step} | Loss: {loss_accum.item():.4f} | lr: {lr:.4e} | Norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}"
            )
            logger.info(f"Step {step}, Loss: {loss_accum.item():.4f}")

            step += 1
    except KeyboardInterrupt:
        console.print("[red]Keyboard interrupt detected[/red]")
        console.print("[red]Saving checkpoint...[/red]")

        save_checkpoint(step, model, optimizer, scaler)
        exit(1)
