# AlmostGPT

This repository contains a from-scratch implementation of a GPT-style Transformer language model with a strong emphasis on architectural correctness, numerical stability, and production-oriented training mechanics. The project explores deviations from the original Transformer paper, incorporates modern optimization techniques, and serves as a foundation for further research into efficient attention mechanisms and scalable training.

The implementation is intentionally explicit and modular to support experimentation, benchmarking, and incremental system-level improvements.

---

## Architectural Notes

This implementation follows the GPT-2 architectural conventions with the following deviations from the original Transformer paper:

- Layer Normalization is applied at the input of each sub-block (Pre-LN formulation).
- An additional Layer Normalization is applied after the final self-attention block.

These changes improve optimization stability for deeper networks and enable higher learning rates without divergence.

---

## Implemented Features

### Model Architecture

- **Activation Functions**
  - Gaussian Error Linear Units (GeLU) to avoid dead activations and preserve local gradient flow.
  - SwiGLU (Swish-Gated Linear Units) for improved expressiveness and gradient conditioning.

- **Normalization**
  - RMSNorm as a drop-in alternative to LayerNorm for reduced computational overhead.

- **Positional Encoding**
  - Learned positional embeddings.
  - Sinusoidal positional embeddings.

- **Attention Sampling**
  - Top-k filtering.
  - Top-p (nucleus) sampling.

---

### Training and Optimization

- Gradient clipping with maximum norm = 1.0.
- Gradient accumulation to simulate larger effective batch sizes.
- Cosine learning rate decay schedule.
- Parameter weight decay for regularization.
- Automatic Mixed Precision (AMP) with gradient scaling to prevent underflow in float16 training.
- Checkpointing for fault tolerance and experiment reproducibility.
- Integer arithmetic optimization where applicable.

---

## Current Status

The model supports end-to-end training and autoregressive text generation with configurable architecture and sampling strategies. The training pipeline includes stability-focused features suitable for long-running experiments and constrained hardware environments.

---

## Roadmap

Planned improvements and research extensions:

- Rotary Positional Embeddings (RoPE).
- Grouped Multi-Query Attention (GQA / MQA).
- Sliding Window Attention for long-context efficiency.
- Rolling KV cache for streaming inference.
- Custom Flash Attention kernels (CUDA).
- Custom Byte Pair Encoding (BPE) tokenizer.

---

## Design Goals

- Maintain architectural clarity over abstraction-heavy frameworks.
- Enable controlled experimentation with attention mechanisms and normalization strategies.
- Emphasize numerical stability and reproducibility.
- Keep the system extensible for research-driven iteration.

---

Example:

```bash
python src/gpt2/train.py
```

---

## Notes

This repository is intended for educational, research, and experimental purposes. It is not optimized for production inference at scale. Performance-critical components such as attention kernels and tokenizer pipelines are expected to evolve.
