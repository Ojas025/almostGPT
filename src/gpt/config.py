"""Typed configuration for almostGPT.

Values come from `config.yml` (`defaults:` plus optional `profiles:` overrides) and
are declared as pydantic models, so field types are checked at construction.

Schemas are nested: `GPT.__init__` takes one `ModelConfig`, which carries the
attention, MoE and KV-cache sub-configs. Each is still a distinct type.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# Dataset locations, declared once so the tokenizer that writes shards and the loader
# that reads them cannot drift apart. They previously disagreed -- shards were written
# to src/dataset/edu_fineweb10B/ while the loader looked in src/gpt/edu_fineweb10B/ --
# which meant the FineWeb path could never find its own output.
SHARD_DIR = Path(__file__).resolve().parent.parent / "dataset" / "edu_fineweb10B"
SHAKESPEARE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "shakespeare.txt"
)


class AttentionConfig(BaseModel):
    """Grouped-query / sliding-window attention settings."""

    n_group: int = Field(default=6, gt=0, description="Number of KV groups (<= n_head)")
    is_flash_attention: bool = True
    sliding_window: int | None = Field(default=None, gt=0)
    attention_sink: int | None = Field(default=None, ge=0)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)


class MoEConfig(BaseModel):
    """Mixture-of-Experts routing settings."""

    enabled: bool = True
    num_experts: int = Field(default=32, gt=0)
    d_hidden_multiplier: int = Field(default=4, gt=0)
    top_k: int = Field(default=1, gt=0)
    capacity_factor: float = Field(default=1.0, gt=0.0)
    min_capacity: int = Field(default=4, gt=0)
    z_loss_coef: float = Field(default=1e-3, ge=0.0)
    aux_loss_coef: float = Field(default=1e-2, ge=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class KVCacheConfig(BaseModel):
    """Rolling KV cache settings, used during generation."""

    enabled: bool = True
    window: int | None = Field(default=None, gt=0, description="Defaults to context_length")
    sink: int = Field(default=4, ge=0, description="Attention-sink tokens kept verbatim")


class ModelConfig(BaseModel):
    """Architecture of the transformer. This is what `GPT.__init__` accepts."""

    context_length: int = Field(default=1024, gt=0)
    vocab_size: int = Field(default=50304, gt=0)  # 50257 real tokens padded to a multiple of 64
    n_layer: int = Field(default=12, gt=0)
    n_head: int = Field(default=12, gt=0)
    n_embed: int = Field(default=768, gt=0)
    norm_type: Literal["rms", "layer"] = "rms"
    tie_embeddings: bool = True
    attention: AttentionConfig = AttentionConfig()
    moe: MoEConfig = MoEConfig()
    kv_cache: KVCacheConfig = KVCacheConfig()

    @model_validator(mode="after")
    def _check_head_geometry(self) -> "ModelConfig":
        # Cross-field consistency. Without these, a bad head count only surfaces deep
        # inside MaskedSelfAttention.__init__, long after the config was loaded.
        if self.n_embed % self.n_head != 0:
            raise ValueError(
                f"n_embed ({self.n_embed}) must be divisible by n_head ({self.n_head})"
            )

        if self.n_head % self.attention.n_group != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by "
                f"attention.n_group ({self.attention.n_group})"
            )

        if self.attention.sliding_window is not None and self.attention.attention_sink is None:
            raise ValueError("attention_sink must be set when sliding_window is enabled")

        return self


class DataConfig(BaseModel):
    """Which dataset to train on, and where its shards live."""

    dataset: Literal["shakespeare", "fineweb"] = "shakespeare"
    path: Path | None = Field(default=None, description="Dataset file or shard directory")
    tokenizer: Literal["gpt2"] = "gpt2"


class TrainConfig(BaseModel):
    """Optimization loop, schedule and runtime switches."""

    batch_size: int = Field(default=4, gt=0, description="Micro-batch size (B)")
    context: int = Field(default=1024, gt=0, description="Sequence length (T)")
    total_batch_size: int = Field(default=524288, gt=0)
    learning_rate: float = Field(default=6e-4, gt=0.0)
    min_lr_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
    warmup_steps: int = Field(default=100, ge=0)
    max_steps: int = Field(default=19073, gt=0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    amp: bool = True
    compile: bool = True
    eval_interval: int = Field(default=250, gt=0)
    checkpoint_interval: int = Field(default=5000, gt=0)
    log_interval: int = Field(default=1, gt=0)
    seed: int = 20260911
    run_hellaswag: bool = False

    @model_validator(mode="after")
    def _check_batch_geometry(self) -> "TrainConfig":
        micro_batch = self.batch_size * self.context

        if self.total_batch_size % micro_batch != 0:
            raise ValueError(
                f"total_batch_size ({self.total_batch_size}) must be divisible by "
                f"batch_size * context ({micro_batch})"
            )

        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be less than "
                f"max_steps ({self.max_steps})"
            )

        return self


class GenerateConfig(BaseModel):
    """Sampling parameters for `GPT.generate`."""

    max_new_tokens: int = Field(default=200, gt=0)
    num_return_sequences: int = Field(default=3, gt=0)
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=50, gt=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


class AppConfig(BaseModel):
    """Root config. Constructed from `config.yml` via `from_yaml`."""

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    generate: GenerateConfig = GenerateConfig()
    profile: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, profile: str | None = None) -> "AppConfig":
        """Load `config.yml`, merging `profiles.<profile>` over `defaults`.

        A profile only states what it changes; anything it omits keeps the default.
        """
        if path is None:
            path = Path(__file__).resolve().parent.parent / "config.yml"

        path = Path(path)

        with open(path, "r") as handle:
            raw = yaml.safe_load(handle) or {}

        if profile is None:
            merged = raw["defaults"]
        else:
            if profile not in raw.get("profiles", {}):
                raise KeyError(f"unknown profile: {profile!r}")

            # Merge in place: nested mappings merge key-by-key, scalars and lists replace.
            merged = dict(raw["defaults"])
            for section, values in raw["profiles"][profile].items():
                if isinstance(merged.get(section), dict) and isinstance(values, dict):
                    merged[section] = {**merged[section], **values}
                else:
                    merged[section] = values

        merged["profile"] = profile

        return cls(**merged)