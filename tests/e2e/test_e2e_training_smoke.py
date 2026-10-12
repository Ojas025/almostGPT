"""E2E: miniature training cycle (plan Phase 6, item 1).

Data -> optimizer/scaler -> gradient-accumulation steps -> validation eval ->
checkpoint save/reload -> generation. Asserts probe loss drops across the run.

Everything is CPU, seeded and dropout-free, so the run is fully deterministic.
"""

import numpy as np
import pytest
import torch

from src.gpt.checkpoint import load_checkpoint, save_checkpoint
from src.gpt.config import AppConfig, DataConfig, GenerateConfig, TrainConfig
from src.gpt.data import CustomDataLoader
from src.gpt.evaluate import evaluate_model
from src.gpt.gpt import GPT
from src.gpt.train import probe_loss, set_seed, train_model

B, T = 2, 8
MICRO_BATCHES = 2  # total_batch_size = 2 * B * T -> exercises grad accumulation
STEPS = 40
SEED = 42


@pytest.fixture
def smoke_env(tmp_path, tiny_model_config):
    """Tiny model + sharded periodic corpus + full AppConfig."""
    set_seed(SEED)

    # A perfectly periodic corpus: the tiny model can memorise it quickly,
    # making the loss drop deterministic rather than lucky.
    pattern = np.arange(1, 17, dtype=np.uint16)
    corpus = np.tile(pattern, 64)  # 1024 tokens

    directory = tmp_path / "shards"
    directory.mkdir(exist_ok=True)
    np.save(directory / "edufineweb_train_000000.npy", corpus)
    np.save(directory / "edufineweb_train_000001.npy", corpus.copy())
    np.save(directory / "edufineweb_val_000000.npy", corpus[:257])

    model_config = tiny_model_config(moe=False)
    train_config = TrainConfig(
        batch_size=B,
        context=T,
        total_batch_size=MICRO_BATCHES * B * T,  # 2 micro-batches per step
        learning_rate=3e-3,
        min_lr_ratio=0.1,
        warmup_steps=5,
        max_steps=STEPS + 5,
        weight_decay=0.1,
        grad_clip=1.0,
        device="cpu",
        amp=False,
        compile=False,
        seed=SEED,
    )
    app_config = AppConfig(
        model=model_config,
        train=train_config,
        data=DataConfig(dataset="fineweb", path=directory),
        generate=GenerateConfig(
            max_new_tokens=5, num_return_sequences=1, temperature=1.0, top_k=8, top_p=0.95
        ),
    )

    def build():
        model = GPT(model_config)
        optimizer = model.configure_optimizers(
            weight_decay=train_config.weight_decay,
            learning_rate=train_config.learning_rate,
            device="cpu",
        )
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)
        train_loader = CustomDataLoader(
            B, T, app_config.data, split="train"
        )
        val_loader = CustomDataLoader(B, T, app_config.data, split="val")
        return model, optimizer, scaler, train_loader, val_loader

    return app_config, build


def test_e2e_training_smoke(smoke_env, tmp_path):
    """Full cycle: train, evaluate, checkpoint, reload, generate -- loss drops."""
    app_config, build = smoke_env
    model, optimizer, scaler, train_loader, val_loader = build()

    baseline = probe_loss(model, train_loader, app_config, device="cpu")

    for step in range(STEPS):
        if step % 10 == 0:
            val_loss = evaluate_model(
                model, val_loader, step, device="cpu", use_amp=False, max_batches=2
            )
            assert isinstance(val_loss, float) and np.isfinite(val_loss)

        loss, lr, grad_norm = train_model(
            step, model, optimizer, scaler, train_loader, app_config, device="cpu"
        )
        assert np.isfinite(float(loss)), f"non-finite loss at step {step}"
        assert lr > 0 and np.isfinite(grad_norm)

    final = probe_loss(model, train_loader, app_config, device="cpu")
    assert final < baseline, f"probe loss did not drop: {baseline:.4f} -> {final:.4f}"

    # --- Checkpoint save -> reload into a fresh instance. ---
    ckpt_path = str(tmp_path / "ckpt")
    save_checkpoint(STEPS, model, optimizer, scaler, train_loader, path=ckpt_path)

    fresh_model, fresh_opt, fresh_scaler, fresh_train, _ = build()
    restored_step = load_checkpoint(
        f"{ckpt_path}/latest.pt", fresh_model, fresh_opt, fresh_scaler, fresh_train
    )
    assert restored_step == STEPS
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), fresh_model.named_parameters()
    ):
        assert n1 == n2 and torch.allclose(p1, p2), f"reload mismatch on {n1}"

    # The reloaded model produces identical probe loss.
    reloaded_probe = probe_loss(fresh_model, fresh_train, app_config, device="cpu")
    assert reloaded_probe == pytest.approx(final, rel=1e-5, abs=1e-6)

    # --- Sample generation completes with the contracted shape. ---
    prompt = torch.randint(0, app_config.model.vocab_size, (1, 4))
    out = fresh_model.generate(prompt, app_config.generate, device="cpu")
    assert out.shape == (
        app_config.generate.num_return_sequences,
        4 + app_config.generate.max_new_tokens,
    )
    assert (out < app_config.model.vocab_size).all()
