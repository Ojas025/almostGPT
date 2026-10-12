"""Integration: checkpoint save -> resume -> next step matches (plan Phase 5, item 3).

Contract: a step executed after loading a checkpoint produces the same loss,
learning rate and weights as the same step in an uninterrupted run.
"""

import pytest
import torch

from src.gpt.checkpoint import load_checkpoint, save_checkpoint
from src.gpt.config import AppConfig, DataConfig, TrainConfig
from src.gpt.gpt import GPT
from src.gpt.train import train_model

B, T = 2, 8


@pytest.fixture
def training_setup(tiny_model_config, mock_loader):
    """Shared config + batch stream for both branches of the comparison."""
    model_config = tiny_model_config(moe=True)

    train_config = TrainConfig(
        batch_size=B,
        context=T,
        total_batch_size=B * T,  # exactly one micro-batch per optimizer step
        learning_rate=1e-3,
        min_lr_ratio=0.1,
        warmup_steps=4,
        max_steps=50,
        weight_decay=0.1,
        grad_clip=1.0,
        device="cpu",
        amp=False,
        compile=False,
        seed=1234,
    )
    app_config = AppConfig(
        model=model_config,
        train=train_config,
        data=DataConfig(dataset="shakespeare", path=None),
    )

    # One fixed batch stream consumed identically by both branches.
    batches = mock_loader(num_batches=16, B=B, T=T)

    def build():
        model = GPT(model_config)
        optimizer = model.configure_optimizers(
            weight_decay=train_config.weight_decay,
            learning_rate=train_config.learning_rate,
            device="cpu",
        )
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)
        loader = type(
            "BranchLoader",
            (),
            {
                "_batches": list(batches._batches),
                "index": 0,
                "reset": lambda self: None,
                "next_batch": lambda self: (
                    self._batches[self.index % len(self._batches)],
                    setattr(self, "index", self.index + 1),
                )[0],
            },
        )()
        return model, optimizer, scaler, loader

    return app_config, build


def test_checkpoint_resume_training(training_setup, tmp_path):
    """Resume at step N reproduces the uninterrupted run's step N exactly."""
    app_config, build = training_setup
    path = str(tmp_path / "ckpt")

    # --- Train steps 0..2, then save at step 3. ---
    model, optimizer, scaler, loader = build()
    for step in range(3):
        train_model(step, model, optimizer, scaler, loader, app_config, "cpu")

    save_checkpoint(3, model, optimizer, scaler, loader, path=path)

    # --- Branch A: the uninterrupted continuation of step 3. ---
    loss_a, lr_a, _ = train_model(3, model, optimizer, scaler, loader, app_config, "cpu")
    weights_a = {n: p.detach().clone() for n, p in model.named_parameters()}

    # --- Branch B: a completely fresh instance resumed from the checkpoint. ---
    model2, optimizer2, scaler2, loader2 = build()
    loaded_step = load_checkpoint(
        f"{path}/latest.pt", model2, optimizer2, scaler2, loader2
    )
    assert loaded_step == 3
    assert loader2.index == 3, "resumed loader must continue where the save happened"

    loss_b, lr_b, _ = train_model(3, model2, optimizer2, scaler2, loader2, app_config, "cpu")

    # --- The two branches must agree. ---
    assert lr_a == pytest.approx(lr_b), "resumed learning rate diverged"
    assert float(loss_a) == pytest.approx(float(loss_b), rel=1e-6, abs=1e-7), (
        "resumed step loss diverged from the uninterrupted run"
    )
    resumed = dict(model2.named_parameters())
    for name, p_a in weights_a.items():
        assert torch.allclose(p_a, resumed[name], atol=1e-6), (
            f"weights diverged after resume: {name}"
        )
