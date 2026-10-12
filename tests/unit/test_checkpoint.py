"""Unit tests for checkpoint persistence (plan Component 13).

Contracts: atomic save (temp file + os.replace), full state serialization
(model/optimizer/scaler/dataloader/4 RNG states), exact RNG restoration.
"""

import random

import numpy as np
import pytest
import torch

from src.gpt.checkpoint import load_checkpoint, save_checkpoint
from src.gpt.gpt import GPT


@pytest.fixture
def make_pieces(tiny_model_config):
    """Factory for independent (model, optimizer, scaler, loader) quartets."""

    def make():
        model = GPT(tiny_model_config())
        optimizer = model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3, device="cpu"
        )
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)
        loader = type("StubLoader", (), {"index": 7})()
        return model, optimizer, scaler, loader

    return make


def test_checkpoint_atomic_save_no_partial(make_pieces, tmp_path):
    """Save lands in latest.pt with no leftover .temp file."""
    model, optimizer, scaler, loader = make_pieces()
    path = str(tmp_path / "ckpt")

    save_checkpoint(3, model, optimizer, scaler, loader, path=path)

    final = tmp_path / "ckpt" / "latest.pt"
    temp = tmp_path / "ckpt" / "latest.pt.temp"
    assert final.exists()
    assert not temp.exists(), "temporary file must be renamed away (atomic save)"

    state = torch.load(final, map_location="cpu", weights_only=False)
    assert state["step"] == 3
    for key in ("model", "optimizer", "scaler", "dataloader_state", "rng_state"):
        assert key in state
    rng = state["rng_state"]
    for key in ("torch", "numpy", "python"):
        assert rng[key] is not None
    assert state["dataloader_state"]["index"] == 7


def test_checkpoint_roundtrip_restoration(make_pieces, tmp_path):
    """Loading into fresh instances restores identical model parameters."""
    model, optimizer, scaler, loader = make_pieces()
    path = str(tmp_path / "ckpt")

    save_checkpoint(5, model, optimizer, scaler, loader, path=path)

    fresh_model, fresh_optimizer, fresh_scaler, fresh_loader = make_pieces()
    step = load_checkpoint(
        f"{path}/latest.pt", fresh_model, fresh_optimizer, fresh_scaler, fresh_loader
    )

    assert step == 5
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), fresh_model.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"parameter {n1} not restored"
    assert fresh_loader.index == 7


def test_checkpoint_rng_state_reproducibility(make_pieces, tmp_path):
    """After load, torch / numpy / python RNGs replay the saved sequence exactly."""
    model, optimizer, scaler, loader = make_pieces()
    path = str(tmp_path / "ckpt")

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    save_checkpoint(1, model, optimizer, scaler, loader, path=path)

    expected = (
        torch.rand(4).clone(),
        np.random.rand(4).copy(),
        [random.random() for _ in range(4)],
    )

    fresh_model, fresh_optimizer, fresh_scaler, fresh_loader = make_pieces()
    load_checkpoint(
        f"{path}/latest.pt", fresh_model, fresh_optimizer, fresh_scaler, fresh_loader
    )

    got = (
        torch.rand(4),
        np.random.rand(4),
        [random.random() for _ in range(4)],
    )

    assert torch.equal(got[0], expected[0]), "torch RNG state not restored"
    assert np.array_equal(got[1], expected[1]), "numpy RNG state not restored"
    assert got[2] == expected[2], "python RNG state not restored"
