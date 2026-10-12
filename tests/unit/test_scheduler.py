"""Unit tests for the cosine LR scheduler (plan Component 12).

Contract: linear warmup -> cosine decay to lr*min_lr_ratio -> floor after
max_steps; the warmup/decay boundary is continuous.
"""

import math

import pytest

from src.gpt.scheduler import get_learning_rate

LR = 1e-3
MIN_RATIO = 0.1
WARMUP = 10
MAX_STEPS = 100


@pytest.fixture
def config(tiny_train_config):
    return tiny_train_config(
        learning_rate=LR,
        min_lr_ratio=MIN_RATIO,
        warmup_steps=WARMUP,
        max_steps=MAX_STEPS,
    )


def test_scheduler_linear_warmup(config):
    """Step 0 -> lr/warmup; step warmup-1 -> lr_max; boundary is continuous."""
    assert get_learning_rate(0, config) == pytest.approx(LR / WARMUP)
    assert get_learning_rate(WARMUP - 1, config) == pytest.approx(LR)

    # Boundary continuity: the first cosine step matches the warmup endpoint.
    at_boundary = get_learning_rate(WARMUP, config)
    assert at_boundary == pytest.approx(LR), (
        f"sharp jump at warmup boundary: {at_boundary} != {LR}"
    )

    # Monotone increase through warmup.
    lrs = [get_learning_rate(s, config) for s in range(WARMUP)]
    assert all(b >= a for a, b in zip(lrs, lrs[1:]))


def test_scheduler_cosine_decay(config):
    """Midpoint of decay yields (lr_max + lr_min) / 2; the curve is monotone."""
    lr_min = LR * MIN_RATIO
    midpoint = (WARMUP + MAX_STEPS) // 2

    assert get_learning_rate(midpoint, config) == pytest.approx((LR + lr_min) / 2)

    lrs = [
        get_learning_rate(s, config) for s in range(WARMUP, MAX_STEPS + 1)
    ]
    assert all(b <= a for a, b in zip(lrs, lrs[1:])), "cosine decay must be monotone"


def test_scheduler_post_max_steps_floor(config):
    """Past max_steps the lr is pinned at lr_max * min_lr_ratio."""
    floor = LR * MIN_RATIO

    assert get_learning_rate(MAX_STEPS, config) == pytest.approx(floor)
    assert get_learning_rate(MAX_STEPS + 1, config) == pytest.approx(floor)
    assert get_learning_rate(MAX_STEPS + 500, config) == pytest.approx(floor)
