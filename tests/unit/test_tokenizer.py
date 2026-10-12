"""Unit tests for the GPT-2 BPE Tokenizer wrapper (plan Component 11)."""

import pytest
import torch

from src.gpt.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()


def test_tokenizer_encode_decode_roundtrip(tokenizer):
    """decode(encode(text)) == text losslessly for standard text."""
    text = "Hello, world! AlmostGPT round-trips text 123 -- with punctuation."

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_tensor_device(tokenizer):
    """encode(..., device='cpu') returns a torch.LongTensor on the target device."""
    tokens = tokenizer.encode("device check", device="cpu")

    assert isinstance(tokens, torch.Tensor)
    assert tokens.device.type == "cpu"
    assert tokens.dtype == torch.long


def test_tokenizer_decode_tensor_and_list(tokenizer):
    """decode accepts both tensors and raw python lists, identically."""
    text = "tensor or list"
    tokens = tokenizer.encode(text)

    from_tensor = tokenizer.decode(tokens)
    from_list = tokenizer.decode(tokens.tolist())

    assert from_tensor == text
    assert from_list == text
