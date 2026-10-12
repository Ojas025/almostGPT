"""Unit tests for fineweb.py ingestion (plan Component 16).

Contracts: tokenize prepends 50256 / enforces uint16 bounds; write_file
produces a loadable .npy; the sharding loop spills remainders losslessly and
names shard 0 'val', shard >=1 'train'. The sharding loop is driven through
the fineweb_pipeline fixture (network/process edges faked, real shard logic).
"""

import numpy as np
import pytest

import src.dataset.fineweb as fineweb


def _shards_by_index(directory) -> list:
    """Load shards sorted by their numeric index (names don't sort correctly:
    'edufineweb_train_*' < 'edufineweb_val_*' alphabetically)."""
    indexed = []
    for path in directory.glob("*.npy"):
        index = int(path.stem.rsplit("_", 1)[-1])
        indexed.append((index, path))
    indexed.sort()
    return [np.load(path) for _, path in indexed]

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "AlmostGPT shards FineWeb documents into uint16 numpy arrays.",
    "Attention is all you need, but caching helps too.",
    "Romeo and Juliet is a play written by William Shakespeare.",
    "Large language models learn statistical patterns from text.",
]


def test_fineweb_tokenize_eot_prepended():
    """First token is always 50256 endoftext."""
    tokens = fineweb.tokenize({"text": "hello world"})

    assert int(tokens[0]) == fineweb.eot == 50256


def test_fineweb_tokenize_uint16_bounds():
    """All token IDs fit [0, 65535] and the dtype is np.uint16."""
    tokens = fineweb.tokenize({"text": "Some moderately long sample text. " * 5})

    assert tokens.dtype == np.uint16
    assert (tokens >= 0).all()
    assert (tokens < 2**16).all()


def test_fineweb_write_file(tmp_path):
    """write_file output is readable by np.load and content-identical."""
    path = str(tmp_path / "shard")
    data = np.arange(10, dtype=np.uint16)

    fineweb.write_file(path, data)

    loaded = np.load(f"{path}.npy")  # np.save appends .npy
    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, data)


def test_fineweb_sharding_remainder_spill(fineweb_pipeline):
    """Tokens crossing the shard boundary spill to the next shard losslessly."""
    shard_size = 50
    cache_dir = fineweb_pipeline(DOCS, shard_size=shard_size)

    files = sorted(cache_dir.glob("*.npy"))
    assert len(files) >= 2, "overflow must produce more than one shard"

    shards = _shards_by_index(cache_dir)
    for shard in shards[:-1]:
        assert len(shard) == shard_size, "every non-final shard must be full"

    # Lossless, in-order, no duplication: concatenation == concatenation of
    # every tokenized document.
    expected = np.concatenate([fineweb.tokenize({"text": d}) for d in DOCS])
    actual = np.concatenate(shards)
    assert np.array_equal(actual, expected), (
        "shard spilling lost, duplicated or reordered tokens"
    )


def test_fineweb_split_assignment(fineweb_pipeline):
    """Shard 0 is saved as 'val', shards >= 1 as 'train'."""
    cache_dir = fineweb_pipeline(DOCS, shard_size=50)

    indexed = sorted(
        (int(p.stem.rsplit("_", 1)[-1]), p.name) for p in cache_dir.glob("*.npy")
    )
    assert len(indexed) >= 2

    for index, name in indexed:
        if index == 0:
            assert "val" in name, "shard 0 must use the val split name"
        else:
            assert "train" in name, f"shard {index} must use the train split name"
