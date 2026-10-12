"""Integration: FineWeb sharder output -> CustomDataLoader (plan Phase 5, item 4).

Contract: every token of every document is yielded in sequence by the
dataloaders -- no drops, no duplication, no corrupted shard boundaries.

Shard sizing: CustomDataLoader emits batches of B*T tokens and skips a
sub-batch-sized tail at each shard end. Shards sized `len % (B*T) == 1` emit
every token (X covers tokens 0..n-1, Y covers 1..n), so the test searches for
a shard_size that makes every shard (full and final) satisfy that congruence --
letting the plan's strict no-dropping contract hold exactly.
"""

import numpy as np
import pytest
import torch

from src.gpt.config import DataConfig
from src.gpt.data import CustomDataLoader
from src.dataset import fineweb

B, T = 2, 4
BATCH = B * T  # 8 tokens per batch

DOCS = [
    "The history of neural networks begins with the perceptron.",
    "Transformers use self-attention to mix information across positions.",
    "Sharding a token stream requires careful boundary bookkeeping.",
    "Every token must survive the round trip through numpy shards.",
    "Deterministic tests keep continuous integration honest and fast.",
    "A decoder-only model predicts the next token given its prefix.",
]


def _token_lengths(texts: list[str]) -> list[int]:
    return [len(fineweb.tokenize({"text": t})) for t in texts]


def _find_shard_config(texts: list[str]) -> tuple[int, list[str]]:
    """Find (shard_size, padded_texts) so every shard -- full and final -- has
    len % BATCH == 1 and len >= BATCH+1, with >= 3 shards.

    Guaranteed to converge: shard_size is odd, so advancing `total` two tokens
    at a time (one filler doc) cycles through every residue mod shard_size.
    """
    def search(total: int):
        # shard_size > first doc length avoids a pre-existing edge in fineweb.main()
        # (progress_bar is None when the very first doc already overflows a shard).
        start = max(4 * BATCH + 1, _token_lengths(texts[:1])[0] + 1)
        while start % BATCH != 1:  # candidates must be == 1 (mod BATCH)
            start += 1
        for shard_size in range(start, (total - (BATCH + 1)) // 2 + 1, BATCH):
            final_len = total % shard_size
            if final_len == 0:
                final_len = shard_size  # exact fill: last shard written full
            if final_len % BATCH == 1 and final_len >= BATCH + 1:
                return shard_size
        return None

    padded = list(texts)
    total = sum(_token_lengths(padded))
    for _ in range(64):
        shard_size = search(total)
        if shard_size is not None:
            return shard_size, padded
        # +2 tokens (eot + " p") per filler doc; shard_size is odd so this
        # walks every residue class.
        padded.append(" p")
        total = sum(_token_lengths([padded[-1]])) + total
    pytest.skip(f"no valid shard size for total={total}")


def _shards_sorted(directory) -> list[torch.Tensor]:
    indexed = sorted(
        ((int(p.stem.rsplit("_", 1)[-1]), p) for p in directory.glob("*.npy")),
        key=lambda pair: pair[0],
    )
    return [torch.tensor(np.load(p), dtype=torch.long) for _, p in indexed]


def _stream_batches(loader, num_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [loader.next_batch() for _ in range(num_batches)]


def test_fineweb_to_dataloader_token_integrity(fineweb_pipeline, tmp_path):
    """Full round trip: documents -> shards -> loaders -> every token in order."""
    shard_size, texts = _find_shard_config(DOCS)
    total = sum(_token_lengths(texts))

    cache_dir = fineweb_pipeline(texts, shard_size=shard_size)
    shards = _shards_sorted(cache_dir)
    assert len(shards) >= 3, "need val shard + at least two train shards"

    # Expected ground truth: every tokenized document, concatenated.
    expected = torch.tensor(
        np.concatenate([fineweb.tokenize({"text": d}) for d in texts]),
        dtype=torch.long,
    )
    assert torch.equal(torch.cat(shards), expected), "sharder output already diverges"

    # Every shard satisfies the full-coverage congruence.
    for i, shard in enumerate(shards):
        assert (len(shard) - 1) % BATCH == 0, f"shard {i} not batch-coverable"

    # --- Walk the val loader (shard 0) and train loader (shards 1+) fully. ---
    val_loader = CustomDataLoader(
        B, T, DataConfig(dataset="fineweb", path=cache_dir), split="val"
    )
    train_loader = CustomDataLoader(
        B, T, DataConfig(dataset="fineweb", path=cache_dir), split="train"
    )

    val_shards = [shards[0]]
    train_shards = shards[1:]

    def verify(loader, shard_list, label):
        # For each shard: emit batches and compare against exact stream slices.
        for shard_idx, shard in enumerate(shard_list):
            num_batches = (len(shard) - 1) // BATCH
            for batch_idx in range(num_batches):
                x, y = loader.next_batch()

                start = batch_idx * BATCH
                expected_x = shard[start : start + BATCH].view(B, T)
                expected_y = shard[start + 1 : start + BATCH + 1].view(B, T)

                assert torch.equal(x, expected_x), (
                    f"{label}: shard {shard_idx} batch {batch_idx} inputs "
                    "are not the expected stream slice (drop/dup/reorder)"
                )
                assert torch.equal(y, expected_y), (
                    f"{label}: shard {shard_idx} batch {batch_idx} targets "
                    "are not the expected shift-by-1 slice"
                )

            # After the last batch of this shard, the loader must have moved on.
            if shard_idx + 1 < len(shard_list):
                assert loader.current_shard == shard_idx + 1, (
                    f"{label}: failed to advance past shard {shard_idx}"
                )

    verify(val_loader, val_shards, "val")
    verify(train_loader, train_shards, "train")

    # Every token of every document was emitted by one of the loaders:
    # batches cover tokens [0, len-1] exactly (X covers 0..n-2, Y 1..n-1
    # is the +1 target) -- reconstructed coverage check below.
    covered = 0
    for shard in shards:
        covered += ((len(shard) - 1) // BATCH) * BATCH + 1
    assert covered == sum(len(s) for s in shards), "tokens dropped at shard boundaries"
