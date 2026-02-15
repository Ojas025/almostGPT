"""FineWeb-Edu dataset"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(5e7)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(tokenizer.encode(doc["text"], allowed_special={"<|endoftext|>"}))
    
    arr = np.array(tokens)
    assert (0 <= arr).all() and (arr < 2**16).all(), "token dictionary too large for uint16"
    
    arr = arr.astype(np.uint16)
    return arr

def write_file(filename, tokens):
    np.save(filename, tokens)

def main():
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        streaming=True
    )

    n_processes = max(1, os.cpu_count())
    shard_index = 0
    all_tokens = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    with mp.Pool(n_processes) as pool:
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                
                remainder = shard_size - token_count
                all_tokens[token_count:token_count+remainder] = tokens[:remainder]
                progress_bar.update(remainder)
                write_file(filename, all_tokens)

                shard_index += 1
                all_tokens = np.empty((shard_size,), dtype=np.uint16)
                token_count = len(tokens) - remainder
                all_tokens[:token_count] = tokens[remainder:]
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_file(filename, all_tokens[:token_count])

if __name__ == "__main__":
    mp.freeze_support()  # safe Windows multiprocessing
    main()