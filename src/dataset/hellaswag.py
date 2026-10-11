from pathlib import Path
import requests
from tqdm import tqdm
import json
import torch
import tiktoken

DATA_DIR = Path(__file__).parent / "hellaswag"

hellaswag_urls = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download_file(url: str, filename: str, chunk_size: int = 1024):
    response = requests.get(url, stream=True)
    content_length = int(response.headers.get("content-length", 0))
    
    with open(filename, "wb") as file, tqdm(desc=filename,  total=content_length, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
            
def download(split):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    assert split in { "train", "val", "test" }, "Invalid split"
    url = hellaswag_urls[split]
    
    filename = Path(DATA_DIR) / f"hellaswag_{split}.jsonl"
    
    if not filename.exists():
        print(f"Downloading {url} to {filename}")
        
        try:
            download_file(url, filename)
        except Exception as e:
            print(f"Error downloading file from {url}", str(e))
            
def iterate_examples(split):
    download(split)
    filepath = Path(DATA_DIR) / f"hellaswag_{split}.jsonl"
    
    with open(filepath, "r") as file:
        for line in file:
            example = json.loads(line)
            yield example
   
# TLDR;  
def get_formatted_example(example):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label