# PLAN — almostGPT Fix & Complete

> Merged from:
> - Part I: /home/ojas/docs/superpowers/plans/2026-09-07-almostgpt-minimal-training.md (2026-09-07)
> - Part II: /home/ojas/projects/project-guides/almostGPT/guide.md (audit 2026-07-30)

---

# Part I — Minimal Runnable Training Plan (fix to first smoke-train + generate)

# AlmostGPT Minimal Runnable Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get almostGPT to import cleanly, run a tiny CPU smoke-train, and generate text with the minimal fix set.

**Architecture:** Keep flat `app/gpt` layout (no pyproject/CLI refactor yet); break the `train.py` import cycle by pointing model files at existing `config.py`/`console.py`, gate AMP by device, and add a tiny-CPU override for smoke runs.

**Tech Stack:** Python 3.11, PyTorch CPU (>=2.2), tiktoken, numpy, tqdm, rich, datasets, requests

**Spec:** `/home/ojas/projects/project-guides/almostGPT/guide.md` (audit 2026-07-30, sections 3 P0 + 10 First pass)

## Global Constraints

- Do not add features (no MoE default change, no RoPE rewrite, no HellaSwag fix, no KV benchmark) — runnable first.
- Keep `app/gpt` flat layout; run scripts from `/home/ojas/projects/almostGPT/app/gpt`.
- CPU must work with no CUDA (`DEVICE=cpu` path, AMP disabled on CPU).
- Every change must keep `PYTHONDONTWRITEBYTECODE=1 python -c "import train"` working once torch is installed.
- Shakespeare path is the default smoke dataset (`data/shakespeare.txt`).

---

### Task 1: Env + requirements fix

**Files:**
- Modify: `/home/ojas/projects/almostGPT/requirements.txt`
- Test: shell smoke (`python -c "import torch, tiktoken"`)

**Interfaces:**
- Consumes: nothing (fresh venv)
- Produces: working venv with torch CPU + correct deps for Tasks 2-5

- [x] **Step 1: Create venv and install CPU torch**

```bash
cd /home/ojas/projects/almostGPT
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . 2>/dev/null || echo "no pyproject yet, skipping"
```

- [x] **Step 2: Fix requirements.txt (remove stdlib, add missing)**

Replace full contents of `/home/ojas/projects/almostGPT/requirements.txt` with:

```text
torch>=2.2
numpy>=1.26
tiktoken>=0.7
rich>=13.7
tqdm>=4.66
datasets>=2.19
requests>=2.31
```

Why: `logging` is stdlib (pip install fails/confuses), `requests` is used by `app/hellaswag.py` but missing, `tokenizers` is wrong dep (code should use `tiktoken`), unpinned dev deps (`ipykernel`) break repro.

- [x] **Step 3: Install fixed requirements and verify**

```bash
. .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch, tiktoken, numpy, tqdm, rich, datasets, requests; print('deps-ok', torch.__version__)"
```

Expected: PASS with `deps-ok 2.x.x`, no `ModuleNotFoundError`.

- [x] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "fix: correct requirements for CPU smoke train"
```

---

### Task 2: Fix hard import blockers (tokenizer + data)

**Files:**
- Modify: `/home/ojas/projects/almostGPT/app/gpt/tokenizer.py:1-10`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/data.py:1-15`
- Test: `PYTHONDONTWRITEBYTECODE=1 python -c "from tokenizer import Tokenizer; from data import CustomDataLoader"`

**Interfaces:**
- Consumes: Task 1 venv (tiktoken installed)
- Produces: `Tokenizer` and `CustomDataLoader` importable without `train.py` cycle

- [x] **Step 1: Fix tokenizer.py import**

In `/home/ojas/projects/almostGPT/app/gpt/tokenizer.py`, replace:

```python
from tokenizers import tiktoken
from train import DEVICE
```

with:

```python
import tiktoken
```

And change `encode` to take device explicitly:

```python
class Tokenizer:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, data, device="cpu"):
        tokens = self.encoder.encode(data)
        return torch.tensor(tokens, device=device)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.encoder.decode(ids)
```

- [x] **Step 2: Fix data.py imports**

In `/home/ojas/projects/almostGPT/app/gpt/data.py`, replace:

```python
from train import DATASET_PATH, Dataset, console, load_data
```

with:

```python
from pathlib import Path
from torch.utils.data import Dataset
from console import console
```

Add local path constant (do not import from train):

```python
DATASET_PATH = Path("../../data/shakespeare.txt").resolve()
```

Keep `from tokenizer import Tokenizer` and `from hellaswag import get_formatted_example, iterate_examples` as-is for now.

- [x] **Step 3: Verify imports**

```bash
cd /home/ojas/projects/almostGPT/app/gpt
../../.venv/bin/python -c "from tokenizer import Tokenizer; t=Tokenizer(); print(t.decode(t.encode('hello', device='cpu')))"
```

Expected: PASS, prints `hello`.

- [x] **Step 4: Commit**

```bash
git add app/gpt/tokenizer.py app/gpt/data.py
git commit -m "fix: break tokenizer/data import cycle with train"
```

---

### Task 3: Break remaining train.py cycle (device/config/console)

**Files:**
- Modify: `/home/ojas/projects/almostGPT/app/gpt/gpt.py:8`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/attention.py:7`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/transformer.py:7`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/feed_forward.py:5`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/embeddings.py:5`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/evaluate.py:5`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/checkpoint.py:7`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/inference.py:5`
- Test: `PYTHONDONTWRITEBYTECODE=1 python -c "import train; print('train-import-ok')"`

**Interfaces:**
- Consumes: Task 2 (`console.py`, `config.py` exist and hold `console`, `GPTConfig`)
- Produces: all model modules importable standalone; `train.py` is entrypoint only

- [x] **Step 1: Replace config imports (4 files)**

In each file, replace `from train import ...GPTConfig` with `from config import GPTConfig`:

`gpt.py`:
```python
from config import GPTConfig
from console import console
import torch
```

`attention.py`:
```python
from config import GPTConfig
```

`transformer.py`:
```python
from config import GPTConfig
```

`feed_forward.py`:
```python
from config import GPTConfig
```

- [x] **Step 2: Replace device/console imports (4 files)**

`embeddings.py` — replace `from train import DEVICE` with device param (default cpu, caller passes real device):
```python
# delete: from train import DEVICE
# build_sliding_window_mask and RoPE must take device from input tensor, e.g.:
# i = torch.arange(q_len, device=x.device)[:, None]
```
Minimal edit: delete the `from train import DEVICE` line and use `x.device` / passed tensor device instead of global `DEVICE`.

`evaluate.py` — replace:
```python
from train import DEVICE
```
with:
```python
from console import console
```
and change signature to `def evaluate_model(model, val_loader, step, device="cpu")` replacing all `DEVICE` uses with `device`.

`checkpoint.py` — replace:
```python
from train import DEVICE, console
```
with:
```python
from console import console
```
(`DEVICE` is unused in checkpoint save/load except logging; drop it.)

`tokenizer.py` — already fixed in Task 2 (no DEVICE import).

- [x] **Step 3: Fix inference.py model import**

In `/home/ojas/projects/almostGPT/app/gpt/inference.py`, replace:
```python
from train import console, model
```
with:
```python
from console import console
```
Delete any use of global `model` at import time; functions must take `model` as arg:
```python
def get_predicted_row(tokens, mask, logits):
    ...
```

- [x] **Step 4: Verify full import chain**

```bash
cd /home/ojas/projects/almostGPT/app/gpt
../../.venv/bin/python -c "import train; print('train-import-ok')"
../../.venv/bin/python -c "from gpt import GPT; from config import GPTConfig; m=GPT(GPTConfig(n_layer=1,n_head=2,n_embed=16,vocab_size=32,context_length=8)); print('model-ok')"
```

Expected: both PASS. If `torch` missing error appears, redo Task 1.

- [ ] **Step 5: Commit**

```bash
git add app/gpt/gpt.py app/gpt/attention.py app/gpt/transformer.py app/gpt/feed_forward.py app/gpt/embeddings.py app/gpt/evaluate.py app/gpt/checkpoint.py app/gpt/inference.py
git commit -m "refactor: move config/console/device out of train"
```

---

### Task 4: Fix CPU runtime crashes (4 one-liners)

**Files:**
- Modify: `/home/ojas/projects/almostGPT/app/gpt/attention.py:forward`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/train.py:111,162,212`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/evaluate.py:23`
- Modify: `/home/ojas/projects/almostGPT/app/gpt/checkpoint.py:27`
- Test: CPU forward + checkpoint save without AttributeError / autocast error

**Interfaces:**
- Consumes: Task 3 importable model
- Produces: CPU-safe forward, train step, checkpoint round-trip

- [x] **Step 1: Init sliding_window_mask in attention.py**

At top of `MaskedSelfAttention.forward()`, before `if use_cache:` add:

```python
sliding_window_mask = None
```

Full context:
```python
B, T, C = x.shape
sliding_window_mask = None
```

Why: `use_cache=False` + `sliding_window=None` + flash path reads `sliding_window_mask is None` before assignment → `UnboundLocalError`.

- [x] **Step 2: Gate autocast by device (train.py + evaluate.py)**

In `train.py` `train_model()`, replace:
```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
```
with:
```python
use_amp = DEVICE == "cuda"
with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
```

Same in `train.py` HellaSwag loop (~line 212) and `evaluate.py:23`:
```python
with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
```

- [x] **Step 3: Fix checkpoint current_shard + model.compile**

In `checkpoint.py:27`, replace:
```python
"current_shard": train_loader.current_shard,
```
with:
```python
"current_shard": getattr(train_loader, "current_shard", None),
```

In `train.py:162`, replace:
```python
model.compile()
```
with:
```python
model = torch.compile(model)
```
Gate behind flag for CPU smoke (skip compile on CPU — slow/not needed):
```python
if DEVICE == "cuda" and hasattr(torch, "compile"):
    model = torch.compile(model)
```

- [x] **Step 4: Verify CPU forward + checkpoint**

```bash
cd /home/ojas/projects/almostGPT/app/gpt
../../.venv/bin/python -c "
from gpt import GPT
from config import GPTConfig
import torch
m = GPT(GPTConfig(n_layer=1,n_head=2,n_embed=16,vocab_size=32,context_length=8))
m.eval()
x = torch.randint(0, 32, (2, 8))
with torch.no_grad():
    logits, loss = m(x, x)
print('forward-ok', tuple(logits.shape), float(loss))
"
```

Expected: PASS `forward-ok (2, 8, 32) <loss-float>`, no `UnboundLocalError`, no autocast error.

- [ ] **Step 5: Commit**

```bash
git add app/gpt/attention.py app/gpt/train.py app/gpt/evaluate.py app/gpt/checkpoint.py
git commit -m "fix: CPU-safe autocast, mask init, checkpoint shard, compile"
```

---

### Task 5: Tiny CPU smoke train + generate + README

**Files:**
- Modify: `/home/ojas/projects/almostGPT/app/gpt/train.py:__main__` (tiny override)
- Modify: `/home/ojas/projects/almostGPT/README.md` (quickstart)
- Modify: `/home/ojas/projects/almostGPT/.gitignore` (ignore logs/runs)
- Test: 5-step CPU train completes, sample written

**Interfaces:**
- Consumes: Tasks 1-4 (importable, CPU-safe)
- Produces: proven `loss decreases` smoke run + correct README

- [ ] **Step 1: Add tiny smoke override in train.py**

After `if __name__ == "__main__":` add env-gated tiny config (keeps default GPU values intact):

```python
import os
if os.environ.get("ALMOSTGPT_SMOKE") == "1":
    total_batch_size = 2 * 32
    B, T = 2, 32
    gradient_accumulation_steps = 1
    max_steps_override = 5
else:
    total_batch_size = 524288
    B, T = 4, 1024
    gradient_accumulation_steps = total_batch_size // (B * T)
    max_steps_override = None
```

And after model init, use `GPTConfig(n_layer=2, n_head=2, n_embed=64, vocab_size=50304, context_length=32)` when `ALMOSTGPT_SMOKE==1`. Skip HellaSwag loop when smoke (dataset download too heavy).

- [ ] **Step 2: Run 5-step CPU smoke**

```bash
cd /home/ojas/projects/almostGPT/app/gpt
ALMOSTGPT_SMOKE=1 ../../.venv/bin/python train.py
```

Expected: completes <2 min on CPU, prints train loss per step, writes checkpoint to `checkpoints/latest.pt`, no traceback. If `data/shakespeare.txt` missing, run `ls ../../data/` first — file must exist.

- [ ] **Step 3: Generate one sample (proves weights load)**

```bash
cd /home/ojas/projects/almostGPT/app/gpt
../../.venv/bin/python -c "
import torch
from gpt import GPT
from config import GPTConfig
from checkpoint import load_checkpoint
"
```

Minimal generate (add to `inference.py` only if trivial; else manual):
```bash
../../.venv/bin/python inference.py --help 2>&1 || echo "no CLI yet, use python -c generate"
```

Acceptance: at minimum, checkpoint file exists and `model.load_state_dict` succeeds.

- [ ] **Step 4: Fix README quickstart + gitignore**

In `README.md`, replace:
```bash
python src/gpt2/train.py
```
with:
```md
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd app/gpt
ALMOSTGPT_SMOKE=1 python train.py
```
```

Append to `.gitignore`:
```gitignore
app/logs/
runs/
checkpoints/
*.pt
__pycache__/
```

Untrack log if present:
```bash
git rm --cached app/logs/train.log 2>/dev/null || true
```

- [ ] **Step 5: Commit**

```bash
git add app/gpt/train.py README.md .gitignore
git commit -m "feat: tiny CPU smoke train and fixed quickstart"
```

---

## Self-Review

1. **Spec coverage:** P0 README path → Task 5; requirements stdlib/missing → Task 1; train-cycle imports → Tasks 2-3; tokenizer tiktoken → Task 2; Dataset import → Task 2; inference model import → Task 3; sliding mask → Task 4; CPU autocast → Task 4; checkpoint shard → Task 4; model.compile → Task 4. Deliberately out of scope (not minimal): HellaSwag indexing, top-k guard, MoE default, RoPE flatten, tests dir, pyproject/CLI, ablations, model card — per Global Constraints.
2. **Placeholder scan:** no TBD/TODO/appropriate-handling; every step has exact file:line, code block, and shell command.
3. **Type consistency:** `device` is `str` ("cpu"/"cuda") threaded through `evaluate_model(model, val_loader, step, device)` and `Tokenizer.encode(data, device)`; `GPTConfig` always from `config`; `console` always from `console`; matches Tasks 2-3 definitions.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-07-almostgpt-minimal-training.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**


---

# Part II — Audit, Logging, Experiment Tracking, Observability, CI & Production Hardening

> Source: project-guides/almostGPT/guide.md — full audit + production guidance. Part I above is the executable minimal fix set; this part is the proper hardening track.

# Guide: almostGPT

Repository reviewed: `github.com/Ojas025/almostGPT`  
Local audit path: `/home/ojas/projects/almostGPT`  
Audit date: 2026-07-30

Note: this repository already existed locally and had uncommitted changes at audit time. I did not edit it. The guide below is based on the current local working tree, not only the last committed GitHub state.

## 1. What this repo can become

This project has a good portfolio angle: a from-scratch GPT-style language model lab in PyTorch. The valuable part is not "I trained a chatbot." The valuable part is:

- decoder-only Transformer implementation
- causal self-attention
- grouped-query attention style key/value sharing
- rotary positional embeddings
- RMSNorm / LayerNorm variants
- SwiGLU / GeLU feed-forward variants
- top-k and top-p sampling
- rolling KV cache concept
- gradient accumulation
- cosine LR schedule
- AMP/GradScaler
- checkpointing with RNG state
- HellaSwag-style multiple-choice evaluation path
- FineWeb-Edu preprocessing script
- Mixture-of-Experts experiment
- LoRA layer experiment

The repo can be interview-strong if it becomes a clean "LLM systems from scratch" project with:

- one command that runs a tiny smoke train on CPU
- one command that generates text
- unit tests proving tensor shapes, masking, KV cache, sampling, checkpoint resume, and HellaSwag scoring
- benchmark/ablation tables
- docs that explain design tradeoffs against GPT-2, RoPE, GQA, Switch/MoE, and PyTorch SDPA

Right now, the biggest issue is that the implementation is not cleanly runnable from a fresh environment. Fix that before adding more model features.

## 2. Verified current state

### Repository state

The local repo has uncommitted changes:

- modified: `README.md`
- modified: `app/gpt/attention.py`
- modified: `app/gpt/checkpoint.py`
- modified: `app/gpt/evaluate.py`
- modified: `app/gpt/feed_forward.py`
- modified: `app/gpt/gpt.py`
- modified: `app/gpt/kv_cache.py`
- modified: `app/gpt/mixture_of_experts.py`
- modified: `app/gpt/normalization.py`
- modified: `app/gpt/train.py`
- modified: `requirements.txt`
- untracked: `app/gpt/config.py`
- untracked: `app/gpt/console.py`

Remote:

- `https://github.com/Ojas025/almostGPT.git`

Main files:

- `app/gpt/gpt.py`: main GPT module.
- `app/gpt/attention.py`: masked self-attention, RoPE, GQA-style K/V grouping, SDPA.
- `app/gpt/transformer.py`: transformer block.
- `app/gpt/mixture_of_experts.py`: top-1 MoE experiment.
- `app/gpt/kv_cache.py`: rolling KV cache.
- `app/gpt/train.py`: training script.
- `app/gpt/data.py`: Shakespeare/FineWeb loader and HellaSwag dataset wrapper.
- `app/gpt/evaluate.py`: validation loss loop.
- `app/gpt/inference.py`: generation and HellaSwag row scoring helper.
- `app/fineweb.py`: FineWeb-Edu tokenization/sharding.
- `app/hellaswag.py`: HellaSwag downloader/formatter.
- `notebooks/`: experiments.

### Commands run

Import smoke checks could not run because the audit Python environment does not have `torch` installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "import train"
```

Result:

```text
ModuleNotFoundError: No module named 'torch'
```

This means runtime behavior could not be fully executed in this environment. However, static review found several import/runtime bugs that should be fixed regardless.

## 3. Bugs and missing things

### P0: README quickstart is wrong

README says:

```bash
python src/gpt2/train.py
```

Observed repo path:

```bash
app/gpt/train.py
```

Impact:

- A reviewer following README fails immediately.

Fix:

```bash
python app/gpt/train.py
```

Better fix: make the repo installable and expose console commands:

```bash
almostgpt-train --config configs/tiny_shakespeare.yaml
almostgpt-generate --checkpoint runs/tiny/latest.pt --prompt "Once upon a time"
almostgpt-eval --checkpoint runs/tiny/latest.pt --benchmark hellaswag
```

### P0: `requirements.txt` contains a stdlib package and misses direct dependencies

Current `requirements.txt`:

```text
numpy
logging
rich
ipykernel
torch
tiktoken
tqdm
datasets
```

Problems:

- `logging` is part of the Python standard library, not a pip dependency.
- `requests` is used by `app/hellaswag.py` but not listed.
- `tokenizer.py` imports from `tokenizers`, but requirements list `tiktoken`.
- dependencies are unpinned, so fresh installs are not reproducible.
- notebooks/dev dependencies are mixed with runtime dependencies.

Fix:

- remove `logging`
- add `requests`
- fix tokenizer import to `import tiktoken`
- move to `pyproject.toml`
- define extras:
  - base/runtime
  - dev
  - notebooks

Minimal `requirements.txt` if you do not move to `pyproject.toml` yet:

```text
torch>=2.2
numpy>=1.26
tiktoken>=0.7
rich>=13.7
tqdm>=4.66
datasets>=2.19
requests>=2.31
```

### P0: module imports are tangled around `train.py`

Many model/library modules import from `train.py`:

- `app/gpt/gpt.py`: `from train import DEVICE, GPTConfig, console`
- `app/gpt/attention.py`: `from train import DEVICE, GPTConfig`
- `app/gpt/transformer.py`: `from train import GPTConfig`
- `app/gpt/feed_forward.py`: `from train import GPTConfig`
- `app/gpt/embeddings.py`: `from train import DEVICE`
- `app/gpt/tokenizer.py`: `from train import DEVICE`
- `app/gpt/evaluate.py`: `from train import DEVICE`
- `app/gpt/checkpoint.py`: `from train import DEVICE, console`
- `app/gpt/data.py`: `from train import DATASET_PATH, Dataset, console, load_data`
- `app/gpt/inference.py`: `from train import console, model`

This is backwards. Training should import model code. Model code should not import the training script.

Impact:

- circular imports are likely once dependencies are installed
- standalone imports are fragile
- tests cannot import model components cleanly
- inference cannot be a separate command
- global `DEVICE` makes CPU/GPU testing harder

Fix:

- move `GPTConfig` to `app/gpt/config.py`
- move `console` to `app/gpt/console.py`
- move `DEVICE` selection into CLI/runtime functions
- pass `device` explicitly where needed
- keep `train.py` as an entrypoint only
- use package-relative imports:

```python
from .config import GPTConfig
from .attention import MaskedSelfAttention
```

Recommended structure:

```text
almostgpt/
  __init__.py
  config.py
  model.py
  attention.py
  embeddings.py
  kv_cache.py
  moe.py
  tokenizer.py
  data.py
  train.py
  evaluate.py
  generate.py
```

If you want to keep `app/gpt`, add `__init__.py` and still use relative imports.

### P0: `app/gpt/tokenizer.py` imports `tiktoken` incorrectly

Current code:

```python
from tokenizers import tiktoken
```

Correct direct dependency usage:

```python
import tiktoken
```

The official `tiktoken` package exposes `tiktoken.get_encoding(...)`, and the README example uses `import tiktoken`.

### P0: `app/gpt/data.py` imports `Dataset` from `train.py`

Current code:

```python
from train import DATASET_PATH, Dataset, console, load_data
```

`train.py` imports `DataLoader` from `torch.utils.data`, but not `Dataset`. So this is wrong.

Fix:

```python
from torch.utils.data import Dataset
```

Also remove `DATASET_PATH`, `console`, and `load_data` dependency from `train.py`. Put path/data helpers in `data.py`.

### P0: `app/gpt/inference.py` imports `model` from `train.py`

Current code:

```python
from train import console, model
```

But `model` is created inside:

```python
if __name__ == "__main__":
    model = GPT(...)
```

So importing `model` from `train` will fail for normal module use.

Fix:

- create `generate.py`
- load checkpoint explicitly
- build model from config
- expose a CLI:

```bash
almostgpt-generate --checkpoint checkpoints/latest.pt --prompt "Hello"
```

### P0: attention mask variable can be used before assignment

In `app/gpt/attention.py`, `sliding_window_mask` is only set in some branches.

Path:

- `use_cache=False`
- `self.sliding_window is None`
- `self.is_flash_attention=True`
- then `is_causal = ... and (sliding_window_mask is None)` references a variable that has not been assigned

Fix:

```python
sliding_window_mask = None
```

Set this near the top of `forward()`, before branches.

Then add a test:

```python
def test_attention_forward_without_sliding_window():
    ...
```

### P0: CPU training likely breaks because autocast always uses CUDA

In `app/gpt/train.py`:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
```

This is used even when `DEVICE == "cpu"`. Same issue in HellaSwag eval loop.

In `app/gpt/evaluate.py`:

```python
with torch.autocast(dtype=torch.float16, device_type="cuda"):
```

Impact:

- CPU smoke training likely fails or behaves incorrectly.
- A reviewer without GPU cannot try the project.

Fix:

```python
use_amp = device == "cuda"
with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
    ...
```

For CPU, skip autocast.

### P0: checkpoint save assumes `train_loader.current_shard` exists

In `app/gpt/checkpoint.py`:

```python
"current_shard": train_loader.current_shard,
```

But `CustomDataLoader` sets `current_shard` only for `dataset == "fineweb"` in `reset()`. For Shakespeare, this attribute does not exist.

Impact:

- checkpoint save can fail in the default Shakespeare training path.

Fix:

```python
"current_shard": getattr(train_loader, "current_shard", None)
```

### P1: HellaSwag scoring helper has suspicious tensor indexing

In `app/gpt/inference.py`:

```python
shifted_tokens = tokens[..., 1:, :].contiguous()
shifted_mask = mask[..., 1:, :].contiguous()
```

But `app/hellaswag.py` returns:

```python
tokens = torch.zeros((4, max_len), dtype=torch.long)
mask = torch.zeros((4, max_len), dtype=torch.long)
```

Those tensors are 2D, not 3D. The indexing pattern with `[..., 1:, :]` expects another dimension and will likely fail.

Expected scoring shape:

- tokens: `[num_choices, T]`
- logits: `[num_choices, T, vocab_size]`
- mask: `[num_choices, T]`

The shifted code should look more like:

```python
shifted_logits = logits[:, :-1, :]
shifted_tokens = tokens[:, 1:]
shifted_mask = mask[:, 1:]
```

Then compute per-token cross entropy with `reduction="none"` and average over masked continuation tokens.

### P1: `model.compile()` call is wrong

In `app/gpt/train.py`:

```python
if hasattr(torch, "compile"):
    model.compile()
```

Correct pattern is:

```python
if hasattr(torch, "compile"):
    model = torch.compile(model)
```

If you want to keep it optional, hide behind a config flag because compile startup can be slow and not always supported.

### P1: generation length semantics are confusing

In `GPT.generate()`:

```python
while tokens.size(1) < max_output_tokens:
```

This treats `max_output_tokens` as final total sequence length, not number of new tokens. If prompt has length 30 and `max_output_tokens=32`, it generates only 2 tokens.

Rename or fix:

- `max_length`: total length
- `max_new_tokens`: generated continuation length

Recommended:

```python
for _ in range(max_new_tokens):
    ...
```

### P1: `top_k` needs guardrails

Current:

```python
v, _ = torch.topk(filtered, k, dim=-1)
```

If `k > vocab_size`, this fails.

Fix:

```python
k = min(int(k), logits.size(-1))
```

Also define behavior for `k <= 0`: no filtering or raise `ValueError`.

### P1: MoE is now in the default block path

In `app/gpt/transformer.py`:

```python
# self.feed_forward = FeedForward(config)
self.MoE = MixtureOfExperts(config.n_embed, 4 * config.n_embed, 32)
```

The README still reads like a mostly GPT-2 architecture, but the active block uses MoE by default.

Impact:

- model is much heavier than expected
- parameter count jumps
- MoE routing/drop behavior changes training dynamics
- the README does not disclose the active architecture accurately

Fix:

- Make `feed_forward_type` a real config field:
  - `"mlp"`
  - `"swiglu"`
  - `"moe"`
- Default to dense MLP/SwiGLU for baseline.
- Make MoE an explicit experiment.

### P1: MoE drops overflow tokens to zero without reporting drop rate

In `app/gpt/mixture_of_experts.py`, tokens beyond expert capacity are not processed:

```python
end_cap = min(end, start + capacity)
...
output_sorted[start:end_cap] = out_e
```

The rest remain zeros and then get scattered back.

This can be valid for a Switch-style capacity mechanism, but it must be tracked:

- dropped token count
- dropped token rate
- per-expert load
- auxiliary loss
- z-loss

Add debug metrics in training logs. Otherwise it is impossible to tell whether MoE is learning or silently dropping a lot of tokens.

### P1: RoPE implementation likely flattens dimensions incorrectly

In `app/gpt/embeddings.py`:

```python
rotated_x = torch.flatten(rotated_x, 2, 3)
```

Shape before flatten is documented as:

```text
[B, T, n_dim, d_head/2, 2]
```

Flattening dimensions `2..3` merges `n_dim` and `d_head/2`, leaving the final `2` dimension separate. The desired final shape is `[B, T, n_dim, d_head]`, so this should likely flatten the last two dimensions:

```python
rotated_x = torch.flatten(rotated_x, -2)
```

Add a shape test:

```python
x = torch.randn(2, 8, 4, 16)
assert rope(x).shape == x.shape
```

### P1: no tests

No `tests/` directory was observed.

Add the smallest useful test suite:

- config construction
- tokenizer encode/decode round trip
- attention output shape
- causal mask prevents future-token leakage
- RoPE preserves shape
- KV cache append and rolling window behavior
- GPT forward returns `[B, T, vocab]` and scalar loss
- generation returns prompt + `max_new_tokens`
- checkpoint save/load restores step and loader state
- HellaSwag scoring chooses the lowest continuation loss
- MoE returns shape, aux loss, and exposes/drop metrics

### P1: no installable package

Current code depends on working directory because imports are like:

```python
from attention import MaskedSelfAttention
```

This only works when running from `app/gpt`. It does not work reliably as a package.

Fix:

- add `pyproject.toml`
- add package directory and `__init__.py`
- use relative imports
- expose CLI scripts

### P1: no experiment tracking

The code logs loss, validation loss, and HellaSwag accuracy, but there is no durable experiment table.

Add `runs/` output:

```text
runs/tiny-shakespeare-001/
  config.yaml
  metrics.jsonl
  checkpoints/latest.pt
  samples.txt
  eval.json
```

Log:

- step
- train loss
- val loss
- lr
- grad norm
- tokens/sec
- device
- parameter count
- effective batch size
- context length
- HellaSwag accuracy
- MoE load/drop metrics if MoE enabled

### P2: tracked generated files should be cleaned

Tracked:

- `app/logs/train.log`

Usually remove this from git and ignore logs:

```gitignore
app/logs/
runs/
checkpoints/
*.pt
*.bin
*.pkl
```

Notebooks can stay if they are polished and intentionally part of the project. If they are scratch, move to `experiments/` or clean outputs before commit.

### P2: no LICENSE

No license file was observed. Add one if public/interview-facing.

Also document dataset licenses:

- Shakespeare/Sherlock source
- HellaSwag
- FineWeb-Edu

## 4. What to add for maximum interview value

### 1. Make the baseline boring and correct

Before MoE/RoPE/GQA, prove a tiny GPT can train and overfit a small batch.

Add:

```bash
almostgpt-train --config configs/tiny_shakespeare.yaml
```

Expected result:

- runs on CPU
- completes in under 2 minutes
- loss decreases
- writes a sample generation
- writes metrics JSONL

This is the single highest-value improvement.

### 2. Add an ablation table

Create `docs/ablations.md`:

| Run | Model | Params | Context | Dataset | Val loss | HellaSwag | Tok/sec | Notes |
|---|---:|---:|---:|---|---:|---:|---:|---|
| baseline | Pre-LN GPT + GeLU | TBD | 256 | tiny Shakespeare | TBD | n/a | TBD | CPU smoke |
| swiglu | Pre-LN GPT + SwiGLU | TBD | 256 | tiny Shakespeare | TBD | n/a | TBD | compare FFN |
| rope | RoPE | TBD | 256 | tiny Shakespeare | TBD | n/a | TBD | compare positional |
| gqa | grouped K/V heads | TBD | 256 | tiny Shakespeare | TBD | n/a | TBD | compare KV memory |
| moe | top-1 MoE | TBD | 256 | tiny Shakespeare | TBD | n/a | TBD | report drop rate |

This turns the repo from code collection into research engineering.

### 3. Add a "from scratch but checked against PyTorch" test

For attention:

- compare your manual attention path against `torch.nn.functional.scaled_dot_product_attention`
- use small tensors
- fixed seed
- no dropout
- assert close outputs

This shows rigor.

### 4. Add model card / training card

Create `docs/model-card.md`:

- architecture
- parameter count
- tokenizer
- dataset
- training tokens
- hardware
- metrics
- limitations
- intended use
- non-goals

### 5. Add a KV cache benchmark

Measure generation with and without cache:

```bash
almostgpt-bench-kv --checkpoint runs/tiny/latest.pt --prompt-len 128 --new-tokens 128
```

Report:

- latency/token
- memory use if easy
- speedup
- correctness check: cached and uncached logits close for same prefix

### 6. Add HellaSwag mini-eval

The repo already includes HellaSwag logic. Make it reliable:

```bash
almostgpt-eval-hellaswag --checkpoint runs/small/latest.pt --limit 100
```

Report:

- raw accuracy
- normalized accuracy
- number of examples
- seed
- tokenization method

For a small student model, score can be low. That is okay. The value is building the evaluator correctly.

### 7. Add architecture notes that cite papers

Create `docs/architecture.md` with sections:

- GPT-style decoder-only Transformer
- Pre-LN versus Post-LN
- RoPE
- GQA/MQA
- KV cache
- SwiGLU
- MoE/Switch routing
- training loop
- evaluation

Tie each feature to the exact file implementing it.

## 5. Productionizing this repo

"Production" for this repo should mean reproducible training/evaluation and clean inference, not serving a huge LLM.

### Phase 1: package and import cleanup

1. Add `pyproject.toml`.
2. Create a real package namespace.
3. Replace cross-imports from `train.py`.
4. Fix tokenizer import.
5. Fix CPU autocast.
6. Fix checkpoint `current_shard`.
7. Add `tests/`.
8. Add CI.

Exit criteria:

```bash
python -m pip install -e ".[dev]"
pytest
almostgpt-train --config configs/tiny_shakespeare.yaml --max-steps 5
almostgpt-generate --checkpoint runs/tiny/latest.pt --prompt "Hello"
```

### Phase 2: reproducible experiments

1. Use YAML/TOML config files.
2. Save config with every run.
3. Save metrics to JSONL.
4. Save checkpoint atomically.
5. Save generated samples.
6. Record git SHA.
7. Record device and PyTorch version.
8. Add `docs/ablations.md`.

### Phase 3: inference CLI

Add:

```bash
almostgpt-generate \
  --checkpoint runs/small/latest.pt \
  --prompt "The meaning of life is" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

Features:

- deterministic mode with seed
- CPU fallback
- no training imports
- KV cache optional
- clear errors for missing checkpoint/config

### Phase 4: benchmark and evaluation

Add:

```bash
almostgpt-benchmark --config configs/tiny_shakespeare.yaml
almostgpt-eval-hellaswag --checkpoint runs/small/latest.pt --limit 1000
```

Metrics:

- tokens/sec
- p50/p95 step time
- peak GPU memory if CUDA
- validation loss
- HellaSwag accuracy
- generation latency/token
- KV cache speedup

### Phase 5: optional demo

Do not build a big web app. A simple terminal demo is enough:

```bash
almostgpt-chat --checkpoint runs/small/latest.pt
```

For a from-scratch tiny model, call it "completion demo," not "ChatGPT clone."

## 6. README rewrite plan

### Top

```md
# AlmostGPT

A from-scratch PyTorch GPT systems lab: decoder-only Transformer, RoPE, grouped-query attention, KV cache, SwiGLU, MoE experiments, checkpointed training, text generation, and HellaSwag evaluation.
```

### Demo

Show commands that actually run:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
almostgpt-train --config configs/tiny_shakespeare.yaml --max-steps 50
almostgpt-generate --checkpoint runs/tiny/latest.pt --prompt "To be or not to be"
```

### What is implemented

Separate stable from experimental:

Stable:

- GPT decoder block
- causal self-attention
- RMSNorm/LayerNorm
- RoPE
- SwiGLU/GeLU FFN
- top-k/top-p sampling
- checkpoint save/load
- tiny Shakespeare training

Experimental:

- GQA
- rolling KV cache
- MoE
- LoRA
- HellaSwag evaluator
- FineWeb-Edu preprocessing

### Architecture table

| Feature | File | Status | Test |
|---|---|---|---|
| Causal attention | `app/gpt/attention.py` | stable | yes/no |
| RoPE | `app/gpt/embeddings.py` | needs shape test | yes/no |
| KV cache | `app/gpt/kv_cache.py` | experimental | yes/no |
| MoE | `app/gpt/mixture_of_experts.py` | experimental | yes/no |
| Checkpointing | `app/gpt/checkpoint.py` | needs fix | yes/no |

### Results

Add real numbers:

```md
## Results

| Config | Dataset | Steps | Val loss | Tok/sec | Device |
|---|---|---:|---:|---:|---|
| tiny | Shakespeare | 500 | TBD | TBD | CPU |
```

### Known limitations

- tiny model is not comparable to production LLMs
- no safety tuning
- generated text may be incoherent
- training data is small by default
- FineWeb/HellaSwag downloads require network
- GPU performance depends on hardware

## 7. Production-level practices

### Testing

Add tests in this order:

1. `test_config.py`
2. `test_tokenizer.py`
3. `test_attention.py`
4. `test_rope.py`
5. `test_kv_cache.py`
6. `test_model_forward.py`
7. `test_sampling.py`
8. `test_checkpoint.py`
9. `test_hellaswag.py`

Keep tests small. Use tiny configs:

```python
GPTConfig(vocab_size=128, context_length=16, n_layer=2, n_head=2, n_embed=32)
```

### CI

Add `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: pytest -q
      - run: almostgpt-train --config configs/tiny_shakespeare.yaml --max-steps 2
```

### Configs

Add:

```text
configs/
  tiny_shakespeare.yaml
  small_fineweb.yaml
  moe_tiny.yaml
  gqa_tiny.yaml
```

Fields:

- model dimensions
- tokenizer
- dataset
- batch size
- context length
- max steps
- learning rate
- warmup
- weight decay
- gradient accumulation
- AMP
- compile
- seed
- output dir

### Reproducibility

Set:

- Python random seed
- NumPy seed
- PyTorch seed
- CUDA seed if available
- deterministic flag when desired

Save:

- config
- git SHA
- dependency versions
- tokenizer name
- dataset/source
- checkpoint step

### Logging

Do not track `app/logs/train.log`.

Use:

```text
runs/<run-name>/metrics.jsonl
runs/<run-name>/train.log
```

### Release hygiene

Add:

- `LICENSE`
- `CONTRIBUTING.md` optional
- `docs/`
- CI badge
- `pyproject.toml`
- `.pre-commit-config.yaml` optional

## 8. System design things to involve

### Training system design

Explain:

```text
Dataset source -> tokenizer -> token shards -> batch loader -> model forward -> loss -> optimizer/scaler -> checkpoint -> metrics/eval -> generation samples
```

Key tradeoffs:

- streaming versus pre-tokenized shards
- CPU smoke config versus GPU training config
- dense MLP baseline versus MoE
- MHA versus GQA memory/speed
- cached versus uncached generation
- full validation versus sampled validation

### Inference design

Explain:

```text
Prompt -> tokenizer -> model prefill -> KV cache -> token sampling loop -> detokenizer -> text
```

Include:

- prefill phase
- decode phase
- top-k/top-p/temperature
- max_new_tokens
- cache window and attention sink
- deterministic seed

### Evaluation design

Explain HellaSwag:

```text
context + candidate ending -> token log-likelihood -> mask continuation tokens -> average loss -> choose lowest loss
```

This is a good CS interview topic because it tests tensor shapes and language-model evaluation understanding.

### MoE design

Explain:

```text
token hidden states -> router logits -> top-1 expert -> capacity limit -> expert FFN -> combine -> aux/z loss
```

Add metrics:

- per-expert token counts
- drop rate
- router entropy
- aux loss
- z-loss

### KV cache design

Explain:

- cache stores keys/values per layer
- prompt prefill writes many tokens
- decode appends one token per step
- rolling window bounds memory
- attention sink preserves early tokens

Add one correctness check:

- cached decode logits should match uncached decode logits for the same prefix when using equivalent attention context.

## 9. CS student additions that are worth doing

Do these:

1. Tiny CPU training smoke command.
2. Unit tests for attention, RoPE, KV cache, model forward, checkpointing.
3. HellaSwag mini-evaluator with correct masked loss.
4. KV cache speed benchmark.
5. Ablation table.
6. Model card.
7. Architecture diagram.
8. Clean package install.
9. CI.
10. Generation CLI.

Skip for now:

- custom CUDA FlashAttention kernel
- distributed training
- full chat UI
- RLHF
- serving infrastructure

Those are too much until the basic package is runnable and tested.

## 10. Best fix order

### First pass

1. Convert `app/gpt` into a real package.
2. Move config/console/device out of `train.py`.
3. Fix `tiktoken` import.
4. Fix `Dataset` import.
5. Fix attention `sliding_window_mask`.
6. Fix CPU autocast.
7. Fix checkpoint `current_shard`.
8. Fix README command.

### Second pass

1. Add tiny config.
2. Add tests.
3. Add pyproject/CLI scripts.
4. Add CI.
5. Remove tracked log file.

### Third pass

1. Add generation CLI.
2. Add HellaSwag mini-eval.
3. Add KV cache benchmark.
4. Add ablation docs.
5. Add model card.

## 11. External references used

- Transformer paper: `https://arxiv.org/abs/1706.03762`
- GPT-2 repository/paper reference: `https://github.com/openai/gpt-2`
- GPT-2 release explanation: `https://openai.com/index/better-language-models/`
- PyTorch scaled dot product attention docs: `https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html`
- tiktoken README: `https://github.com/openai/tiktoken/blob/main/README.md`
- RoPE/RoFormer paper: `https://arxiv.org/abs/2104.09864`
- GQA paper: `https://arxiv.org/abs/2305.13245`
- Switch Transformer / MoE paper: `https://arxiv.org/abs/2101.03961`
- HellaSwag paper: `https://arxiv.org/abs/1905.07830`
- HellaSwag dataset card: `https://huggingface.co/datasets/Rowan/hellaswag`
- FineWeb-Edu dataset card: `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu`

## 12. Final recommendation

This repo has real interview potential, but it currently looks like an active scratchpad. Stop adding features for now. Make it importable, runnable, tested, and measured.

The killer version is not huge. It is:

- `pip install -e ".[dev]"`
- `pytest`
- `almostgpt-train --config configs/tiny_shakespeare.yaml --max-steps 50`
- `almostgpt-generate --checkpoint runs/tiny/latest.pt --prompt "Once upon a time"`
- a README table showing what RoPE/GQA/MoE/KV cache changed in loss and speed

That is enough to stand out as a 4th-year CS project because it shows you understand both the ML architecture and the engineering needed to make it trustworthy.
