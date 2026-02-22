# Week 1 Technical Spec

## Constraints

- New scripts import from `src/utils.py`; original `src/experiment_*.py` files are not edited
- Results saved to `results/{model_slug}_results.json` (slug = model ID with `/` → `-`)
- `--mock` flag replaces model with random activations of correct shape for local pipeline testing
- Seed: 42 everywhere (matching baseline)

---

## `src/utils.py`

Extracted verbatim from the 4 existing experiment files that each duplicate these functions.

```python
SEED = 42

def extract_activations(model, tokenizer, texts, batch_size=32, max_length=128):
    """
    Forward pass through model; collect last-token hidden states at every layer.
    Returns dict: {layer_idx (int): np.array of shape (N, hidden_dim)}
    """

def probe_at_ranks(train_acts, train_labels, test_acts, test_labels,
                   ranks=[1, 2, 4, 8, 16, 32, 64]):
    """
    StandardScaler → PCA(k) → LogisticRegression at each rank k.
    Also runs full-rank (no PCA) as final entry.
    Returns list of {"rank": int, "accuracy": float, "f1": float}
    """

def mean_diff_accuracy(train_acts, train_labels, test_acts, test_labels):
    """
    Rank-1 mean-difference probe (Tigges et al. methodology).
    Returns float accuracy.
    """
```

Implementation: copy from `src/experiment_activations.py` lines 164–236 (probe) and `src/experiment_hard_dataset.py` lines 152–163 (mean_diff), adjusting imports only.

---

## `src/data_hahackathon.py`

**Input files**: `datasets/hahackathon/train.csv`, `dev.csv`, `test.csv`
**Expected columns**: `text`, `is_humor` (0/1 int), `humor_rating` (float 0–5), `offense_rating` (float 0–5)

```python
PROJECT_ROOT = Path(__file__).parent.parent

def load_hahackathon(binary=True, rating_split=False, seed=42):
    """
    binary=True:
        Label = is_humor (0/1). Balanced 50/50 by undersampling majority class.
        Train pool = train.csv + dev.csv combined. Test = test.csv.

    rating_split=True:
        Label = 1 if humor_rating >= 3.5, 0 if humor_rating <= 1.5 (hard task analog).
        Drops texts in the middle range.

    Returns:
        {"train": {"texts": List[str], "labels": List[int]},
         "test":  {"texts": List[str], "labels": List[int]}}
    """
```

Saves `results/hahackathon_data.json` with counts and label distribution.

Runnable standalone:
```bash
python src/data_hahackathon.py
# Prints: n_train, n_test, label balance, first 3 examples
```

---

## `src/experiment_new_model.py`

**CLI**: `python src/experiment_new_model.py --model MODEL_ID [--mock] [--n_samples N]`

**Imports**: `from utils import extract_activations, probe_at_ranks, mean_diff_accuracy`

### Tasks

| Task | Positive | Negative | N/class (train/test) |
|------|----------|----------|----------------------|
| easy | Short Jokes (`datasets/short_jokes/`) | Factual sentences (from `data_preparation.NON_HUMOR_TEMPLATES` + `FACTUAL_SENTENCES`) | 1200 / 400 |
| hard | Reddit score ≥50 (`one_million_reddit_jokes`) | Reddit score ≤2 | 1000 / 300 |
| hahackathon | is_humor=1 | is_humor=0 | balanced, from CSVs |

Data loading:
- Easy task: import `load_short_jokes` and `generate_non_humor_texts` from `data_preparation.py`
- Hard task: import and adapt logic from `experiment_hard_dataset.load_harder_humor_data()`
- HaHackathon: import `load_hahackathon` from `data_hahackathon.py`

### Mock mode

```python
if args.mock:
    # Infer hidden_dim and n_layers from a tiny config or hardcode for Gemma 3 4B
    # Gemma 3 1B: 26 layers, hidden_dim=1152
    # Gemma 3 4B: 34 layers, hidden_dim=2560
    hidden_dims = {"google/gemma-3-1b-it": (26, 1152), "google/gemma-3-4b-it": (34, 2560)}
    n_layers, hidden_dim = hidden_dims.get(args.model, (32, 2048))
    train_acts = {l: np.random.randn(n_train, hidden_dim) for l in range(n_layers)}
    test_acts  = {l: np.random.randn(n_test,  hidden_dim) for l in range(n_layers)}
```

### Output format

`results/{model_slug}_results.json`:
```json
{
  "model": "google/gemma-3-4b-it",
  "hidden_size": 2560,
  "n_layers": 34,
  "tasks": {
    "easy": {
      "n_train": 2400,
      "n_test": 800,
      "probe_by_layer": [
        {"layer": 0, "probes": [{"rank": 1, "accuracy": 0.72, "f1": 0.72}, ...], "mean_diff_acc": 0.71}
      ]
    },
    "hard": { ... },
    "hahackathon": { ... }
  }
}
```

---

## `src/experiment_cross_transfer.py`

**CLI**: `python src/experiment_cross_transfer.py --model MODEL_ID [--mock]`

### Approach

Loads activations from `results/{model_slug}_results.json` if available (avoids re-running the model). If not found, exits with a message asking to run `experiment_new_model.py` first.

The activations themselves are not stored in the JSON (too large) — re-extract using the same texts and model. So this script re-runs the forward pass for each needed dataset, then trains probes and cross-evaluates.

Alternative (simpler): just run everything inline, accepting the extra forward passes.

### Transfer directions

```
easy_train_acts  → haha_test_acts   (easy probe, HaHackathon test)
hard_train_acts  → haha_test_acts   (hard probe, HaHackathon test)
haha_train_acts  → easy_test_acts   (HaHackathon probe, easy test)
haha_train_acts  → hard_test_acts   (HaHackathon probe, hard test)
```

For each direction × each layer:
- `mean_diff_accuracy(train_acts[l], train_labels, test_acts[l], test_labels)` → rank-1
- `probe_at_ranks(train_acts[l], train_labels, test_acts[l], test_labels)` → full rank sweep

### Output format

`results/{model_slug}_cross_transfer.json`:
```json
{
  "model": "google/gemma-3-4b-it",
  "transfers": {
    "easy_to_haha": {
      "probe_by_layer": [{"layer": 0, "mean_diff_acc": 0.51, "probes": [...]}]
    },
    "hard_to_haha": { ... },
    "haha_to_easy": { ... },
    "haha_to_hard": { ... }
  }
}
```

---

## `modal/run_probing.py`

```python
import modal
from pathlib import Path

app = modal.App("humor-probing")

hf_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate",
        "scikit-learn", "numpy", "datasets",
        "tqdm", "pandas",
    )
)

REPO_DIR = Path(__file__).parent.parent  # project root


@app.function(
    image=image,
    gpu="a10g",          # override to "t4" for 1B model via local_entrypoint
    volumes={"/hf-cache": hf_cache_vol},
    timeout=7200,
    mounts=[modal.Mount.from_local_dir(str(REPO_DIR), remote_path="/repo")],
)
def run_probing(model_id: str, skip_transfer: bool = False):
    import os, subprocess, sys
    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    cmd_probe = [sys.executable, "/repo/src/experiment_new_model.py", "--model", model_id]
    subprocess.run(cmd_probe, check=True, env=env)
    if not skip_transfer:
        cmd_xfer = [sys.executable, "/repo/src/experiment_cross_transfer.py", "--model", model_id]
        subprocess.run(cmd_xfer, check=True, env=env)


@app.local_entrypoint()
def main(model: str = "google/gemma-3-4b-it", skip_transfer: bool = False):
    run_probing.remote(model, skip_transfer)
```

**Run commands**:
```bash
# 1B model (fast, cheap — first iteration)
modal run modal/run_probing.py --model google/gemma-3-1b-it

# 4B model (main results)
modal run modal/run_probing.py --model google/gemma-3-4b-it

# Probing only, skip cross-transfer (if debugging)
modal run modal/run_probing.py --model google/gemma-3-4b-it --skip-transfer
```

**GPU selection**: The `@app.function` decorator hardcodes A10G. To use T4 for the 1B model, change `gpu="t4"` in the decorator or parameterize it via a Modal image config. Simplest: keep A10G for everything (overkill for 1B but avoids a code branch).

**Results retrieval**: The mount maps the local repo into the container. Results written to `/repo/results/` will appear in the local `results/` directory after the run completes.

---

## `src/visualize_new.py`

### Figure 6: GPT-2 vs Gemma 3 4B — rank-1 accuracy by normalized layer position

One subplot per task (easy, hard, HaHackathon). X-axis: normalized layer position 0→1 (like existing Figure 4). Y-axis: rank-1 (mean-diff) accuracy. Lines: GPT-2 (blue, from `activation_analysis_results.json` and `hard_dataset_results.json`) vs Gemma 3 4B (red, from `gemma-3-4b-it_results.json`).

### Figure 7: Cross-transfer heatmap

4-row × 2-column table (rank-1 acc at best layer, full-rank acc at best layer). Color-coded by accuracy. Rows: the 4 transfer directions. Columns: rank-1 / best-rank.

Saves:
- `results/plots/figure6_new_model_comparison.png`
- `results/plots/figure7_cross_transfer.png`

---

## Open Questions

1. **Qwen3 4B HF ID**: verify `Qwen/Qwen3-4B` vs `Qwen/Qwen3-4B-Instruct` on HuggingFace before running
2. **HaHackathon column names**: run `python src/data_hahackathon.py` after uploading CSVs and confirm output before Modal run
3. **Gemma 3 4B hidden dim / n_layers**: mock mode hardcodes these — verify against HuggingFace model card (expected: 34 layers, hidden_dim=2560 for 4B)
