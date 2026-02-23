"""
Null-label control experiment for humor probing.

Verifies that probes are detecting genuine humor signal and not artefacts of
the activation geometry or probe pipeline.

Protocol:
  1. Load HaHackathon (binary, balanced 50/50).
  2. Extract activations with real labels (positive control).
  3. Permute labels independently on train and test sets (null condition).
  4. Run identical probe pipeline on both conditions.
  5. Expected: real-label >> 0.5, shuffled-label ≈ 0.5.

If shuffled accuracy is consistently above 0.5 it suggests the probe is
picking up spurious structure (e.g., text-length, position) rather than humor.

Results saved to results/{model_slug}_null_control.json.

Usage:
    # On Modal (real run):
    python src/experiment_null_control.py --model google/gemma-3-4b-it

    # Local pipeline test (random activations, no GPU):
    python src/experiment_null_control.py --mock --model google/gemma-3-4b-it
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import extract_activations, probe_at_ranks, mean_diff_accuracy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "google/gemma-3-1b-it":   (18, 1152),
    "google/gemma-3-4b-it":   (34, 2560),
    "Qwen/Qwen3-4B":          (36, 2560),
    "Qwen/Qwen3-4B-Instruct": (36, 2560),
}


def model_slug(model_id):
    return model_id.replace("/", "-")


def permute_labels(labels, seed):
    """Return a copy of labels with entries independently shuffled."""
    rng = np.random.default_rng(seed)
    arr = np.array(labels, dtype=int)
    rng.shuffle(arr)
    return arr.tolist()


def run_probe_condition(condition_name, train_acts, train_labels, test_acts, test_labels, n_layers):
    """
    Run probing at every layer for one label condition (real or shuffled).

    Returns list of per-layer dicts compatible with experiment_cross_transfer format.
    """
    print(f"\n  Condition: {condition_name}  "
          f"(train n={len(train_labels)}, test n={len(test_labels)})")

    # Sanity-check label balance
    train_pos = sum(train_labels)
    test_pos = sum(test_labels)
    print(f"    Train: {train_pos}/{len(train_labels)} positive "
          f"({100*train_pos/len(train_labels):.1f}%)")
    print(f"    Test:  {test_pos}/{len(test_labels)} positive "
          f"({100*test_pos/len(test_labels):.1f}%)")

    probe_by_layer = []
    for layer in range(n_layers + 1):
        probes = probe_at_ranks(
            train_acts[layer], train_labels,
            test_acts[layer], test_labels,
        )
        md_acc = mean_diff_accuracy(
            train_acts[layer], train_labels,
            test_acts[layer], test_labels,
        )
        probe_by_layer.append({
            "layer": layer,
            "mean_diff_acc": md_acc,
            "probes": probes,
        })

    best_md_layer = max(probe_by_layer, key=lambda x: x["mean_diff_acc"])
    best_md_acc = best_md_layer["mean_diff_acc"]
    best_lr_layer = max(probe_by_layer,
                        key=lambda x: max(p["accuracy"] for p in x["probes"]))
    best_lr_acc = max(p["accuracy"] for p in best_lr_layer["probes"])
    print(f"    Best rank-1 mean-diff (any layer): {best_md_acc:.3f}")
    print(f"    Best full-rank LR     (any layer): {best_lr_acc:.3f}")

    return probe_by_layer


def run_experiment(model_id, mock=False):
    print("=" * 60)
    print(f"EXPERIMENT: Null-Label Control — {model_id}")
    print(f"  Mock: {mock}  |  Device: {DEVICE}")
    print("=" * 60)

    cfg = MODEL_CONFIGS.get(model_id, (32, 2048))
    n_layers, hidden_dim = cfg

    if mock:
        model, tokenizer = None, None
        print(f"  Mock mode: n_layers={n_layers}, hidden_dim={hidden_dim}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\nLoading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
        ).to(DEVICE)
        model.eval()
        _d = model.config.to_dict()
        _text = _d.get("text_config", _d)
        n_layers = _text["num_hidden_layers"]
        hidden_dim = _text["hidden_size"]
        print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}, device={DEVICE}")

    # ------------------------------------------------------------------
    # Load HaHackathon
    # ------------------------------------------------------------------
    print("\nLoading HaHackathon dataset...")

    if mock:
        n_train, n_test = 80, 30
        train_texts = [f"train text {i}" for i in range(n_train)]
        test_texts  = [f"test text {i}"  for i in range(n_test)]
        train_labels = [i % 2 for i in range(n_train)]
        test_labels  = [i % 2 for i in range(n_test)]
    else:
        from data_hahackathon import load_hahackathon
        haha = load_hahackathon(binary=True)
        train_texts  = haha["train"]["texts"]
        train_labels = haha["train"]["labels"]
        test_texts   = haha["test"]["texts"]
        test_labels  = haha["test"]["labels"]

    print(f"  HaHackathon: train={len(train_texts)}, test={len(test_texts)}")

    # ------------------------------------------------------------------
    # Extract activations once (texts are unchanged between conditions)
    # ------------------------------------------------------------------
    print("\nExtracting activations...")
    t0 = time.time()

    if mock:
        train_acts = {
            layer: np.random.randn(len(train_texts), hidden_dim).astype(np.float32)
            for layer in range(n_layers + 1)
        }
        test_acts = {
            layer: np.random.randn(len(test_texts), hidden_dim).astype(np.float32)
            for layer in range(n_layers + 1)
        }
    else:
        train_acts = extract_activations(model, tokenizer, train_texts)
        test_acts  = extract_activations(model, tokenizer, test_texts)

    print(f"  Activations extracted in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Build shuffled-label condition
    # ------------------------------------------------------------------
    shuffled_train_labels = permute_labels(train_labels, seed=SEED)
    shuffled_test_labels  = permute_labels(test_labels,  seed=SEED + 1)

    # ------------------------------------------------------------------
    # Run both conditions
    # ------------------------------------------------------------------
    real_results     = run_probe_condition(
        "real labels",
        train_acts, train_labels,
        test_acts,  test_labels,
        n_layers,
    )
    shuffled_results = run_probe_condition(
        "shuffled labels (null control)",
        train_acts, shuffled_train_labels,
        test_acts,  shuffled_test_labels,
        n_layers,
    )

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY: Real vs Shuffled (best layer, best rank)")
    real_best_md     = max(r["mean_diff_acc"] for r in real_results)
    shuffled_best_md = max(r["mean_diff_acc"] for r in shuffled_results)
    real_best_lr     = max(max(p["accuracy"] for p in r["probes"]) for r in real_results)
    shuffled_best_lr = max(max(p["accuracy"] for p in r["probes"]) for r in shuffled_results)
    print(f"  Mean-diff (rank-1):  real={real_best_md:.3f}  shuffled={shuffled_best_md:.3f}")
    print(f"  Full-rank LR:        real={real_best_lr:.3f}  shuffled={shuffled_best_lr:.3f}")
    print(f"  Gap (mean-diff): {real_best_md - shuffled_best_md:+.3f}")
    print(f"  Gap (full-rank): {real_best_lr - shuffled_best_lr:+.3f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "model": model_id,
        "n_layers": n_layers,
        "hidden_size": hidden_dim,
        "mock": mock,
        "dataset": "hahackathon_binary",
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "conditions": {
            "real": real_results,
            "shuffled": shuffled_results,
        },
        "summary": {
            "real_best_mean_diff": real_best_md,
            "shuffled_best_mean_diff": shuffled_best_md,
            "real_best_full_rank": real_best_lr,
            "shuffled_best_full_rank": shuffled_best_lr,
        },
    }

    slug = model_slug(model_id)
    output_path = PROJECT_ROOT / "results" / f"{slug}_null_control.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Null-label control for humor probing")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--mock", action="store_true",
                        help="Use random activations (no GPU needed, for pipeline testing)")
    args = parser.parse_args()

    run_experiment(model_id=args.model, mock=args.mock)
