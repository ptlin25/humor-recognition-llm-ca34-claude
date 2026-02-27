"""
Null-label control experiment for humor probing.

Verifies that probes are detecting genuine humor signal and not artefacts of
the activation geometry or probe pipeline.

Three conditions:
  real          — HaHackathon with true is_humor labels (positive control).
  shuffled      — HaHackathon texts, labels independently permuted.
                  Tests for spurious structure in humor-adjacent text.
  non_humor_rnd — Factual (non-humor) texts with balanced random labels.
                  Tests whether the probe finds signal in completely
                  non-humorous text when labels carry no information.

Expected: real >> 0.5, shuffled ≈ 0.5, non_humor_rnd ≈ 0.5.
If shuffled or non_humor_rnd score above chance, the probe is picking up
something other than humor (e.g., text-length, register, position bias).

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


def make_balanced_random_labels(n, seed):
    """Return n balanced (50/50) labels in random order."""
    rng = np.random.default_rng(seed)
    half = n // 2
    labels = [1] * half + [0] * (n - half)
    rng.shuffle(labels)
    return [int(x) for x in labels]


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
    # Non-humor dataset with random labels
    # Uses deduplicated factual sentences — no humor signal by construction.
    # Small (~99 unique texts) but sufficient for a null check.
    # ------------------------------------------------------------------
    print("\nLoading non-humor (factual) dataset...")

    if mock:
        nh_train_texts = [f"fact train {i}" for i in range(40)]
        nh_test_texts  = [f"fact test {i}"  for i in range(20)]
    else:
        from data_preparation import generate_non_humor_texts
        # Deduplicate while preserving order
        seen = set()
        nh_unique = []
        for t in generate_non_humor_texts(2000):
            if t not in seen:
                seen.add(t)
                nh_unique.append(t)
        n_nh_train = int(len(nh_unique) * 0.7)
        nh_train_texts = nh_unique[:n_nh_train]
        nh_test_texts  = nh_unique[n_nh_train:]

    nh_train_labels = make_balanced_random_labels(len(nh_train_texts), seed=SEED + 10)
    nh_test_labels  = make_balanced_random_labels(len(nh_test_texts),  seed=SEED + 11)
    print(f"  Non-humor factual: train={len(nh_train_texts)}, test={len(nh_test_texts)}")

    # ------------------------------------------------------------------
    # Extract activations once per unique text set
    # ------------------------------------------------------------------
    print("\nExtracting activations — HaHackathon...")
    t0 = time.time()

    if mock:
        def _mock_acts(n):
            return {
                layer: np.random.randn(n, hidden_dim).astype(np.float32)
                for layer in range(n_layers + 1)
            }
        train_acts    = _mock_acts(len(train_texts))
        test_acts     = _mock_acts(len(test_texts))
        nh_train_acts = _mock_acts(len(nh_train_texts))
        nh_test_acts  = _mock_acts(len(nh_test_texts))
    else:
        train_acts    = extract_activations(model, tokenizer, train_texts)
        test_acts     = extract_activations(model, tokenizer, test_texts)
        print(f"  HaHackathon done in {time.time() - t0:.1f}s")

        print("Extracting activations — non-humor factual...")
        t1 = time.time()
        nh_train_acts = extract_activations(model, tokenizer, nh_train_texts)
        nh_test_acts  = extract_activations(model, tokenizer, nh_test_texts)
        print(f"  Non-humor done in {time.time() - t1:.1f}s")

    print(f"  Total extraction time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Build shuffled-label condition (same HaHackathon texts, random labels)
    # ------------------------------------------------------------------
    shuffled_train_labels = permute_labels(train_labels, seed=SEED)
    shuffled_test_labels  = permute_labels(test_labels,  seed=SEED + 1)

    # ------------------------------------------------------------------
    # Run all three conditions
    # ------------------------------------------------------------------
    real_results     = run_probe_condition(
        "real labels",
        train_acts, train_labels,
        test_acts,  test_labels,
        n_layers,
    )
    shuffled_results = run_probe_condition(
        "shuffled labels (null — humor text)",
        train_acts, shuffled_train_labels,
        test_acts,  shuffled_test_labels,
        n_layers,
    )
    nh_rnd_results   = run_probe_condition(
        "random labels on non-humor text (null — factual text)",
        nh_train_acts, nh_train_labels,
        nh_test_acts,  nh_test_labels,
        n_layers,
    )

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    def _best(results):
        md  = max(r["mean_diff_acc"] for r in results)
        lr  = max(max(p["accuracy"] for p in r["probes"]) for r in results)
        return md, lr

    real_md,     real_lr     = _best(real_results)
    shuffled_md, shuffled_lr = _best(shuffled_results)
    nh_rnd_md,   nh_rnd_lr   = _best(nh_rnd_results)

    print("\n" + "=" * 60)
    print("SUMMARY (best layer, best rank)")
    print(f"  {'Condition':<38} {'mean-diff':>9} {'full-rank':>9}")
    print(f"  {'-'*38} {'-'*9} {'-'*9}")
    print(f"  {'real labels (HaHackathon)':<38} {real_md:>9.3f} {real_lr:>9.3f}")
    print(f"  {'shuffled labels (humor text)':<38} {shuffled_md:>9.3f} {shuffled_lr:>9.3f}")
    print(f"  {'random labels (non-humor text)':<38} {nh_rnd_md:>9.3f} {nh_rnd_lr:>9.3f}")
    print(f"  Gap real→shuffled  (mean-diff): {real_md - shuffled_md:+.3f}")
    print(f"  Gap real→non-humor (mean-diff): {real_md - nh_rnd_md:+.3f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "model": model_id,
        "n_layers": n_layers,
        "hidden_size": hidden_dim,
        "mock": mock,
        "conditions": {
            "real":        {"dataset": "hahackathon_binary",
                            "n_train": len(train_texts), "n_test": len(test_texts),
                            "probe_by_layer": real_results},
            "shuffled":    {"dataset": "hahackathon_binary_shuffled_labels",
                            "n_train": len(train_texts), "n_test": len(test_texts),
                            "probe_by_layer": shuffled_results},
            "non_humor_rnd": {"dataset": "factual_random_labels",
                              "n_train": len(nh_train_texts), "n_test": len(nh_test_texts),
                              "probe_by_layer": nh_rnd_results},
        },
        "summary": {
            "real_best_mean_diff":        real_md,
            "shuffled_best_mean_diff":    shuffled_md,
            "non_humor_rnd_best_mean_diff": nh_rnd_md,
            "real_best_full_rank":        real_lr,
            "shuffled_best_full_rank":    shuffled_lr,
            "non_humor_rnd_best_full_rank": nh_rnd_lr,
        },
    }

    slug = model_slug(model_id)
    output_path = PROJECT_ROOT / "results" / f"{slug}_null_control.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if not mock and model is not None:
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
