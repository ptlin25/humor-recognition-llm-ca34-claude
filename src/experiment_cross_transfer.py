"""
Cross-dataset transfer probing experiment.

Trains probes on one dataset's training split and evaluates zero-shot on another
dataset's test split — no retraining. Tests whether the humor direction learned
from one data source generalizes to another.

Four transfer directions:
  easy   → hahackathon : Does the text-register direction generalize? (expect: near chance)
  hard   → hahackathon : Does the humor-quality direction generalize? (expect: slightly better)
  haha   → easy        : Does real humor signal generalize to easy task? (expect: high)
  haha   → hard        : Does HaHackathon probe generalize to Reddit quality? (expect: low)

Results saved to results/{model_slug}_cross_transfer.json.

Usage:
    # On Modal (real run) — run after experiment_new_model.py:
    python src/experiment_cross_transfer.py --model google/gemma-3-4b-it

    # Local pipeline test (random activations):
    python src/experiment_cross_transfer.py --mock --model google/gemma-3-4b-it
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


def make_mock_activations(n_samples, n_layers, hidden_dim):
    return {
        layer: np.random.randn(n_samples, hidden_dim).astype(np.float32)
        for layer in range(n_layers)
    }


def get_activations(model, tokenizer, texts, n_layers, hidden_dim, mock=False):
    """Extract activations or return mocks."""
    if mock:
        return make_mock_activations(len(texts), n_layers + 1, hidden_dim)
    return extract_activations(model, tokenizer, texts)


def run_transfer(transfer_id, train_acts, train_labels, test_acts, test_labels, n_layers):
    """
    Given pre-extracted activations for train and test datasets, run probing
    in transfer mode: probe trained on train, evaluated on test.

    Returns list of per-layer dicts.
    """
    print(f"\n  Transfer: {transfer_id}  "
          f"(train n={len(train_labels)}, test n={len(test_labels)})")

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

    best_layer = max(probe_by_layer, key=lambda x: x["mean_diff_acc"])
    rank1_best = best_layer["mean_diff_acc"]
    best_probe_layer = max(probe_by_layer,
                           key=lambda x: max(p["accuracy"] for p in x["probes"]))
    full_rank_best = max(p["accuracy"] for p in best_probe_layer["probes"])
    print(f"    Best rank-1 (any layer): {rank1_best:.3f}  "
          f"Best full-rank: {full_rank_best:.3f}")

    return probe_by_layer


def run_experiment(model_id, mock=False):
    """
    Run all four cross-dataset transfer directions.

    Requires: experiment_new_model.py must have been run first (to confirm
    data loaders work), but this script re-extracts activations independently
    to avoid storing large numpy arrays.
    """
    print("=" * 60)
    print(f"EXPERIMENT: Cross-Dataset Transfer — {model_id}")
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
    # Load all datasets
    # ------------------------------------------------------------------
    print("\nLoading datasets...")

    if mock:
        # Generate synthetic data to avoid any real dataset dependencies
        def _mock(n): return [f"text {i}" for i in range(n)], [i % 2 for i in range(n)]
        easy_train_texts,  easy_train_labels  = _mock(120)
        easy_test_texts,   easy_test_labels   = _mock(40)
        hard_train_texts,  hard_train_labels  = _mock(70)
        hard_test_texts,   hard_test_labels   = _mock(30)
        haha_train_texts,  haha_train_labels  = _mock(80)
        haha_test_texts,   haha_test_labels   = _mock(30)
    else:
        from data_preparation import load_short_jokes, generate_non_humor_texts
        from data_hahackathon import load_hahackathon

        # Easy task
        jokes = load_short_jokes(2000)
        factual = generate_non_humor_texts(2000)
        n_easy = min(len(jokes), len(factual), 2000)
        easy_texts = jokes[:n_easy] + factual[:n_easy]
        easy_labels = [1] * n_easy + [0] * n_easy
        combined = list(zip(easy_texts, easy_labels))
        random.shuffle(combined)
        easy_texts, easy_labels = zip(*combined)
        easy_texts, easy_labels = list(easy_texts), list(easy_labels)
        easy_n_train = int(len(easy_texts) * 0.6)
        easy_train_texts  = easy_texts[:easy_n_train]
        easy_train_labels = easy_labels[:easy_n_train]
        easy_test_texts   = easy_texts[easy_n_train:]
        easy_test_labels  = easy_labels[easy_n_train:]

        # Hard task
        from datasets import load_from_disk
        reddit_ds = load_from_disk(
            str(PROJECT_ROOT / "datasets" / "one_million_reddit_jokes" / "train"))
        low_score, high_score = [], []
        for row in reddit_ds:
            title = row.get("title", "") or ""
            selftext = row.get("selftext", "") or ""
            text = (title + " " + selftext).strip()
            if not text or "[removed]" in text or "[deleted]" in text:
                continue
            if len(text) < 20 or len(text) > 200:
                continue
            score = row.get("score", 0)
            if score <= 2:
                low_score.append(text)
            elif score >= 50:
                high_score.append(text)
        random.shuffle(low_score)
        random.shuffle(high_score)
        n_hard = min(len(low_score), len(high_score), 1000)
        hard_texts = high_score[:n_hard] + low_score[:n_hard]
        hard_labels = [1] * n_hard + [0] * n_hard
        combined = list(zip(hard_texts, hard_labels))
        random.shuffle(combined)
        hard_texts, hard_labels = zip(*combined)
        hard_texts, hard_labels = list(hard_texts), list(hard_labels)
        hard_n_train = int(len(hard_texts) * 0.7)
        hard_train_texts  = hard_texts[:hard_n_train]
        hard_train_labels = hard_labels[:hard_n_train]
        hard_test_texts   = hard_texts[hard_n_train:]
        hard_test_labels  = hard_labels[hard_n_train:]

        # HaHackathon
        haha_data = load_hahackathon(binary=True)
        haha_train_texts  = haha_data["train"]["texts"]
        haha_train_labels = haha_data["train"]["labels"]
        haha_test_texts   = haha_data["test"]["texts"]
        haha_test_labels  = haha_data["test"]["labels"]

    print(f"  Easy:        train={len(easy_train_texts)}, test={len(easy_test_texts)}")
    print(f"  Hard:        train={len(hard_train_texts)}, test={len(hard_test_texts)}")
    print(f"  HaHackathon: train={len(haha_train_texts)}, test={len(haha_test_texts)}")

    # ------------------------------------------------------------------
    # Extract activations (each unique split once)
    # ------------------------------------------------------------------
    print("\nExtracting activations...")

    t0 = time.time()
    easy_train_acts  = get_activations(model, tokenizer, easy_train_texts,  n_layers, hidden_dim, mock)
    easy_test_acts   = get_activations(model, tokenizer, easy_test_texts,   n_layers, hidden_dim, mock)
    hard_train_acts  = get_activations(model, tokenizer, hard_train_texts,  n_layers, hidden_dim, mock)
    hard_test_acts   = get_activations(model, tokenizer, hard_test_texts,   n_layers, hidden_dim, mock)
    haha_train_acts  = get_activations(model, tokenizer, haha_train_texts,  n_layers, hidden_dim, mock)
    haha_test_acts   = get_activations(model, tokenizer, haha_test_texts,   n_layers, hidden_dim, mock)
    print(f"  All activations extracted in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Run four transfer directions
    # ------------------------------------------------------------------
    transfers = {}

    # easy → hahackathon
    # Probe trained on Short Jokes vs factual, tested on HaHackathon is_humor
    transfers["easy_to_haha"] = run_transfer(
        "easy → HaHackathon",
        easy_train_acts, easy_train_labels,
        haha_test_acts, haha_test_labels,
        n_layers,
    )

    # hard → hahackathon
    # Probe trained on high-score vs low-score Reddit, tested on HaHackathon
    transfers["hard_to_haha"] = run_transfer(
        "hard → HaHackathon",
        hard_train_acts, hard_train_labels,
        haha_test_acts, haha_test_labels,
        n_layers,
    )

    # hahackathon → easy
    # Probe trained on HaHackathon, tested on Short Jokes vs factual
    transfers["haha_to_easy"] = run_transfer(
        "HaHackathon → easy",
        haha_train_acts, haha_train_labels,
        easy_test_acts, easy_test_labels,
        n_layers,
    )

    # hahackathon → hard
    # Probe trained on HaHackathon, tested on Reddit quality task
    transfers["haha_to_hard"] = run_transfer(
        "HaHackathon → hard",
        haha_train_acts, haha_train_labels,
        hard_test_acts, hard_test_labels,
        n_layers,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "model": model_id,
        "n_layers": n_layers,
        "hidden_size": hidden_dim,
        "mock": mock,
        "transfers": transfers,
    }

    slug = model_slug(model_id)
    output_path = PROJECT_ROOT / "results" / f"{slug}_cross_transfer.json"
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
    parser = argparse.ArgumentParser(description="Cross-dataset transfer probing")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--mock", action="store_true",
                        help="Use random activations (no GPU needed, for pipeline testing)")
    args = parser.parse_args()

    run_experiment(model_id=args.model, mock=args.mock)
