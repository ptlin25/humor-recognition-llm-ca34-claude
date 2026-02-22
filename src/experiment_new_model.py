"""
Experiment: Probing humor representations in a new model (Gemma 3 / Qwen3).

Replicates the GPT-2 probing pipeline on a parameterized model. Runs three task
variants to allow direct comparison with the baseline:
  - easy:        Short Jokes vs factual sentences
  - hard:        High-score Reddit jokes vs low-score Reddit jokes
  - hahackathon: HaHackathon binary (is_humor)

Results are saved to results/{model_slug}_results.json.

Usage:
    # On Modal (real run):
    python src/experiment_new_model.py --model google/gemma-3-4b-it

    # Local pipeline test (random activations, no GPU):
    python src/experiment_new_model.py --mock --model google/gemma-3-4b-it
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

# Model architecture info for mock mode (n_layers includes embedding layer)
# Values: (num_hidden_layers, hidden_size) from HuggingFace model configs
MODEL_CONFIGS = {
    "google/gemma-3-1b-it":  (18, 1152),
    "google/gemma-3-4b-it":  (34, 2560),
    "Qwen/Qwen3-4B":         (36, 2560),
    "Qwen/Qwen3-4B-Instruct": (36, 2560),
}


def model_slug(model_id):
    return model_id.replace("/", "-")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_easy_task(n_samples=2000):
    """Short Jokes vs factual sentences (same as experiment_activations.py easy task)."""
    from data_preparation import load_short_jokes, generate_non_humor_texts
    humor = load_short_jokes(n_samples)
    non_humor = generate_non_humor_texts(n_samples)
    n = min(len(humor), len(non_humor), n_samples)
    texts = humor[:n] + non_humor[:n]
    labels = [1] * n + [0] * n
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    # 70/30 train/test split (matching experiment_activations.py: ~60/20/20 but simplified)
    n_train = int(len(texts) * 0.6)
    return (
        texts[:n_train], labels[:n_train],
        texts[n_train:], labels[n_train:],
    )


def load_hard_task(n_samples=1000):
    """High-score Reddit (≥50) vs low-score Reddit (≤2)."""
    from datasets import load_from_disk
    reddit_ds = load_from_disk(str(PROJECT_ROOT / "datasets" / "one_million_reddit_jokes" / "train"))

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
    n = min(len(low_score), len(high_score), n_samples)

    texts = high_score[:n] + low_score[:n]
    labels = [1] * n + [0] * n
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    n_train = int(len(texts) * 0.7)
    return (
        texts[:n_train], labels[:n_train],
        texts[n_train:], labels[n_train:],
    )


def load_hahackathon_task():
    """HaHackathon binary (is_humor), balanced 50/50."""
    from data_hahackathon import load_hahackathon
    data = load_hahackathon(binary=True)
    return (
        data["train"]["texts"], data["train"]["labels"],
        data["test"]["texts"], data["test"]["labels"],
    )


# ---------------------------------------------------------------------------
# Mock mode: random activations with correct shape
# ---------------------------------------------------------------------------

def make_mock_activations(n_samples, n_layers, hidden_dim):
    """Generate random activations for local pipeline testing."""
    return {
        layer: np.random.randn(n_samples, hidden_dim).astype(np.float32)
        for layer in range(n_layers)
    }


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_task(task_name, train_texts, train_labels, test_texts, test_labels,
             model, tokenizer, n_layers, hidden_dim, mock=False):
    """
    Run probing for a single task. Returns per-layer probe results dict.
    """
    print(f"\n--- Task: {task_name} ---")
    print(f"  Train: {len(train_texts)} | Test: {len(test_texts)}")

    if mock:
        train_acts = make_mock_activations(len(train_texts), n_layers + 1, hidden_dim)
        test_acts = make_mock_activations(len(test_texts), n_layers + 1, hidden_dim)
    else:
        t0 = time.time()
        train_acts = extract_activations(model, tokenizer, train_texts)
        test_acts = extract_activations(model, tokenizer, test_texts)
        print(f"  Activation extraction: {time.time() - t0:.1f}s")

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
        best = max(probes, key=lambda x: x["accuracy"])
        print(f"  Layer {layer:3d}: md={md_acc:.3f}  r1={probes[0]['accuracy']:.3f}  "
              f"best_rank={best['rank']}(acc={best['accuracy']:.3f})")

    return {
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "probe_by_layer": probe_by_layer,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(model_id, mock=False, n_samples=None, tasks=None):
    """
    Run probing experiments for the given model on all specified tasks.

    Args:
        model_id: HuggingFace model ID (e.g. "google/gemma-3-4b-it")
        mock: If True, use random activations (no model loading)
        n_samples: Override default sample count per class (useful for quick tests)
        tasks: List of task names to run; defaults to ["easy", "hard", "hahackathon"]
    """
    if tasks is None:
        tasks = ["easy", "hard", "hahackathon"]

    print("=" * 60)
    print(f"EXPERIMENT: New Model Probing — {model_id}")
    print(f"  Tasks: {tasks}  |  Mock: {mock}  |  Device: {DEVICE}")
    print("=" * 60)

    # Determine model architecture
    if mock:
        cfg = MODEL_CONFIGS.get(model_id, (32, 2048))
        n_layers, hidden_dim = cfg
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
        # to_dict() normalises nested sub-configs (e.g. Gemma 3 multimodal
        # stores text arch under text_config) into plain dicts uniformly.
        _d = model.config.to_dict()
        _text = _d.get("text_config", _d)
        n_layers = _text["num_hidden_layers"]
        hidden_dim = _text["hidden_size"]
        print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}, device={DEVICE}")

    results = {
        "model": model_id,
        "n_layers": n_layers,
        "hidden_size": hidden_dim,
        "mock": mock,
        "tasks": {},
    }

    def mock_data(n_train, n_test):
        """Generate synthetic data for mock mode — avoids loading real datasets."""
        train_texts = [f"mock training text {i}" for i in range(n_train)]
        train_labels = [i % 2 for i in range(n_train)]
        test_texts = [f"mock test text {i}" for i in range(n_test)]
        test_labels = [i % 2 for i in range(n_test)]
        return train_texts, train_labels, test_texts, test_labels

    real_task_loaders = {
        "easy": lambda: load_easy_task(n_samples or 2000),
        "hard": lambda: load_hard_task(n_samples or 1000),
        "hahackathon": load_hahackathon_task,
    }

    mock_sizes = {"easy": (120, 40), "hard": (70, 30), "hahackathon": (80, 30)}

    for task_name in tasks:
        if task_name not in real_task_loaders:
            print(f"  Unknown task '{task_name}', skipping.")
            continue
        if mock:
            n_tr, n_te = mock_sizes.get(task_name, (100, 40))
            train_texts, train_labels, test_texts, test_labels = mock_data(n_tr, n_te)
        else:
            train_texts, train_labels, test_texts, test_labels = real_task_loaders[task_name]()

        if n_samples is not None:
            # Truncate to n_samples per class for quick testing
            half = n_samples // 2
            train_pos = [(t, l) for t, l in zip(train_texts, train_labels) if l == 1][:half]
            train_neg = [(t, l) for t, l in zip(train_texts, train_labels) if l == 0][:half]
            combined = train_pos + train_neg
            random.shuffle(combined)
            train_texts, train_labels = zip(*combined) if combined else ([], [])
            train_texts, train_labels = list(train_texts), list(train_labels)

            test_pos = [(t, l) for t, l in zip(test_texts, test_labels) if l == 1][:half]
            test_neg = [(t, l) for t, l in zip(test_texts, test_labels) if l == 0][:half]
            combined = test_pos + test_neg
            random.shuffle(combined)
            test_texts, test_labels = zip(*combined) if combined else ([], [])
            test_texts, test_labels = list(test_texts), list(test_labels)

        task_results = run_task(
            task_name, train_texts, train_labels, test_texts, test_labels,
            model, tokenizer, n_layers, hidden_dim, mock=mock,
        )
        results["tasks"][task_name] = task_results

    # Save
    slug = model_slug(model_id)
    output_path = PROJECT_ROOT / "results" / f"{slug}_results.json"
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
    parser = argparse.ArgumentParser(description="New model probing experiment")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--mock", action="store_true",
                        help="Use random activations (no GPU needed, for pipeline testing)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Samples per class override (e.g. 50 for quick local test)")
    parser.add_argument("--tasks", nargs="+",
                        default=["easy", "hard", "hahackathon"],
                        choices=["easy", "hard", "hahackathon"],
                        help="Which tasks to run")
    args = parser.parse_args()

    run_experiment(
        model_id=args.model,
        mock=args.mock,
        n_samples=args.n_samples,
        tasks=args.tasks,
    )
