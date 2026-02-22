"""
Generalization Experiment: Cross-dataset humor probe transfer.

Datasets:
  A — Short Jokes vs. Factual (prepared_data.json)
  B — HaHackathon (SemEval 2021 Task 7), loaded from datasets/hahackathon/

For each dataset independently:
  - Extract GPT-2 hidden states (all layers, last-token pooling)
  - Train layerwise linear probes; report accuracy/F1 vs layer
  - PCA compression curves (probe accuracy vs number of components)
  - Compute effective dimension (min components to reach within 1% of full-rank)

Generalization:
  - Cross-dataset probe transfer: train on A, zero-shot eval on B (and B→A)
  - Direction transfer: learn humor direction on A, calibrate threshold on B train,
    evaluate on B test (and B→A)
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_A():
    """Load Dataset A from results/prepared_data.json.
    Returns train and test splits as (texts, labels) pairs.
    """
    path = PROJECT_ROOT / "results" / "prepared_data.json"
    with open(path) as f:
        data = json.load(f)
    train_texts = data["train"]["texts"]
    train_labels = data["train"]["labels"]
    test_texts = data["test"]["texts"]
    test_labels = data["test"]["labels"]
    return (train_texts, train_labels), (test_texts, test_labels)


def load_dataset_B():
    """Load Dataset B (HaHackathon) from local CSVs.
    Uses train.csv split 80/20 for train/test.
    Returns train and test splits as (texts, labels) pairs.
    """
    def _read_csv(split):
        path = PROJECT_ROOT / "datasets" / "hahackathon" / f"{split}.csv"
        texts, labels = [], []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"].strip()
                label = int(row["is_humor"])
                if text:
                    texts.append(text)
                    labels.append(label)
        return texts, labels

    texts, labels = _read_csv("train")

    # 80/20 split
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    n_test = int(len(texts) * 0.2)
    train_texts, train_labels = texts[n_test:], labels[n_test:]
    test_texts, test_labels = texts[:n_test], labels[:n_test]

    return (train_texts, train_labels), (test_texts, test_labels)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, texts, batch_size=64, max_length=128):
    """Extract hidden states from all layers using last-token pooling.
    Returns dict[layer_idx -> ndarray(n_texts, hidden_dim)].
    """
    model.eval()
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    all_activations = {layer: [] for layer in range(n_layers)}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        attention_mask = inputs["attention_mask"]
        last_token_pos = attention_mask.sum(dim=1) - 1

        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            batch_acts = []
            for b in range(hidden_state.shape[0]):
                pos = last_token_pos[b].item()
                batch_acts.append(hidden_state[b, pos, :].cpu().numpy())
            all_activations[layer_idx].append(np.stack(batch_acts))

    for layer in all_activations:
        all_activations[layer] = np.concatenate(all_activations[layer], axis=0)

    return all_activations


# ---------------------------------------------------------------------------
# Probing helpers
# ---------------------------------------------------------------------------

def linear_probe_full_rank(train_acts, train_labels, test_acts, test_labels):
    """Fit StandardScaler + LogisticRegression; return accuracy, f1, scaler, probe."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_acts)
    test_scaled = scaler.transform(test_acts)
    probe = LogisticRegression(max_iter=1000, random_state=SEED)
    probe.fit(train_scaled, train_labels)
    preds = probe.predict(test_scaled)
    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "f1": float(f1_score(test_labels, preds, zero_division=0)),
        "scaler": scaler,
        "probe": probe,
    }


def linear_probe_varying_rank(train_acts, train_labels, test_acts, test_labels,
                               ranks=(1, 2, 4, 8, 16, 32, 64)):
    """PCA compression curve: probe accuracy vs number of retained components."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_acts)
    test_scaled = scaler.transform(test_acts)

    results = []
    for rank in ranks:
        if rank > min(train_scaled.shape):
            continue
        pca = PCA(n_components=rank, random_state=SEED)
        train_r = pca.fit_transform(train_scaled)
        test_r = pca.transform(test_scaled)
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(train_r, train_labels)
        preds = lr.predict(test_r)
        results.append({
            "rank": rank,
            "accuracy": float(accuracy_score(test_labels, preds)),
            "f1": float(f1_score(test_labels, preds, zero_division=0)),
            "explained_var": float(sum(pca.explained_variance_ratio_)),
        })

    # Full-rank baseline
    lr_full = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_full.fit(train_scaled, train_labels)
    preds_full = lr_full.predict(test_scaled)
    results.append({
        "rank": train_scaled.shape[1],
        "accuracy": float(accuracy_score(test_labels, preds_full)),
        "f1": float(f1_score(test_labels, preds_full, zero_division=0)),
        "explained_var": 1.0,
    })

    return results


def mean_diff_probe(train_acts, train_labels, test_acts, test_labels):
    """Rank-1 probe via mean-difference direction (Tigges et al.).
    Returns accuracy, f1, threshold, and the direction vector.
    """
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    humor_mean = train_acts[train_labels == 1].mean(axis=0)
    non_humor_mean = train_acts[train_labels == 0].mean(axis=0)
    direction = humor_mean - non_humor_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    train_proj = train_acts @ direction
    test_proj = test_acts @ direction

    threshold = float(np.median(train_proj))
    preds = (test_proj > threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "f1": float(f1_score(test_labels, preds, zero_division=0)),
        "threshold": threshold,
        "direction": direction,  # kept in memory, not serialised
    }


def compute_effective_dimension(results_by_rank, tolerance=0.01):
    """Min number of PCA components to reach within `tolerance` of full-rank accuracy."""
    full_rank_acc = results_by_rank[-1]["accuracy"]
    target = full_rank_acc - tolerance
    for r in results_by_rank:
        if r["accuracy"] >= target:
            return int(r["rank"])
    return int(results_by_rank[-1]["rank"])


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def analyse_dataset(name, train_acts_all, train_labels, test_acts_all, test_labels, n_layers):
    """Run layerwise probing and PCA compression for one dataset.
    Returns a dict ready for JSON serialisation plus the best-layer artifacts.
    """
    probe_by_layer = []
    best_layer = 0
    best_acc = -1.0
    best_layer_train_acts = None
    best_layer_test_acts = None

    print(f"\n  Layerwise probing ({name})...")
    for layer_idx in tqdm(range(n_layers)):
        tr = train_acts_all[layer_idx]
        te = test_acts_all[layer_idx]

        full = linear_probe_full_rank(tr, train_labels, te, test_labels)
        md = mean_diff_probe(tr, train_labels, te, test_labels)
        results_by_rank = linear_probe_varying_rank(tr, train_labels, te, test_labels)

        probe_by_layer.append({
            "layer": layer_idx,
            "full_rank_acc": full["accuracy"],
            "full_rank_f1": full["f1"],
            "mean_diff_acc": md["accuracy"],
            "results_by_rank": [{k: v for k, v in r.items()} for r in results_by_rank],
        })

        if full["accuracy"] > best_acc:
            best_acc = full["accuracy"]
            best_layer = layer_idx
            best_layer_train_acts = tr
            best_layer_test_acts = te

    eff_dim = compute_effective_dimension(probe_by_layer[best_layer]["results_by_rank"])

    return {
        "name": name,
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "probe_by_layer": probe_by_layer,
        "best_layer": best_layer,
        "effective_dimension": eff_dim,
    }, best_layer, best_layer_train_acts, best_layer_test_acts, train_labels, test_labels


# ---------------------------------------------------------------------------
# Cross-dataset transfer
# ---------------------------------------------------------------------------

def probe_transfer(src_train_acts, src_train_labels,
                   tgt_test_acts, tgt_test_labels,
                   n_layers):
    """Train probe on source at each layer; evaluate on target (source scaler applied)."""
    results = []
    for layer_idx in tqdm(range(n_layers), desc="  Probe transfer"):
        src_tr = src_train_acts[layer_idx]
        tgt_te = tgt_test_acts[layer_idx]
        r = linear_probe_full_rank(src_tr, src_train_labels, tgt_te, tgt_test_labels)
        results.append({
            "layer": layer_idx,
            "accuracy": r["accuracy"],
            "f1": r["f1"],
        })
    return results


def direction_transfer(src_train_acts, src_train_labels,
                       tgt_train_acts, tgt_train_labels,
                       tgt_test_acts, tgt_test_labels,
                       src_best_layer):
    """Learn humor direction at src best layer; calibrate threshold on tgt train; eval on tgt test."""
    src_tr = src_train_acts[src_best_layer]
    tgt_tr = tgt_train_acts[src_best_layer]
    tgt_te = tgt_test_acts[src_best_layer]

    # Standardise using source statistics
    scaler = StandardScaler().fit(src_tr)
    src_tr_s = scaler.transform(src_tr)
    tgt_tr_s = scaler.transform(tgt_tr)
    tgt_te_s = scaler.transform(tgt_te)

    # Direction from source
    src_labels = np.array(src_train_labels)
    humor_mean = src_tr_s[src_labels == 1].mean(axis=0)
    non_humor_mean = src_tr_s[src_labels == 0].mean(axis=0)
    direction = humor_mean - non_humor_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Calibrate threshold on target train
    tgt_tr_proj = tgt_tr_s @ direction
    threshold = float(np.median(tgt_tr_proj))

    # Evaluate on target test
    tgt_te_proj = tgt_te_s @ direction
    preds = (tgt_te_proj > threshold).astype(int)
    tgt_test_labels = np.array(tgt_test_labels)

    return {
        "source_best_layer": src_best_layer,
        "accuracy": float(accuracy_score(tgt_test_labels, preds)),
        "f1": float(f1_score(tgt_test_labels, preds, zero_division=0)),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_generalization():
    print("=" * 60)
    print("EXPERIMENT: Cross-Dataset Generalization")
    print("=" * 60)

    # Load data
    print("\nLoading datasets...")
    (A_train_texts, A_train_labels), (A_test_texts, A_test_labels) = load_dataset_A()
    (B_train_texts, B_train_labels), (B_test_texts, B_test_labels) = load_dataset_B()

    print(f"  Dataset A: train={len(A_train_texts)}, test={len(A_test_texts)}")
    print(f"  Dataset B: train={len(B_train_texts)}, test={len(B_test_texts)}")
    print(f"  Dataset B humor rate (train): "
          f"{sum(B_train_labels)/len(B_train_labels):.2f}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model = model.to(DEVICE)
    model.eval()

    n_layers = model.config.num_hidden_layers + 1
    print(f"  Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")

    # Extract activations
    print("\nExtracting activations...")
    t0 = time.time()
    A_train_acts = extract_activations(model, tokenizer, A_train_texts)
    A_test_acts = extract_activations(model, tokenizer, A_test_texts)
    B_train_acts = extract_activations(model, tokenizer, B_train_texts)
    B_test_acts = extract_activations(model, tokenizer, B_test_texts)
    print(f"  Extraction took {time.time() - t0:.1f}s")

    # Per-dataset layerwise analysis
    print("\nAnalysing Dataset A...")
    A_results, A_best_layer, A_bl_train, A_bl_test, _, _ = analyse_dataset(
        "Short Jokes vs Factual",
        A_train_acts, A_train_labels, A_test_acts, A_test_labels, n_layers,
    )

    print("\nAnalysing Dataset B...")
    B_results, B_best_layer, B_bl_train, B_bl_test, _, _ = analyse_dataset(
        "HaHackathon (SemEval 2021 Task 7)",
        B_train_acts, B_train_labels, B_test_acts, B_test_labels, n_layers,
    )

    print(f"\n  A best layer: {A_best_layer}  "
          f"(acc={A_results['probe_by_layer'][A_best_layer]['full_rank_acc']:.3f}, "
          f"eff_dim={A_results['effective_dimension']})")
    print(f"  B best layer: {B_best_layer}  "
          f"(acc={B_results['probe_by_layer'][B_best_layer]['full_rank_acc']:.3f}, "
          f"eff_dim={B_results['effective_dimension']})")

    # Cross-dataset probe transfer
    print("\nProbe transfer A → B...")
    atob_probe = probe_transfer(
        A_train_acts, A_train_labels,
        B_test_acts, B_test_labels,
        n_layers,
    )

    print("\nProbe transfer B → A...")
    btoa_probe = probe_transfer(
        B_train_acts, B_train_labels,
        A_test_acts, A_test_labels,
        n_layers,
    )

    # Direction transfer
    print("\nDirection transfer A → B...")
    atob_dir = direction_transfer(
        A_train_acts, A_train_labels,
        B_train_acts, B_train_labels,
        B_test_acts, B_test_labels,
        A_best_layer,
    )

    print("\nDirection transfer B → A...")
    btoa_dir = direction_transfer(
        B_train_acts, B_train_labels,
        A_train_acts, A_train_labels,
        A_test_acts, A_test_labels,
        B_best_layer,
    )

    print("\n--- Transfer Summary ---")
    best_atob = max(atob_probe, key=lambda r: r["accuracy"])
    best_btoa = max(btoa_probe, key=lambda r: r["accuracy"])
    print(f"  Probe  A→B  best layer {best_atob['layer']}: "
          f"acc={best_atob['accuracy']:.3f}, f1={best_atob['f1']:.3f}")
    print(f"  Probe  B→A  best layer {best_btoa['layer']}: "
          f"acc={best_btoa['accuracy']:.3f}, f1={best_btoa['f1']:.3f}")
    print(f"  Direction A→B (layer {atob_dir['source_best_layer']}): "
          f"acc={atob_dir['accuracy']:.3f}, f1={atob_dir['f1']:.3f}")
    print(f"  Direction B→A (layer {btoa_dir['source_best_layer']}): "
          f"acc={btoa_dir['accuracy']:.3f}, f1={btoa_dir['f1']:.3f}")

    # Save results
    results = {
        "model": MODEL_NAME,
        "dataset_A": A_results,
        "dataset_B": B_results,
        "transfer_A_to_B": {
            "probe_transfer": atob_probe,
            "direction_transfer": {k: v for k, v in atob_dir.items() if k != "direction"},
        },
        "transfer_B_to_A": {
            "probe_transfer": btoa_probe,
            "direction_transfer": {k: v for k, v in btoa_dir.items() if k != "direction"},
        },
    }

    output_path = PROJECT_ROOT / "results" / "generalization_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_generalization()
