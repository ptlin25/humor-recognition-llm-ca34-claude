"""
Experiment 1 & 2: Activation collection, PCA analysis, and linear probing.

Extracts hidden representations from GPT-2 for humor/non-humor texts,
analyzes the singular value spectrum, and trains linear probes at varying ranks.
"""
import json
import sys
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load prepared dataset."""
    data_path = PROJECT_ROOT / "results" / "prepared_data.json"
    with open(data_path) as f:
        return json.load(f)


def extract_activations(model, tokenizer, texts, batch_size=32, max_length=128):
    """
    Extract hidden states from all layers of the model for the given texts.
    Returns: dict mapping layer_idx -> numpy array of shape (n_texts, hidden_dim)
    Uses the last non-padding token position.
    """
    model.eval()
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size

    all_activations = {layer: [] for layer in range(n_layers)}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get last non-padding token position for each example
        attention_mask = inputs["attention_mask"]
        # last token position = sum of attention mask - 1
        last_token_pos = attention_mask.sum(dim=1) - 1

        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            # hidden_state shape: (batch, seq_len, hidden_dim)
            batch_acts = []
            for b in range(hidden_state.shape[0]):
                pos = last_token_pos[b].item()
                batch_acts.append(hidden_state[b, pos, :].cpu().numpy())
            all_activations[layer_idx].append(np.stack(batch_acts))

    # Concatenate all batches
    for layer in all_activations:
        all_activations[layer] = np.concatenate(all_activations[layer], axis=0)

    return all_activations


def pca_analysis(humor_acts, non_humor_acts, layer_idx, results_dir):
    """
    PCA analysis of humor vs non-humor activation differences.
    Returns effective rank metrics.
    """
    # Compute the difference in means
    humor_mean = humor_acts.mean(axis=0)
    non_humor_mean = non_humor_acts.mean(axis=0)
    mean_diff = humor_mean - non_humor_mean
    mean_diff_norm = np.linalg.norm(mean_diff)

    # Stack all activations with labels for PCA
    all_acts = np.vstack([humor_acts, non_humor_acts])
    labels = np.array([1] * len(humor_acts) + [0] * len(non_humor_acts))

    # Center the data
    scaler = StandardScaler()
    all_acts_centered = scaler.fit_transform(all_acts)

    # PCA on the full data
    n_components = min(all_acts_centered.shape[0], all_acts_centered.shape[1], 100)
    pca = PCA(n_components=n_components)
    pca.fit(all_acts_centered)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # PCA on the humor-specific signal:
    # Project onto the between-class direction
    humor_centered = humor_acts - all_acts.mean(axis=0)
    non_humor_centered = non_humor_acts - all_acts.mean(axis=0)

    # Compute between-class scatter
    diff = humor_centered.mean(axis=0) - non_humor_centered.mean(axis=0)

    # PCA of difference vectors (each sample's deviation from its class mean)
    humor_devs = humor_acts - humor_mean
    non_humor_devs = non_humor_acts - non_humor_mean

    # Within-class PCA
    within_class = np.vstack([humor_devs, non_humor_devs])
    pca_within = PCA(n_components=min(50, within_class.shape[0], within_class.shape[1]))
    pca_within.fit(within_class)

    # Humor-discriminative subspace analysis:
    # Project data onto top-k PCs and measure separability
    # Use a proper stratified split: first half of each class for train, second for test
    n_humor = len(humor_acts)
    n_non_humor = len(non_humor_acts)
    train_idx = list(range(0, n_humor // 2)) + list(range(n_humor, n_humor + n_non_humor // 2))
    test_idx = list(range(n_humor // 2, n_humor)) + list(range(n_humor + n_non_humor // 2, n_humor + n_non_humor))

    separability_by_rank = []
    projected_all = pca.transform(all_acts_centered)
    for k in range(1, min(51, n_components + 1)):
        projected = projected_all[:, :k]
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(projected[train_idx], labels[train_idx])
        preds = lr.predict(projected[test_idx])
        acc = accuracy_score(labels[test_idx], preds)
        separability_by_rank.append(acc)

    return {
        "layer": layer_idx,
        "mean_diff_norm": float(mean_diff_norm),
        "explained_variance": explained_var.tolist(),
        "cumulative_variance": cumulative_var.tolist(),
        "separability_by_rank": separability_by_rank,
        "within_class_variance": pca_within.explained_variance_ratio_.tolist(),
    }


def linear_probe_varying_rank(train_acts, train_labels, test_acts, test_labels,
                               ranks=[1, 2, 4, 8, 16, 32, 64], layer_idx=0):
    """
    Train linear probes with dimensionality reduction at varying ranks.
    Uses PCA to reduce to rank-k before logistic regression.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_acts)
    test_scaled = scaler.transform(test_acts)

    results = []

    # Full rank baseline
    lr_full = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_full.fit(train_scaled, train_labels)
    full_acc = accuracy_score(test_labels, lr_full.predict(test_scaled))
    full_f1 = f1_score(test_labels, lr_full.predict(test_scaled))

    for rank in ranks:
        if rank > min(train_scaled.shape):
            continue

        pca = PCA(n_components=rank, random_state=SEED)
        train_reduced = pca.fit_transform(train_scaled)
        test_reduced = pca.transform(test_scaled)

        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(train_reduced, train_labels)
        preds = lr.predict(test_reduced)
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)

        results.append({
            "rank": rank,
            "accuracy": float(acc),
            "f1": float(f1),
            "explained_var_by_pca": float(sum(pca.explained_variance_ratio_)),
        })

    results.append({
        "rank": train_scaled.shape[1],
        "accuracy": float(full_acc),
        "f1": float(full_f1),
        "explained_var_by_pca": 1.0,
    })

    return results


def mean_diff_probe(train_acts, train_labels, test_acts, test_labels):
    """
    Single-direction probe using mean difference (rank-1).
    Following Tigges et al. methodology.
    """
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    humor_mean = train_acts[train_labels == 1].mean(axis=0)
    non_humor_mean = train_acts[train_labels == 0].mean(axis=0)
    direction = humor_mean - non_humor_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Project onto direction
    train_proj = train_acts @ direction
    test_proj = test_acts @ direction

    # Find threshold
    threshold = np.median(train_proj)
    preds = (test_proj > threshold).astype(int)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)

    return {"accuracy": float(acc), "f1": float(f1), "threshold": float(threshold)}


def run_experiment():
    """Main experiment: extract activations, run PCA, run probing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("EXPERIMENT 1 & 2: Activation Analysis + Linear Probing")
    print("=" * 60)

    # Load data
    data = load_data()
    train_texts = data["train"]["texts"]
    train_labels = data["train"]["labels"]
    val_texts = data["val"]["texts"]
    val_labels = data["val"]["labels"]
    test_texts = data["test"]["texts"]
    test_labels = data["test"]["labels"]

    print(f"\nDataset: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    # Load model
    print("\nLoading GPT-2 small...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(DEVICE)
    model.eval()

    print(f"Model: {model_name}, Device: {DEVICE}")
    print(f"Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")

    # Extract activations
    print("\n--- Extracting activations ---")
    t0 = time.time()
    train_acts = extract_activations(model, tokenizer, train_texts, batch_size=64)
    test_acts = extract_activations(model, tokenizer, test_texts, batch_size=64)
    print(f"Activation extraction took {time.time()-t0:.1f}s")

    results_dir = PROJECT_ROOT / "results"
    all_results = {
        "model": model_name,
        "hidden_size": model.config.hidden_size,
        "n_layers": model.config.num_hidden_layers,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "pca_results": [],
        "probe_results": [],
        "mean_diff_results": [],
    }

    # Per-layer analysis
    train_labels_np = np.array(train_labels)

    print("\n--- Per-layer PCA & Probing Analysis ---")
    for layer_idx in range(model.config.num_hidden_layers + 1):
        layer_train = train_acts[layer_idx]
        layer_test = test_acts[layer_idx]

        # Split by class for PCA analysis
        humor_train = layer_train[train_labels_np == 1]
        non_humor_train = layer_train[train_labels_np == 0]

        # PCA analysis
        pca_result = pca_analysis(humor_train, non_humor_train, layer_idx, results_dir)
        all_results["pca_results"].append(pca_result)

        # Linear probing at varying ranks
        probe_result = linear_probe_varying_rank(
            layer_train, train_labels, layer_test, test_labels,
            ranks=[1, 2, 4, 8, 16, 32, 64],
            layer_idx=layer_idx,
        )
        all_results["probe_results"].append({
            "layer": layer_idx,
            "results_by_rank": probe_result,
        })

        # Mean difference probe (rank-1)
        md_result = mean_diff_probe(layer_train, train_labels, layer_test, test_labels)
        md_result["layer"] = layer_idx
        all_results["mean_diff_results"].append(md_result)

        # Summary for this layer
        best_probe = max(probe_result, key=lambda x: x["accuracy"])
        rank1_probe = probe_result[0] if probe_result else {"accuracy": 0}
        print(f"  Layer {layer_idx:2d}: mean_diff_acc={md_result['accuracy']:.3f}, "
              f"rank1_acc={rank1_probe['accuracy']:.3f}, "
              f"best_rank={best_probe['rank']} (acc={best_probe['accuracy']:.3f})")

    # Save results
    output_path = results_dir / "activation_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return all_results


if __name__ == "__main__":
    results = run_experiment()
