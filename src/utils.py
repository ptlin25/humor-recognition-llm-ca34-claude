"""
Shared utilities for humor probing experiments.

These functions are used by the new extension experiments (experiment_new_model.py,
experiment_cross_transfer.py). The original experiment_*.py files are unchanged and
contain their own copies of these functions.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SEED = 42


def extract_activations(model, tokenizer, texts, batch_size=32, max_length=128):
    """
    Forward pass through model; collect last-token hidden states at every layer.

    Args:
        model: HuggingFace causal LM with output_hidden_states=True
        tokenizer: corresponding tokenizer (pad_token must be set)
        texts: list of strings
        batch_size: inference batch size
        max_length: tokenizer truncation length

    Returns:
        dict mapping layer_idx (int) -> np.array of shape (N, hidden_dim)
    """
    device = next(model.parameters()).device
    model.eval()
    _d = model.config.to_dict()
    n_layers = _d.get("text_config", _d)["num_hidden_layers"] + 1  # +1 for embedding layer
    all_activations = {layer: [] for layer in range(n_layers)}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Use last non-padding token position for each example
        attention_mask = inputs["attention_mask"]
        last_token_pos = attention_mask.sum(dim=1) - 1

        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            batch_acts = []
            for b in range(hidden_state.shape[0]):
                pos = last_token_pos[b].item()
                batch_acts.append(hidden_state[b, pos, :].cpu().to(torch.float32).numpy())
            all_activations[layer_idx].append(np.stack(batch_acts))

    for layer in all_activations:
        all_activations[layer] = np.concatenate(all_activations[layer], axis=0)

    return all_activations


def probe_at_ranks(train_acts, train_labels, test_acts, test_labels,
                   ranks=None):
    """
    Train linear probes at constrained ranks via PCA + logistic regression.

    Args:
        train_acts: np.array (N_train, hidden_dim)
        train_labels: list or np.array of ints (0/1)
        test_acts: np.array (N_test, hidden_dim)
        test_labels: list or np.array of ints (0/1)
        ranks: list of int rank values to try; defaults to [1,2,4,8,16,32,64]

    Returns:
        list of dicts: [{"rank": int, "accuracy": float, "f1": float}, ...]
        Final entry is full-rank (no PCA).
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 64]

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_acts)
    test_s = scaler.transform(test_acts)

    results = []

    for rank in ranks:
        if rank >= min(train_s.shape):
            continue
        pca = PCA(n_components=rank, random_state=SEED)
        tr = pca.fit_transform(train_s)
        te = pca.transform(test_s)
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(tr, train_labels)
        preds = lr.predict(te)
        results.append({
            "rank": rank,
            "accuracy": float(accuracy_score(test_labels, preds)),
            "f1": float(f1_score(test_labels, preds, zero_division=0)),
        })

    # Full-rank baseline (no PCA)
    lr_full = LogisticRegression(max_iter=1000, random_state=SEED)
    lr_full.fit(train_s, train_labels)
    preds_full = lr_full.predict(test_s)
    results.append({
        "rank": int(train_s.shape[1]),
        "accuracy": float(accuracy_score(test_labels, preds_full)),
        "f1": float(f1_score(test_labels, preds_full, zero_division=0)),
    })

    return results


def mean_diff_accuracy(train_acts, train_labels, test_acts, test_labels):
    """
    Rank-1 mean-difference probe (Tigges et al. methodology).

    Computes the unit vector between class centroids in training data,
    projects test data onto it, and classifies by median threshold.

    Returns:
        float: classification accuracy on test set
    """
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    pos_mean = train_acts[train_labels == 1].mean(axis=0)
    neg_mean = train_acts[train_labels == 0].mean(axis=0)
    direction = pos_mean - neg_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    train_proj = train_acts @ direction
    test_proj = test_acts @ direction
    threshold = np.median(train_proj)
    preds = (test_proj > threshold).astype(int)
    return float(accuracy_score(test_labels, preds))
