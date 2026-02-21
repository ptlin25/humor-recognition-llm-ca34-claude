"""
Experiment: Repeat core analysis with GPT-2 (117M) for cross-model comparison.
Uses the same easy task (jokes vs factual) as Gemma to compare how model
architecture and scale affect humor representation rank.
"""
import json
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_activations(model, tokenizer, texts, batch_size=32, max_length=128):
    """Extract hidden states from all layers."""
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    all_activations = {layer: [] for layer in range(n_layers)}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
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


def probe_at_ranks(train_acts, train_labels, test_acts, test_labels, ranks=[1,2,4,8,16,32,64]):
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
        lr = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced", random_state=SEED)
        lr.fit(tr, train_labels)
        acc = accuracy_score(test_labels, lr.predict(te))
        results.append({"rank": rank, "accuracy": float(acc)})
    # Full rank
    lr = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced", random_state=SEED)
    lr.fit(train_s, train_labels)
    results.append({"rank": train_s.shape[1], "accuracy": float(accuracy_score(test_labels, lr.predict(test_s)))})
    return results


def mean_diff_acc(train_acts, train_labels, test_acts, test_labels):
    tl = np.array(train_labels)
    pos_mean = train_acts[tl == 1].mean(axis=0)
    neg_mean = train_acts[tl == 0].mean(axis=0)
    d = pos_mean - neg_mean
    d = d / (np.linalg.norm(d) + 1e-8)
    threshold = np.median(train_acts @ d)
    preds = (test_acts @ d > threshold).astype(int)
    return float(accuracy_score(test_labels, preds))


def run_pythia_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("EXPERIMENT: GPT-2 Cross-Model Comparison")
    print("=" * 60)

    # Load prepared data (jokes vs factual)
    with open(PROJECT_ROOT / "results" / "prepared_data.json") as f:
        data = json.load(f)

    model_name = "gpt2"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")

    # Extract activations for easy task
    train_acts = extract_activations(model, tokenizer, data["train"]["texts"])
    test_acts = extract_activations(model, tokenizer, data["test"]["texts"])

    results = {
        "model": model_name,
        "hidden_size": model.config.hidden_size,
        "n_layers": model.config.num_hidden_layers,
        "probe_results": [],
        "mean_diff_results": [],
    }

    for layer in range(model.config.num_hidden_layers + 1):
        probes = probe_at_ranks(train_acts[layer], data["train"]["labels"],
                                test_acts[layer], data["test"]["labels"])
        md = mean_diff_acc(train_acts[layer], data["train"]["labels"],
                           test_acts[layer], data["test"]["labels"])
        results["probe_results"].append({"layer": layer, "probes": probes, "mean_diff_acc": md})
        best = max(probes, key=lambda x: x["accuracy"])
        print(f"  Layer {layer:2d}: md_acc={md:.3f}, r1={probes[0]['accuracy']:.3f}, "
              f"best_rank={best['rank']}(acc={best['accuracy']:.3f})")

    output_path = PROJECT_ROOT / "results" / "gpt2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    run_pythia_experiment()
