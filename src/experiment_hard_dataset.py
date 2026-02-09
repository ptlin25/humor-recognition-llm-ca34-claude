"""
Experiment with harder non-humor data: use Reddit posts from non-joke subreddits
matched in length and style, to control for surface-level differences.
Also uses a larger sentiment dataset from HuggingFace for a fair comparison.
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
from datasets import load_from_disk, load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_SAMPLES = 1000  # per class


def load_harder_humor_data():
    """
    Load jokes and match with non-jokes that are similar in style.
    Uses short conversational/informal text as non-humor to reduce
    the stylistic confound between factual sentences and jokes.
    """
    # Load jokes
    ds = load_from_disk(str(PROJECT_ROOT / "datasets" / "short_jokes" / "train"))
    jokes = [row["Joke"] for row in ds if 20 < len(row["Joke"]) < 200]
    random.shuffle(jokes)
    jokes = jokes[:N_SAMPLES]

    # Load non-humor from reddit jokes that scored 0 or 1 (unfunny attempts at humor)
    # These are stylistically similar to jokes but not actually funny
    reddit_ds = load_from_disk(str(PROJECT_ROOT / "datasets" / "one_million_reddit_jokes" / "train"))

    # Low-scoring reddit "jokes" (attempted humor that failed)
    # These share the setup-punchline structure but aren't funny
    low_score_jokes = []
    high_score_jokes = []
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
            low_score_jokes.append(text)
        elif score >= 50:
            high_score_jokes.append(text)

    random.shuffle(low_score_jokes)
    random.shuffle(high_score_jokes)

    print(f"Available: {len(jokes)} short jokes, {len(low_score_jokes)} low-score reddit, "
          f"{len(high_score_jokes)} high-score reddit")

    return jokes, low_score_jokes, high_score_jokes


def load_sentiment_data():
    """Load SST-2 from HuggingFace for a proper sentiment comparison."""
    try:
        ds = load_dataset("stanfordnlp/sst2", split="train")
        texts_pos = [row["sentence"] for row in ds if row["label"] == 1][:N_SAMPLES]
        texts_neg = [row["sentence"] for row in ds if row["label"] == 0][:N_SAMPLES]
        print(f"SST-2: {len(texts_pos)} positive, {len(texts_neg)} negative")
        return texts_pos, texts_neg
    except Exception as e:
        print(f"SST-2 load failed: {e}, using built-in sentiment data")
        return None, None


def extract_activations(model, tokenizer, texts, batch_size=64, max_length=128):
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
    """Probe with PCA dimensionality reduction at various ranks."""
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_acts)
    test_s = scaler.transform(test_acts)

    results = []

    # Full rank
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(train_s, train_labels)
    full_acc = accuracy_score(test_labels, lr.predict(test_s))

    for rank in ranks:
        if rank >= min(train_s.shape):
            continue
        pca = PCA(n_components=rank, random_state=SEED)
        tr = pca.fit_transform(train_s)
        te = pca.transform(test_s)
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(tr, train_labels)
        acc = accuracy_score(test_labels, lr.predict(te))
        f1 = f1_score(test_labels, lr.predict(te))
        results.append({"rank": rank, "accuracy": float(acc), "f1": float(f1)})

    results.append({"rank": train_s.shape[1], "accuracy": float(full_acc), "f1": float(full_acc)})
    return results


def mean_diff_accuracy(train_acts, train_labels, test_acts, test_labels):
    """Single direction (rank-1) mean-diff probe."""
    tl = np.array(train_labels)
    pos_mean = train_acts[tl == 1].mean(axis=0)
    neg_mean = train_acts[tl == 0].mean(axis=0)
    direction = pos_mean - neg_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    train_proj = train_acts @ direction
    test_proj = test_acts @ direction
    threshold = np.median(train_proj)
    preds = (test_proj > threshold).astype(int)
    return float(accuracy_score(test_labels, preds))


def run_hard_experiments():
    """Run probing experiments with harder non-humor data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("EXPERIMENT: Hard Dataset + Sentiment Comparison")
    print("=" * 60)

    # Load data
    jokes, low_score, high_score = load_harder_humor_data()

    # Task 1: Jokes vs Low-Score Reddit (hard: both are attempts at humor)
    n = min(len(jokes), len(low_score), N_SAMPLES)
    task1_texts = jokes[:n] + low_score[:n]
    task1_labels = [1]*n + [0]*n

    # Task 2: High-Score Reddit vs Low-Score Reddit (humor quality)
    n2 = min(len(high_score), len(low_score), N_SAMPLES)
    task2_texts = high_score[:n2] + low_score[:n2]
    task2_labels = [1]*n2 + [0]*n2

    # Shuffle each task
    for task_texts, task_labels in [(task1_texts, task1_labels), (task2_texts, task2_labels)]:
        combined = list(zip(task_texts, task_labels))
        random.shuffle(combined)
        for i, (t, l) in enumerate(combined):
            task_texts[i] = t
            task_labels[i] = l

    # Task 3: Sentiment (SST-2)
    sst_pos, sst_neg = load_sentiment_data()

    # Load model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(DEVICE)
    model.eval()

    results = {}

    # ---- Task 1: Short Jokes vs Low-Score Reddit ----
    print(f"\n--- Task 1: Short Jokes vs Low-Score Reddit (n={n} per class) ---")
    split = int(n * 0.7)
    t1_train_texts = task1_texts[:split*2]
    t1_train_labels = task1_labels[:split*2]
    t1_test_texts = task1_texts[split*2:]
    t1_test_labels = task1_labels[split*2:]

    t1_train_acts = extract_activations(model, tokenizer, t1_train_texts)
    t1_test_acts = extract_activations(model, tokenizer, t1_test_texts)

    task1_results = {"probe_by_layer": [], "mean_diff_by_layer": []}
    for layer in range(model.config.num_hidden_layers + 1):
        probes = probe_at_ranks(t1_train_acts[layer], t1_train_labels,
                                t1_test_acts[layer], t1_test_labels)
        md_acc = mean_diff_accuracy(t1_train_acts[layer], t1_train_labels,
                                    t1_test_acts[layer], t1_test_labels)
        task1_results["probe_by_layer"].append({"layer": layer, "probes": probes, "mean_diff_acc": md_acc})
        best = max(probes, key=lambda x: x["accuracy"])
        print(f"  Layer {layer:2d}: md_acc={md_acc:.3f}, r1_acc={probes[0]['accuracy']:.3f}, "
              f"best_rank={best['rank']}(acc={best['accuracy']:.3f})")

    results["task1_jokes_vs_lowscore"] = task1_results

    # ---- Task 2: High-Score vs Low-Score Reddit ----
    print(f"\n--- Task 2: High-Score vs Low-Score Reddit (n={n2} per class) ---")
    split2 = int(n2 * 0.7)
    t2_train_texts = task2_texts[:split2*2]
    t2_train_labels = task2_labels[:split2*2]
    t2_test_texts = task2_texts[split2*2:]
    t2_test_labels = task2_labels[split2*2:]

    t2_train_acts = extract_activations(model, tokenizer, t2_train_texts)
    t2_test_acts = extract_activations(model, tokenizer, t2_test_texts)

    task2_results = {"probe_by_layer": [], "mean_diff_by_layer": []}
    for layer in range(model.config.num_hidden_layers + 1):
        probes = probe_at_ranks(t2_train_acts[layer], t2_train_labels,
                                t2_test_acts[layer], t2_test_labels)
        md_acc = mean_diff_accuracy(t2_train_acts[layer], t2_train_labels,
                                    t2_test_acts[layer], t2_test_labels)
        task2_results["probe_by_layer"].append({"layer": layer, "probes": probes, "mean_diff_acc": md_acc})
        best = max(probes, key=lambda x: x["accuracy"])
        print(f"  Layer {layer:2d}: md_acc={md_acc:.3f}, r1_acc={probes[0]['accuracy']:.3f}, "
              f"best_rank={best['rank']}(acc={best['accuracy']:.3f})")

    results["task2_highscore_vs_lowscore"] = task2_results

    # ---- Task 3: Sentiment (SST-2) ----
    if sst_pos and sst_neg:
        n3 = min(len(sst_pos), len(sst_neg), N_SAMPLES)
        sst_texts = sst_pos[:n3] + sst_neg[:n3]
        sst_labels = [1]*n3 + [0]*n3
        combined = list(zip(sst_texts, sst_labels))
        random.shuffle(combined)
        sst_texts, sst_labels = zip(*combined)
        sst_texts, sst_labels = list(sst_texts), list(sst_labels)

        split3 = int(n3 * 0.7)
        s_train_texts = sst_texts[:split3*2]
        s_train_labels = sst_labels[:split3*2]
        s_test_texts = sst_texts[split3*2:]
        s_test_labels = sst_labels[split3*2:]

        print(f"\n--- Task 3: SST-2 Sentiment (n={n3} per class) ---")
        s_train_acts = extract_activations(model, tokenizer, s_train_texts)
        s_test_acts = extract_activations(model, tokenizer, s_test_texts)

        task3_results = {"probe_by_layer": [], "mean_diff_by_layer": []}
        for layer in range(model.config.num_hidden_layers + 1):
            probes = probe_at_ranks(s_train_acts[layer], s_train_labels,
                                    s_test_acts[layer], s_test_labels)
            md_acc = mean_diff_accuracy(s_train_acts[layer], s_train_labels,
                                        s_test_acts[layer], s_test_labels)
            task3_results["probe_by_layer"].append({"layer": layer, "probes": probes, "mean_diff_acc": md_acc})
            best = max(probes, key=lambda x: x["accuracy"])
            print(f"  Layer {layer:2d}: md_acc={md_acc:.3f}, r1_acc={probes[0]['accuracy']:.3f}, "
                  f"best_rank={best['rank']}(acc={best['accuracy']:.3f})")

        results["task3_sentiment_sst2"] = task3_results

    # Save
    output_path = PROJECT_ROOT / "results" / "hard_dataset_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    run_hard_experiments()
