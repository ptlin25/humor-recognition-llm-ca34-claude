"""
Experiment 4: Sentiment comparison - measure rank of sentiment representations
using the same methodology, for calibration against humor.
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Simple sentiment dataset (positive/negative sentences)
POSITIVE_SENTENCES = [
    "This movie was absolutely wonderful and I loved every minute of it.",
    "The food at this restaurant is incredible and the service is excellent.",
    "I had a fantastic time at the party last night.",
    "This book is a masterpiece of modern literature.",
    "The sunset was breathtakingly beautiful today.",
    "I am so happy with my new car, it drives like a dream.",
    "The concert was amazing, the band played perfectly.",
    "This is the best vacation I have ever been on.",
    "The teacher was inspiring and made learning fun.",
    "I love spending time with my family on weekends.",
    "The garden looks absolutely gorgeous this spring.",
    "This new phone has an incredible camera and great battery life.",
    "The team played brilliantly and deserved to win.",
    "I am thrilled with the results of the project.",
    "The weather today is perfect for a walk in the park.",
    "This coffee shop makes the most delicious lattes.",
    "The hotel room was spacious, clean, and comfortable.",
    "I really enjoyed the presentation, it was very informative.",
    "The baby smiled at me and it made my entire day.",
    "This neighborhood is charming and full of wonderful shops.",
    "The movie had a heartwarming story that brought tears of joy.",
    "I feel grateful for all the wonderful people in my life.",
    "The dessert was absolutely divine, perfectly crafted.",
    "The new park downtown is a beautiful addition to the city.",
    "I had the most relaxing spa day and feel completely refreshed.",
    "The documentary was fascinating and extremely well produced.",
    "My colleagues are supportive and make work enjoyable.",
    "The hiking trail had stunning views at every turn.",
    "This painting captures the beauty of nature perfectly.",
    "The comedy show had us laughing from start to finish.",
    "I appreciate the kind words and thoughtful gesture.",
    "The children played happily in the yard all afternoon.",
    "This recipe turned out even better than I expected.",
    "The autumn leaves create a spectacular display of colors.",
    "I am proud of what we accomplished together as a team.",
    "The music was soothing and helped me relax after a long day.",
    "This city has the most wonderful cultural attractions.",
    "The puppy is adorable and brings so much joy to our home.",
    "I am excited about the new opportunities ahead.",
    "The birthday celebration was full of love and laughter.",
]

NEGATIVE_SENTENCES = [
    "This movie was terrible and a complete waste of time.",
    "The food at this restaurant was cold and the service was awful.",
    "I had a miserable experience at the hotel last night.",
    "This book is poorly written and utterly boring.",
    "The weather has been dreadful and depressing all week.",
    "I am so disappointed with my purchase, it broke immediately.",
    "The concert was a disaster, the sound quality was horrible.",
    "This is the worst vacation I have ever experienced.",
    "The teacher was rude and unhelpful to the students.",
    "I hate dealing with the traffic during rush hour.",
    "The garden is overgrown and full of weeds and dead plants.",
    "This phone has a terrible battery and keeps crashing.",
    "The team played poorly and deserved to lose the match.",
    "I am frustrated with the lack of progress on this project.",
    "The weather today is miserable with cold rain all day.",
    "This coffee shop serves the worst coffee I have ever tasted.",
    "The hotel room was dirty, cramped, and uncomfortable.",
    "The presentation was boring and a waste of everyone's time.",
    "The noise from construction kept me awake all night.",
    "This neighborhood is unsafe and poorly maintained.",
    "The movie had a depressing ending that left me feeling empty.",
    "I feel stressed and overwhelmed by all the problems.",
    "The dessert was stale and clearly not fresh.",
    "The old building downtown is an eyesore that should be demolished.",
    "I had the worst customer service experience of my life.",
    "The documentary was biased and full of misinformation.",
    "My colleagues are difficult to work with and unsupportive.",
    "The hiking trail was dangerous and poorly maintained.",
    "This painting is ugly and a waste of gallery space.",
    "The comedy show was not funny at all and painfully awkward.",
    "I resent the unfair criticism and harsh judgment.",
    "The children were misbehaving and causing problems all day.",
    "This recipe was a total failure and tasted awful.",
    "The winter storms have caused significant damage and misery.",
    "I am ashamed of the poor results we produced as a team.",
    "The music was annoying and gave me a terrible headache.",
    "This city has the worst public transportation system.",
    "The dog destroyed the furniture and made a huge mess.",
    "I am dreading the difficult challenges that lie ahead.",
    "The funeral was the saddest day I have ever experienced.",
]


def extract_activations(model, tokenizer, texts, batch_size=32, max_length=128):
    """Extract hidden states from all layers."""
    model.eval()
    n_layers = model.config.num_hidden_layers + 1
    all_activations = {layer: [] for layer in range(n_layers)}

    for i in range(0, len(texts), batch_size):
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


def run_sentiment_comparison():
    """Run the same PCA and probing analysis on sentiment for comparison."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("EXPERIMENT 4: Sentiment Comparison")
    print("=" * 60)

    # Prepare sentiment data
    n_per_class = len(POSITIVE_SENTENCES)  # 40 each
    all_texts = POSITIVE_SENTENCES + NEGATIVE_SENTENCES
    all_labels = [1] * n_per_class + [0] * n_per_class  # 1=positive, 0=negative

    # Shuffle
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    all_texts, all_labels = list(all_texts), list(all_labels)

    # Split: 60/20/20
    n = len(all_texts)
    n_test = n // 5
    n_val = n // 5
    n_train = n - n_test - n_val

    train_texts, train_labels = all_texts[:n_train], all_labels[:n_train]
    test_texts, test_labels = all_texts[n_train:n_train+n_val], all_labels[n_train:n_train+n_val]

    print(f"Sentiment data: train={len(train_texts)}, test={len(test_texts)}")

    # Load model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(DEVICE)

    # Extract activations
    train_acts = extract_activations(model, tokenizer, train_texts)
    test_acts = extract_activations(model, tokenizer, test_texts)

    # Per-layer analysis
    train_labels_np = np.array(train_labels)
    test_labels_np = np.array(test_labels)

    results = {
        "task": "sentiment",
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "probe_results": [],
        "mean_diff_results": [],
    }

    ranks_to_test = [1, 2, 4, 8, 16, 32, 64]

    for layer_idx in range(model.config.num_hidden_layers + 1):
        layer_train = train_acts[layer_idx]
        layer_test = test_acts[layer_idx]

        # Linear probing at varying ranks
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(layer_train)
        test_scaled = scaler.transform(layer_test)

        probe_by_rank = []
        for rank in ranks_to_test:
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
            probe_by_rank.append({"rank": rank, "accuracy": float(acc), "f1": float(f1)})

        # Full rank
        lr_full = LogisticRegression(max_iter=1000, random_state=SEED)
        lr_full.fit(train_scaled, train_labels)
        full_acc = accuracy_score(test_labels, lr_full.predict(test_scaled))
        probe_by_rank.append({"rank": train_scaled.shape[1], "accuracy": float(full_acc), "f1": float(full_acc)})

        results["probe_results"].append({
            "layer": layer_idx,
            "results_by_rank": probe_by_rank,
        })

        # Mean difference probe
        pos_mean = layer_train[train_labels_np == 1].mean(axis=0)
        neg_mean = layer_train[train_labels_np == 0].mean(axis=0)
        direction = pos_mean - neg_mean
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        train_proj = layer_train @ direction
        test_proj = layer_test @ direction
        threshold = np.median(train_proj)
        preds = (test_proj > threshold).astype(int)
        md_acc = accuracy_score(test_labels, preds)
        results["mean_diff_results"].append({
            "layer": layer_idx,
            "accuracy": float(md_acc),
        })

        best = max(probe_by_rank, key=lambda x: x["accuracy"])
        print(f"  Layer {layer_idx:2d}: mean_diff_acc={md_acc:.3f}, "
              f"best_rank={best['rank']} (acc={best['accuracy']:.3f})")

    # Save
    output_path = PROJECT_ROOT / "results" / "sentiment_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    results = run_sentiment_comparison()
