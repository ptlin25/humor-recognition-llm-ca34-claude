"""
HaHackathon (SemEval 2021 Task 7) dataset loader.

Expects three CSV files at datasets/hahackathon/{train,dev,test}.csv with columns:
    text            - the text sample
    is_humor        - binary label (0 or 1)
    humor_rating    - float 0-5 (average annotator humor rating)
    offense_rating  - float 0-5 (average annotator offense rating)

Run standalone to verify data loads correctly:
    python src/data_hahackathon.py
"""
import json
import random
from pathlib import Path

import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "hahackathon"


def _load_csvs():
    """Load and combine the three CSV files. Returns a dict of DataFrames."""
    dfs = {}
    for split in ("train", "dev", "test"):
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"HaHackathon file not found: {path}\n"
                f"Upload train.csv, dev.csv, and test.csv to {DATA_DIR}"
            )
        dfs[split] = pd.read_csv(path)
    return dfs


def _balance(texts, labels, seed=SEED):
    """Undersample majority class to produce 50/50 balance."""
    rng = np.random.default_rng(seed)
    texts = np.array(texts)
    labels = np.array(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    pos_idx = rng.choice(pos_idx, n, replace=False)
    neg_idx = rng.choice(neg_idx, n, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return texts[idx].tolist(), labels[idx].tolist()


def load_hahackathon(binary=False, rating_split=False, seed=SEED):
    """
    Load HaHackathon dataset with train/test splits.

    Args:
        binary: If True, use is_humor (0/1) as labels, balanced 50/50.
                Train pool = train.csv + dev.csv. Test = test.csv.
        rating_split: If True, use humor_rating as labels:
                      1 if humor_rating >= 3.5, 0 if humor_rating <= 1.5.
                      Texts in the middle range are dropped.
                      (Hard task analog — funny vs clearly unfunny.)
        seed: Random seed for balancing.

    Returns:
        dict with keys "train" and "test", each containing:
            {"texts": List[str], "labels": List[int]}
    """
    dfs = _load_csvs()

    if not binary and not rating_split:
        raise ValueError("Specify binary=True or rating_split=True.")

    if binary:
        train_df = pd.concat([dfs["train"], dfs["dev"]], ignore_index=True)
        test_df = dfs["test"]

        train_texts = train_df["text"].astype(str).tolist()
        train_labels = train_df["is_humor"].astype(int).tolist()
        test_texts = test_df["text"].astype(str).tolist()
        test_labels = test_df["is_humor"].astype(int).tolist()

        train_texts, train_labels = _balance(train_texts, train_labels, seed)
        test_texts, test_labels = _balance(test_texts, test_labels, seed)

    elif rating_split:
        full_df = pd.concat([dfs["train"], dfs["dev"], dfs["test"]], ignore_index=True)

        high_mask = full_df["humor_rating"] >= 3.5
        low_mask = full_df["humor_rating"] <= 1.5

        high_texts = full_df.loc[high_mask, "text"].astype(str).tolist()
        low_texts = full_df.loc[low_mask, "text"].astype(str).tolist()

        rng = random.Random(seed)
        rng.shuffle(high_texts)
        rng.shuffle(low_texts)

        n = min(len(high_texts), len(low_texts))
        n_train = int(n * 0.7)

        train_texts = high_texts[:n_train] + low_texts[:n_train]
        train_labels = [1] * n_train + [0] * n_train
        test_texts = high_texts[n_train:n] + low_texts[n_train:n]
        test_labels = [1] * (n - n_train) + [0] * (n - n_train)

        # Shuffle
        combined_train = list(zip(train_texts, train_labels))
        rng.shuffle(combined_train)
        train_texts, train_labels = zip(*combined_train) if combined_train else ([], [])
        train_texts, train_labels = list(train_texts), list(train_labels)

        combined_test = list(zip(test_texts, test_labels))
        rng.shuffle(combined_test)
        test_texts, test_labels = zip(*combined_test) if combined_test else ([], [])
        test_texts, test_labels = list(test_texts), list(test_labels)
    else:
        raise ValueError("Either binary=True or rating_split=True must be set.")

    return {
        "train": {"texts": train_texts, "labels": train_labels},
        "test": {"texts": test_texts, "labels": test_labels},
    }


def load_hahackathon_with_ratings():
    """
    Load HaHackathon with continuous humor and offense ratings attached.
    Used for regression probing or correlation analysis.

    Returns:
        dict with "train" and "test", each containing:
            {"texts": List[str], "labels": List[int],
             "humor_ratings": List[float], "offense_ratings": List[float]}
    """
    dfs = _load_csvs()
    train_df = pd.concat([dfs["train"], dfs["dev"]], ignore_index=True)
    test_df = dfs["test"]

    def _extract(df):
        return {
            "texts": df["text"].astype(str).tolist(),
            "labels": df["is_humor"].astype(int).tolist(),
            "humor_ratings": df["humor_rating"].astype(float).tolist(),
            "offense_ratings": df["offense_rating"].astype(float).tolist(),
        }

    return {"train": _extract(train_df), "test": _extract(test_df)}


if __name__ == "__main__":
    print("=" * 60)
    print("HaHackathon dataset verification")
    print("=" * 60)

    # Binary split
    print("\n--- Binary (is_humor) ---")
    data = load_hahackathon(binary=True)
    for split in ("train", "test"):
        texts = data[split]["texts"]
        labels = data[split]["labels"]
        n_pos = sum(labels)
        print(f"  {split}: n={len(texts)}, positive={n_pos} ({100*n_pos/len(texts):.1f}%)")
    print("\n  Train examples:")
    for t, l in zip(data["train"]["texts"][:3], data["train"]["labels"][:3]):
        print(f"    [{l}] {t[:100]}")

    # Rating split
    print("\n--- Rating split (humor_rating ≥3.5 vs ≤1.5) ---")
    try:
        data_r = load_hahackathon(rating_split=True)
        for split in ("train", "test"):
            texts = data_r[split]["texts"]
            labels = data_r[split]["labels"]
            n_pos = sum(labels)
            print(f"  {split}: n={len(texts)}, high_humor={n_pos} ({100*n_pos/len(texts):.1f}%)")
    except Exception as e:
        print(f"  Rating split failed: {e}")

    # Save summary
    output_path = PROJECT_ROOT / "results" / "hahackathon_data.json"
    summary = {
        "binary_train_n": len(data["train"]["texts"]),
        "binary_test_n": len(data["test"]["texts"]),
        "binary_train_pos_frac": sum(data["train"]["labels"]) / len(data["train"]["labels"]),
        "binary_test_pos_frac": sum(data["test"]["labels"]) / len(data["test"]["labels"]),
        "train_examples": data["train"]["texts"][:5],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")
