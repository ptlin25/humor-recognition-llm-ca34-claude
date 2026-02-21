"""
Experiment 3: LoRA fine-tuning at varying ranks for humor detection.

Fine-tunes Gemma-3-1b with LoRA adapters at different ranks to measure
the minimum rank needed for humor classification.
"""
import json
import sys
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        # Gemma tokenizer doesn't produce token_type_ids; supply zeros so the
        # model's forward() signature check doesn't raise during training.
        if "token_type_ids" not in self.encodings:
            self.encodings["token_type_ids"] = torch.zeros_like(
                self.encodings["input_ids"]
            )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels": self.labels[idx],
        }


class LoRALinear(nn.Module):
    """Simple LoRA adapter for a linear layer."""
    def __init__(self, original_linear, rank, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA matrices — match dtype of the wrapped layer so bfloat16 models work
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype))
        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def apply_lora(model, rank, target_modules=["q_proj", "v_proj"]):
    """Apply LoRA to specified modules in the model (Gemma uses Linear layers)."""
    lora_params = []
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Gemma uses standard Linear layers
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                lora_layer = LoRALinear(module, rank).to(next(module.parameters()).device)
                setattr(parent, child_name, lora_layer)
                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    return lora_params


def train_lora_model(rank, train_dataset, val_dataset, n_epochs=3, lr=1e-3, batch_size=32):
    """Train a Gemma-3-1b model with LoRA at a specific rank."""
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        "google/gemma-3-1b-pt", num_labels=2, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(DEVICE)

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classification head
    for p in model.score.parameters():
        p.requires_grad = True

    # Apply LoRA
    if rank > 0:
        lora_params = apply_lora(model, rank)
        optimizer = torch.optim.AdamW(
            list(model.score.parameters()) + lora_params, lr=lr
        )
    else:
        # rank=0 means only train the classification head (linear probe baseline)
        optimizer = torch.optim.AdamW(model.score.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Count trainable params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    best_val_acc = 0
    history = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    del model
    torch.cuda.empty_cache()

    return {
        "rank": rank,
        "n_trainable": n_trainable,
        "n_total": n_total,
        "best_val_acc": float(best_val_acc),
        "final_val_acc": float(val_acc),
        "final_val_f1": float(val_f1),
        "history": history,
    }


def run_lora_experiment():
    """Run LoRA fine-tuning at varying ranks."""
    print("=" * 60)
    print("EXPERIMENT 3: LoRA Fine-tuning at Varying Ranks")
    print("=" * 60)

    # Load data
    data_path = PROJECT_ROOT / "results" / "prepared_data.json"
    with open(data_path) as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextClassificationDataset(
        data["train"]["texts"], data["train"]["labels"], tokenizer
    )
    val_dataset = TextClassificationDataset(
        data["val"]["texts"], data["val"]["labels"], tokenizer
    )
    test_dataset = TextClassificationDataset(
        data["test"]["texts"], data["test"]["labels"], tokenizer
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Test ranks: 0 (head-only), 1, 2, 4, 8, 16, 32
    ranks_to_test = [0, 1, 2, 4, 8, 16, 32]
    all_results = []

    for rank in ranks_to_test:
        print(f"\n--- Training with LoRA rank={rank} ---")
        t0 = time.time()
        result = train_lora_model(
            rank, train_dataset, val_dataset,
            n_epochs=5, lr=2e-4, batch_size=4
        )
        elapsed = time.time() - t0
        result["time_seconds"] = elapsed
        all_results.append(result)
        print(f"  Rank {rank}: val_acc={result['best_val_acc']:.3f}, "
              f"trainable_params={result['n_trainable']:,}, time={elapsed:.1f}s")

    # Save results
    output_path = PROJECT_ROOT / "results" / "lora_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_lora_experiment()
