"""
Visualization for new model probing and cross-transfer results.

Generates:
  Figure 6: Rank-1 (mean-diff) accuracy by normalized layer position,
            comparing GPT-2 baseline vs new model on all three tasks.
  Figure 7: Cross-transfer heatmap — best rank-1 accuracy at the best layer
            for each of the four transfer directions.

Usage:
    python src/visualize_new.py --model google/gemma-3-4b-it
    python src/visualize_new.py --model google/gemma-3-4b-it --mock
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "font.family": "serif",
})


def model_slug(model_id):
    return model_id.replace("/", "-")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_md_accs(probe_by_layer):
    """Extract mean-diff accuracy list from probe_by_layer structure."""
    return [layer["mean_diff_acc"] for layer in probe_by_layer]


def get_best_probe_accs(probe_by_layer):
    """Extract best-rank probe accuracy (any rank) per layer."""
    return [max(p["accuracy"] for p in layer["probes"]) for layer in probe_by_layer]


def normalize_layers(n):
    """Return normalized layer positions 0..1."""
    return np.linspace(0, 1, n)


# ---------------------------------------------------------------------------
# Figure 6: GPT-2 vs new model rank-1 accuracy by normalized layer position
# ---------------------------------------------------------------------------

def plot_figure6(model_id, new_results, baseline_acts, baseline_hard):
    """
    Three-panel figure comparing GPT-2 (baseline) vs new model on:
      Panel A: easy task (jokes vs factual)
      Panel B: hard task (high vs low Reddit)
      Panel C: HaHackathon binary
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel A: Easy task ---
    ax = axes[0]
    # GPT-2 baseline (from activation_analysis_results.json)
    gpt2_md = [r["accuracy"] for r in baseline_acts["mean_diff_results"]]
    gpt2_x = normalize_layers(len(gpt2_md))
    ax.plot(gpt2_x, gpt2_md, "o-", color="steelblue", label="GPT-2 (124M)",
            markersize=4, linewidth=1.8)

    # New model
    if "easy" in new_results.get("tasks", {}):
        nm_md = get_md_accs(new_results["tasks"]["easy"]["probe_by_layer"])
        nm_x = normalize_layers(len(nm_md))
        ax.plot(nm_x, nm_md, "s-", color="crimson", label=f"{model_id.split('/')[-1]}",
                markersize=4, linewidth=1.8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("Rank-1 (Mean Diff) Accuracy")
    ax.set_title("A) Easy Task\n(Jokes vs. Factual Text)")
    ax.set_ylim(0.45, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel B: Hard task ---
    ax = axes[1]
    # GPT-2 baseline (from hard_dataset_results.json)
    if "task2_highscore_vs_lowscore" in baseline_hard:
        gpt2_hard_md = [
            layer["mean_diff_acc"]
            for layer in baseline_hard["task2_highscore_vs_lowscore"]["probe_by_layer"]
        ]
        gpt2_x = normalize_layers(len(gpt2_hard_md))
        ax.plot(gpt2_x, gpt2_hard_md, "o-", color="steelblue", label="GPT-2 (124M)",
                markersize=4, linewidth=1.8)

    if "hard" in new_results.get("tasks", {}):
        nm_hard_md = get_md_accs(new_results["tasks"]["hard"]["probe_by_layer"])
        nm_x = normalize_layers(len(nm_hard_md))
        ax.plot(nm_x, nm_hard_md, "s-", color="crimson",
                label=f"{model_id.split('/')[-1]}", markersize=4, linewidth=1.8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("Rank-1 (Mean Diff) Accuracy")
    ax.set_title("B) Hard Task\n(High vs. Low Score Reddit)")
    ax.set_ylim(0.4, 0.85)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel C: HaHackathon ---
    ax = axes[2]
    # No GPT-2 baseline for HaHackathon (new dataset)
    if "hahackathon" in new_results.get("tasks", {}):
        nm_haha_md = get_md_accs(new_results["tasks"]["hahackathon"]["probe_by_layer"])
        nm_x = normalize_layers(len(nm_haha_md))
        ax.plot(nm_x, nm_haha_md, "s-", color="crimson",
                label=f"{model_id.split('/')[-1]}", markersize=4, linewidth=1.8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("Rank-1 (Mean Diff) Accuracy")
    ax.set_title("C) HaHackathon\n(Binary Humor)")
    ax.set_ylim(0.4, 0.85)
    ax.legend()
    ax.grid(True, alpha=0.3)

    model_label = model_id.replace("/", " / ")
    plt.suptitle(
        f"Figure 6: GPT-2 vs {model_label} — Rank-1 Probe Accuracy",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / f"figure6_new_model_{model_slug(model_id)}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 7: Cross-transfer heatmap
# ---------------------------------------------------------------------------

TRANSFER_LABELS = {
    "easy_to_haha": "Easy → HaHackathon",
    "hard_to_haha": "Hard → HaHackathon",
    "haha_to_easy": "HaHackathon → Easy",
    "haha_to_hard": "HaHackathon → Hard",
}

EXPECTED_DIRECTION = {
    "easy_to_haha": "near-chance\n(register ≠ quality)",
    "hard_to_haha": "slightly above chance\n(quality, limited generalization)",
    "haha_to_easy": "high\n(humor → style easy to detect)",
    "haha_to_hard": "low\n(domain mismatch)",
}


def plot_figure7(model_id, transfer_results):
    """
    Two-panel figure:
      Panel A: Best rank-1 accuracy per transfer direction (bar chart)
      Panel B: Full-rank accuracy per transfer direction (bar chart)
    Also annotates expected direction.
    """
    transfers = transfer_results.get("transfers", {})
    direction_ids = ["easy_to_haha", "hard_to_haha", "haha_to_easy", "haha_to_hard"]

    # Extract best-layer metrics for each direction
    rank1_best = {}
    fullrank_best = {}
    for tid in direction_ids:
        if tid not in transfers:
            rank1_best[tid] = 0.0
            fullrank_best[tid] = 0.0
            continue
        layers = transfers[tid]
        rank1_best[tid] = max(l["mean_diff_acc"] for l in layers)
        fullrank_best[tid] = max(
            max(p["accuracy"] for p in l["probes"])
            for l in layers
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    labels = [TRANSFER_LABELS.get(tid, tid) for tid in direction_ids]
    x = np.arange(len(direction_ids))

    for ax, values, title, metric_label in [
        (axes[0], rank1_best, "A) Rank-1 (Mean Diff) Transfer Accuracy", "Rank-1 Accuracy"),
        (axes[1], fullrank_best, "B) Best-Rank Transfer Accuracy", "Best-Rank Accuracy"),
    ]:
        accs = [values[tid] for tid in direction_ids]
        bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=10, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    model_label = model_id.replace("/", " / ")
    plt.suptitle(
        f"Figure 7: Cross-Dataset Transfer Probing — {model_label}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / f"figure7_cross_transfer_{model_slug(model_id)}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_mock_results(model_id, n_layers=34):
    """Generate plausible mock results for testing visualization code."""
    rng = np.random.default_rng(42)

    def mock_probe_by_layer(n_layers, base_acc, noise=0.05):
        layers = []
        for l in range(n_layers + 1):
            acc = float(np.clip(base_acc + rng.normal(0, noise), 0, 1))
            probes = [{"rank": r, "accuracy": float(np.clip(acc + rng.normal(0, 0.01), 0, 1)),
                       "f1": float(np.clip(acc + rng.normal(0, 0.01), 0, 1))}
                      for r in [1, 2, 4, 8, 16, 32, 64]]
            probes.append({"rank": 2560, "accuracy": float(np.clip(acc + 0.02, 0, 1)),
                           "f1": float(np.clip(acc + 0.02, 0, 1))})
            layers.append({"layer": l, "mean_diff_acc": acc, "probes": probes})
        return layers

    return {
        "model": model_id,
        "n_layers": n_layers,
        "hidden_size": 2560,
        "mock": True,
        "tasks": {
            "easy": {"n_train": 2400, "n_test": 800,
                     "probe_by_layer": mock_probe_by_layer(n_layers, 0.97)},
            "hard": {"n_train": 1400, "n_test": 600,
                     "probe_by_layer": mock_probe_by_layer(n_layers, 0.60)},
            "hahackathon": {"n_train": 3000, "n_test": 1000,
                            "probe_by_layer": mock_probe_by_layer(n_layers, 0.68)},
        },
    }


def make_mock_transfer_results(model_id, n_layers=34):
    rng = np.random.default_rng(42)

    def mock_transfer(base_md, base_full, n_layers):
        layers = []
        for l in range(n_layers + 1):
            md = float(np.clip(base_md + rng.normal(0, 0.03), 0, 1))
            probes = [{"rank": r, "accuracy": float(np.clip(base_full + rng.normal(0, 0.02), 0, 1)),
                       "f1": 0.0}
                      for r in [1, 2, 4, 8, 16, 32, 64, 2560]]
            layers.append({"layer": l, "mean_diff_acc": md, "probes": probes})
        return layers

    return {
        "model": model_id,
        "n_layers": n_layers,
        "mock": True,
        "transfers": {
            "easy_to_haha": mock_transfer(0.52, 0.55, n_layers),
            "hard_to_haha": mock_transfer(0.58, 0.60, n_layers),
            "haha_to_easy": mock_transfer(0.91, 0.94, n_layers),
            "haha_to_hard": mock_transfer(0.54, 0.57, n_layers),
        },
    }


def main(model_id, mock=False):
    slug = model_slug(model_id)

    # Load baseline GPT-2 results
    baseline_acts_path = RESULTS_DIR / "activation_analysis_results.json"
    baseline_hard_path = RESULTS_DIR / "hard_dataset_results.json"

    if not baseline_acts_path.exists() or not baseline_hard_path.exists():
        print("Warning: baseline result files not found, skipping GPT-2 comparison lines")
        baseline_acts = {"mean_diff_results": [], "probe_results": []}
        baseline_hard = {}
    else:
        baseline_acts = load_json(baseline_acts_path)
        baseline_hard = load_json(baseline_hard_path)

    # Load new model results
    new_results_path = RESULTS_DIR / f"{slug}_results.json"
    transfer_results_path = RESULTS_DIR / f"{slug}_cross_transfer.json"

    if mock:
        new_results = make_mock_results(model_id)
        transfer_results = make_mock_transfer_results(model_id)
    else:
        if not new_results_path.exists():
            print(f"Error: {new_results_path} not found. Run experiment_new_model.py first.")
            return
        new_results = load_json(new_results_path)

        if not transfer_results_path.exists():
            print(f"Warning: {transfer_results_path} not found. Skipping Figure 7.")
            transfer_results = None
        else:
            transfer_results = load_json(transfer_results_path)

    print(f"Generating figures for {model_id}...")
    plot_figure6(model_id, new_results, baseline_acts, baseline_hard)

    if transfer_results is not None:
        plot_figure7(model_id, transfer_results)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize new model and cross-transfer results")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID (must match the slug in results/)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock data for testing visualization code")
    args = parser.parse_args()
    main(args.model, args.mock)
