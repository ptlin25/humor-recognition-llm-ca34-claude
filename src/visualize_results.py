"""
Visualization of all experimental results.
Creates publication-quality plots for REPORT.md.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def plot_singular_value_spectrum():
    """Plot the singular value / explained variance spectrum per layer."""
    with open(RESULTS_DIR / "activation_analysis_results.json") as f:
        data = json.load(f)

    pca_results = data["pca_results"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cumulative explained variance for selected layers
    ax = axes[0, 0]
    layers_to_show = [0, 3, 6, 9, 12] if data["n_layers"] >= 12 else [0, 2, 4, 6, 8]
    for layer_idx in layers_to_show:
        if layer_idx < len(pca_results):
            cum_var = pca_results[layer_idx]["cumulative_variance"][:50]
            ax.plot(range(1, len(cum_var)+1), cum_var, marker=".", markersize=3,
                    label=f"Layer {layer_idx}")
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA: Cumulative Explained Variance by Layer")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 50)
    ax.grid(True, alpha=0.3)

    # 2. Explained variance (top components) - log scale
    ax = axes[0, 1]
    for layer_idx in layers_to_show:
        if layer_idx < len(pca_results):
            var = pca_results[layer_idx]["explained_variance"][:30]
            ax.plot(range(1, len(var)+1), var, marker=".", markersize=3,
                    label=f"Layer {layer_idx}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA: Individual Component Variance (Singular Value Spectrum)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Mean difference norm by layer
    ax = axes[1, 0]
    layers = [r["layer"] for r in pca_results]
    norms = [r["mean_diff_norm"] for r in pca_results]
    ax.bar(layers, norms, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||Mean Humor - Mean Non-Humor||")
    ax.set_title("Humor Signal Strength by Layer")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Effective rank (components for 90% variance) by layer
    ax = axes[1, 1]
    effective_ranks_90 = []
    effective_ranks_95 = []
    for r in pca_results:
        cum = r["cumulative_variance"]
        r90 = next((i+1 for i, v in enumerate(cum) if v >= 0.90), len(cum))
        r95 = next((i+1 for i, v in enumerate(cum) if v >= 0.95), len(cum))
        effective_ranks_90.append(r90)
        effective_ranks_95.append(r95)

    ax.plot(layers, effective_ranks_90, "o-", color="red", label="90% variance")
    ax.plot(layers, effective_ranks_95, "s-", color="orange", label="95% variance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Components")
    ax.set_title("Effective Rank (Components for X% Variance) by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_analysis.png", bbox_inches="tight")
    plt.close()
    print(f"Saved pca_analysis.png")


def plot_probing_results():
    """Plot linear probe accuracy vs rank for each layer."""
    with open(RESULTS_DIR / "activation_analysis_results.json") as f:
        data = json.load(f)

    probe_results = data["probe_results"]
    mean_diff_results = data["mean_diff_results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Accuracy vs rank for selected layers
    ax = axes[0]
    layers_to_show = [0, 3, 6, 9, 12] if data["n_layers"] >= 12 else [0, 2, 4, 6, 8]
    for layer_idx in layers_to_show:
        if layer_idx < len(probe_results):
            results = probe_results[layer_idx]["results_by_rank"]
            ranks = [r["rank"] for r in results]
            accs = [r["accuracy"] for r in results]
            ax.plot(ranks, accs, "o-", label=f"Layer {layer_idx}", markersize=5)

    ax.set_xlabel("Probe Rank (Number of Dimensions)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Humor Classification: Accuracy vs. Probe Rank")
    ax.set_xscale("log", base=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Best accuracy per layer and rank needed for 90% of best
    ax = axes[1]
    best_accs = []
    rank_for_90pct = []
    for pr in probe_results:
        results = pr["results_by_rank"]
        best_acc = max(r["accuracy"] for r in results)
        best_accs.append(best_acc)
        target = best_acc * 0.95  # 95% of best accuracy
        r90 = next((r["rank"] for r in sorted(results, key=lambda x: x["rank"])
                     if r["accuracy"] >= target), results[-1]["rank"])
        rank_for_90pct.append(r90)

    layers = list(range(len(probe_results)))
    ax2 = ax.twinx()
    bars = ax.bar(layers, best_accs, alpha=0.6, color="steelblue", label="Best accuracy")
    line = ax2.plot(layers, rank_for_90pct, "ro-", label="Min rank for 95% of best", markersize=5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best Classification Accuracy", color="steelblue")
    ax2.set_ylabel("Minimum Rank for 95% of Best", color="red")
    ax.set_title("Humor Probing: Best Accuracy & Effective Rank by Layer")
    ax.grid(True, alpha=0.3, axis="y")

    # Combined legend
    bars_label = plt.Rectangle((0, 0), 1, 1, fc="steelblue", alpha=0.6)
    ax.legend([bars_label, line[0]], ["Best accuracy", "Min rank for 95% of best"],
              loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "probing_results.png", bbox_inches="tight")
    plt.close()
    print("Saved probing_results.png")


def plot_lora_results():
    """Plot LoRA performance vs rank."""
    lora_path = RESULTS_DIR / "lora_results.json"
    if not lora_path.exists():
        print("No LoRA results found, skipping")
        return

    with open(lora_path) as f:
        lora_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ranks = [r["rank"] for r in lora_data]
    accs = [r["best_val_acc"] for r in lora_data]
    params = [r["n_trainable"] for r in lora_data]

    # 1. Accuracy vs rank
    ax = axes[0]
    ax.plot(ranks, accs, "o-", color="darkgreen", markersize=8, linewidth=2)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("LoRA: Humor Detection Accuracy vs. Rank")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    if accs:
        ax.axhline(y=max(accs), color="green", linestyle="--", alpha=0.3, label="Best")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy vs trainable params
    ax = axes[1]
    ax.plot(params, accs, "s-", color="purple", markersize=8, linewidth=2)
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("LoRA: Accuracy vs. Trainable Parameters")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    for i, (p, a, r) in enumerate(zip(params, accs, ranks)):
        ax.annotate(f"r={r}", (p, a), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lora_results.png", bbox_inches="tight")
    plt.close()
    print("Saved lora_results.png")


def plot_humor_vs_sentiment():
    """Compare humor and sentiment rank profiles."""
    humor_path = RESULTS_DIR / "activation_analysis_results.json"
    sentiment_path = RESULTS_DIR / "sentiment_comparison_results.json"

    if not humor_path.exists() or not sentiment_path.exists():
        print("Missing results for comparison, skipping")
        return

    with open(humor_path) as f:
        humor_data = json.load(f)
    with open(sentiment_path) as f:
        sentiment_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Mean diff accuracy comparison by layer
    ax = axes[0]
    humor_md = [r["accuracy"] for r in humor_data["mean_diff_results"]]
    sentiment_md = [r["accuracy"] for r in sentiment_data["mean_diff_results"]]
    layers = list(range(len(humor_md)))
    ax.plot(layers, humor_md, "o-", color="blue", label="Humor (mean diff)", markersize=5)
    sent_layers = list(range(len(sentiment_md)))
    ax.plot(sent_layers, sentiment_md, "s-", color="red", label="Sentiment (mean diff)", markersize=5)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank-1 Probe Accuracy")
    ax.set_title("Rank-1 (Mean Diff) Probe: Humor vs Sentiment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy at rank 1, 4, 8 by layer for humor, with sentiment rank 1 overlay
    ax = axes[1]
    # Humor at different ranks - pick the best layer
    best_layer_humor = max(range(len(humor_data["probe_results"])),
                           key=lambda i: max(r["accuracy"] for r in humor_data["probe_results"][i]["results_by_rank"]))
    humor_probe = humor_data["probe_results"][best_layer_humor]["results_by_rank"]
    h_ranks = [r["rank"] for r in humor_probe if r["rank"] <= 64]
    h_accs = [r["accuracy"] for r in humor_probe if r["rank"] <= 64]

    best_layer_sent = max(range(len(sentiment_data["probe_results"])),
                          key=lambda i: max(r["accuracy"] for r in sentiment_data["probe_results"][i]["results_by_rank"]))
    sent_probe = sentiment_data["probe_results"][best_layer_sent]["results_by_rank"]
    s_ranks = [r["rank"] for r in sent_probe if r["rank"] <= 64]
    s_accs = [r["accuracy"] for r in sent_probe if r["rank"] <= 64]

    ax.plot(h_ranks, h_accs, "o-", color="blue", label=f"Humor (layer {best_layer_humor})", markersize=6)
    ax.plot(s_ranks, s_accs, "s-", color="red", label=f"Sentiment (layer {best_layer_sent})", markersize=6)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Probe Rank")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Accuracy vs. Rank: Humor vs. Sentiment (Best Layer)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "humor_vs_sentiment.png", bbox_inches="tight")
    plt.close()
    print("Saved humor_vs_sentiment.png")


def plot_summary_figure():
    """Create a single summary figure for the paper."""
    humor_path = RESULTS_DIR / "activation_analysis_results.json"
    lora_path = RESULTS_DIR / "lora_results.json"
    sentiment_path = RESULTS_DIR / "sentiment_comparison_results.json"

    with open(humor_path) as f:
        humor_data = json.load(f)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Singular value spectrum (best layer)
    ax = fig.add_subplot(gs[0, 0])
    pca_results = humor_data["pca_results"]
    best_layer = max(range(len(pca_results)),
                     key=lambda i: pca_results[i]["mean_diff_norm"])
    var = pca_results[best_layer]["explained_variance"][:30]
    ax.bar(range(1, len(var)+1), var, color="steelblue", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance")
    ax.set_title(f"A) Singular Value Spectrum (Layer {best_layer})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Cumulative variance
    ax = fig.add_subplot(gs[0, 1])
    cum = pca_results[best_layer]["cumulative_variance"][:50]
    ax.plot(range(1, len(cum)+1), cum, "b-", linewidth=2)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="90%")
    ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.7, label="95%")
    r90 = next((i+1 for i, v in enumerate(cum) if v >= 0.90), len(cum))
    r95 = next((i+1 for i, v in enumerate(cum) if v >= 0.95), len(cum))
    ax.axvline(x=r90, color="red", linestyle=":", alpha=0.5)
    ax.axvline(x=r95, color="orange", linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"B) Cumulative Variance (Layer {best_layer})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Probe accuracy vs rank
    ax = fig.add_subplot(gs[0, 2])
    probe = humor_data["probe_results"][best_layer]["results_by_rank"]
    ranks_p = [r["rank"] for r in probe]
    accs_p = [r["accuracy"] for r in probe]
    ax.plot(ranks_p, accs_p, "o-", color="darkgreen", markersize=6, linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Probe Rank")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"C) Probe Accuracy vs. Rank (Layer {best_layer})")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    # Panel D: LoRA results
    ax = fig.add_subplot(gs[1, 0])
    if lora_path.exists():
        with open(lora_path) as f:
            lora_data = json.load(f)
        l_ranks = [r["rank"] for r in lora_data]
        l_accs = [r["best_val_acc"] for r in lora_data]
        ax.plot(l_ranks, l_accs, "o-", color="purple", markersize=6, linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("D) LoRA: Accuracy vs. Rank")
    ax.grid(True, alpha=0.3)

    # Panel E: Humor vs Sentiment comparison
    ax = fig.add_subplot(gs[1, 1])
    humor_md = [r["accuracy"] for r in humor_data["mean_diff_results"]]
    layers = list(range(len(humor_md)))
    ax.plot(layers, humor_md, "o-", color="blue", label="Humor", markersize=5)
    if sentiment_path.exists():
        with open(sentiment_path) as f:
            sent_data = json.load(f)
        sent_md = [r["accuracy"] for r in sent_data["mean_diff_results"]]
        ax.plot(range(len(sent_md)), sent_md, "s-", color="red", label="Sentiment", markersize=5)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank-1 Probe Accuracy")
    ax.set_title("E) Rank-1 Probe: Humor vs. Sentiment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel F: Effective rank by layer
    ax = fig.add_subplot(gs[1, 2])
    eff_ranks_90 = []
    eff_ranks_95 = []
    for r in pca_results:
        cum = r["cumulative_variance"]
        eff_ranks_90.append(next((i+1 for i, v in enumerate(cum) if v >= 0.90), len(cum)))
        eff_ranks_95.append(next((i+1 for i, v in enumerate(cum) if v >= 0.95), len(cum)))
    ax.plot(layers, eff_ranks_90, "o-", color="red", label="90% variance", markersize=5)
    ax.plot(layers, eff_ranks_95, "s-", color="orange", label="95% variance", markersize=5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("F) Effective Rank by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(PLOTS_DIR / "summary_figure.png", bbox_inches="tight")
    plt.close()
    print("Saved summary_figure.png")


def generate_all_plots():
    """Generate all visualization plots."""
    print("=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)

    plot_singular_value_spectrum()
    plot_probing_results()
    plot_lora_results()
    plot_humor_vs_sentiment()
    plot_summary_figure()

    print("\nAll plots saved to results/plots/")


if __name__ == "__main__":
    generate_all_plots()
