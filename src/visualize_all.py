"""
Comprehensive visualization of all experimental results.
"""
import json
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
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'font.family': 'serif',
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_figure1_main_results():
    """Figure 1: Main results - PCA spectrum and probing (easy task, GPT-2)."""
    data = load_json(RESULTS_DIR / "activation_analysis_results.json")
    pca_results = data["pca_results"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel A: Singular value spectrum (best layer by mean diff)
    best_layer = max(range(len(pca_results)),
                     key=lambda i: pca_results[i]["mean_diff_norm"])
    ax = axes[0, 0]
    var = pca_results[best_layer]["explained_variance"][:30]
    colors = ["#1f77b4" if i < 3 else "#aec7e8" for i in range(len(var))]
    ax.bar(range(1, len(var)+1), var, color=colors)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"A) Singular Value Spectrum\n(GPT-2, Layer {best_layer})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Cumulative variance across layers
    ax = axes[0, 1]
    for layer_idx in [0, 4, 8, 12]:
        if layer_idx < len(pca_results):
            cum = pca_results[layer_idx]["cumulative_variance"][:50]
            ax.plot(range(1, len(cum)+1), cum, label=f"Layer {layer_idx}", linewidth=1.5)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("B) Cumulative Variance by Layer")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 50)
    ax.grid(True, alpha=0.3)

    # Panel C: Mean-diff (rank-1) accuracy across layers
    ax = axes[0, 2]
    md_accs = [r["accuracy"] for r in data["mean_diff_results"]]
    layers = list(range(len(md_accs)))
    ax.plot(layers, md_accs, "o-", color="darkblue", markersize=6, linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("C) Rank-1 (Mean Diff) Probe by Layer\n(Jokes vs. Factual Text)")
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel D: Probe accuracy vs rank (best layer)
    ax = axes[1, 0]
    probe = data["probe_results"][best_layer]["results_by_rank"]
    ranks = [r["rank"] for r in probe]
    accs = [r["accuracy"] for r in probe]
    ax.plot(ranks, accs, "o-", color="darkgreen", markersize=6, linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Probe Rank")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"D) Accuracy vs. Rank (Layer {best_layer})")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    # Panel E: Effective rank by layer
    ax = axes[1, 1]
    eff_90, eff_95 = [], []
    for r in pca_results:
        cum = r["cumulative_variance"]
        eff_90.append(next((i+1 for i, v in enumerate(cum) if v >= 0.90), len(cum)))
        eff_95.append(next((i+1 for i, v in enumerate(cum) if v >= 0.95), len(cum)))
    ax.plot(layers, eff_90, "o-", color="red", label="90% var", markersize=5)
    ax.plot(layers, eff_95, "s-", color="orange", label="95% var", markersize=5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("E) Effective Rank by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel F: LoRA results
    ax = axes[1, 2]
    lora = load_json(RESULTS_DIR / "lora_results.json")
    l_ranks = [r["rank"] for r in lora]
    l_accs = [r["best_val_acc"] for r in lora]
    ax.plot(l_ranks, l_accs, "o-", color="purple", markersize=8, linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("F) LoRA: Accuracy vs. Rank")
    ax.grid(True, alpha=0.3)
    for r, a in zip(l_ranks, l_accs):
        ax.annotate(f"{a:.3f}", (r, a), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)

    plt.suptitle("Figure 1: Humor Recognition Rank Analysis (GPT-2 Small, Jokes vs. Factual Text)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figure1_main_results.png", bbox_inches="tight")
    plt.close()
    print("Saved figure1_main_results.png")


def plot_figure2_hard_tasks():
    """Figure 2: Hard task results - controlling for confounds."""
    hard = load_json(RESULTS_DIR / "hard_dataset_results.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    tasks = [
        ("task1_jokes_vs_lowscore", "A) Short Jokes vs.\nLow-Score Reddit", "steelblue"),
        ("task2_highscore_vs_lowscore", "B) High-Score vs.\nLow-Score Reddit", "coral"),
        ("task3_sentiment_sst2", "C) Sentiment (SST-2)", "forestgreen"),
    ]

    for idx, (key, title, color) in enumerate(tasks):
        if key not in hard:
            continue
        ax = axes[idx]
        task_data = hard[key]
        layers = []
        md_accs = []
        r1_accs = []
        r4_accs = []
        r16_accs = []
        best_accs = []

        for lr in task_data["probe_by_layer"]:
            layers.append(lr["layer"])
            md_accs.append(lr["mean_diff_acc"])
            probes = lr["probes"]
            r1 = next((p["accuracy"] for p in probes if p["rank"] == 1), None)
            r4 = next((p["accuracy"] for p in probes if p["rank"] == 4), None)
            r16 = next((p["accuracy"] for p in probes if p["rank"] == 16), None)
            r1_accs.append(r1)
            r4_accs.append(r4)
            r16_accs.append(r16)
            best_accs.append(max(p["accuracy"] for p in probes))

        ax.plot(layers, md_accs, "o-", color=color, label="Mean Diff (r=1)", markersize=4, linewidth=1.5)
        ax.plot(layers, best_accs, "s--", color="black", label="Best rank", markersize=4, linewidth=1, alpha=0.7)
        if r4_accs[0] is not None:
            ax.plot(layers, r4_accs, "^:", color="gray", label="Rank 4", markersize=3, linewidth=1, alpha=0.7)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0.4, 0.85)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 2: Humor Detection with Controlled Confounds (GPT-2 Small)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figure2_hard_tasks.png", bbox_inches="tight")
    plt.close()
    print("Saved figure2_hard_tasks.png")


def plot_figure3_comparison():
    """Figure 3: Side-by-side comparison of easy vs hard tasks, humor vs sentiment."""
    easy = load_json(RESULTS_DIR / "activation_analysis_results.json")
    hard = load_json(RESULTS_DIR / "hard_dataset_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Rank-1 accuracy comparison across tasks
    ax = axes[0]
    easy_md = [r["accuracy"] for r in easy["mean_diff_results"]]
    layers = list(range(len(easy_md)))
    ax.plot(layers, easy_md, "o-", color="blue", label="Jokes vs. Factual (easy)", markersize=5, linewidth=2)

    for key, label, color, ls in [
        ("task1_jokes_vs_lowscore", "Jokes vs. Low-Score (hard)", "red", "-"),
        ("task2_highscore_vs_lowscore", "High vs. Low Score", "orange", "--"),
        ("task3_sentiment_sst2", "Sentiment (SST-2)", "green", "-."),
    ]:
        if key in hard:
            task_md = [lr["mean_diff_acc"] for lr in hard[key]["probe_by_layer"]]
            ax.plot(range(len(task_md)), task_md, f"s{ls}", color=color, label=label,
                    markersize=4, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Diff (Rank-1) Accuracy")
    ax.set_title("A) Rank-1 Probe Across Tasks")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Best probe accuracy comparison
    ax = axes[1]
    easy_best = [max(r["accuracy"] for r in pr["results_by_rank"]) for pr in easy["probe_results"]]
    ax.plot(layers, easy_best, "o-", color="blue", label="Jokes vs. Factual (easy)", markersize=5, linewidth=2)

    for key, label, color, ls in [
        ("task1_jokes_vs_lowscore", "Jokes vs. Low-Score (hard)", "red", "-"),
        ("task2_highscore_vs_lowscore", "High vs. Low Score", "orange", "--"),
        ("task3_sentiment_sst2", "Sentiment (SST-2)", "green", "-."),
    ]:
        if key in hard:
            task_best = [max(p["accuracy"] for p in lr["probes"]) for lr in hard[key]["probe_by_layer"]]
            ax.plot(range(len(task_best)), task_best, f"s{ls}", color=color, label=label,
                    markersize=4, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best Probe Accuracy (Any Rank)")
    ax.set_title("B) Best Linear Probe Across Tasks")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 3: Humor Recognition Difficulty Depends on Confound Control",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figure3_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved figure3_comparison.png")


def plot_figure4_pythia():
    """Figure 4: Cross-model comparison (GPT-2 vs Pythia-410M)."""
    gpt2 = load_json(RESULTS_DIR / "activation_analysis_results.json")
    pythia = load_json(RESULTS_DIR / "pythia_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Mean diff accuracy
    ax = axes[0]
    gpt2_md = [r["accuracy"] for r in gpt2["mean_diff_results"]]
    pythia_md = [r["mean_diff_acc"] for r in pythia["probe_results"]]
    gpt2_layers = np.linspace(0, 1, len(gpt2_md))
    pythia_layers = np.linspace(0, 1, len(pythia_md))
    ax.plot(gpt2_layers, gpt2_md, "o-", color="blue", label=f"GPT-2 (124M, {len(gpt2_md)-1}L)", markersize=5)
    ax.plot(pythia_layers, pythia_md, "s-", color="red", label=f"Pythia-410M ({len(pythia_md)-1}L)", markersize=4)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("Rank-1 Probe Accuracy")
    ax.set_title("A) Rank-1 Probe: GPT-2 vs Pythia-410M")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Best probe accuracy
    ax = axes[1]
    gpt2_best = [max(r["accuracy"] for r in pr["results_by_rank"]) for pr in gpt2["probe_results"]]
    pythia_best = [max(p["accuracy"] for p in pr["probes"]) for pr in pythia["probe_results"]]
    ax.plot(gpt2_layers, gpt2_best, "o-", color="blue", label="GPT-2 (124M)", markersize=5)
    ax.plot(pythia_layers, pythia_best, "s-", color="red", label="Pythia-410M", markersize=4)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("Best Probe Accuracy")
    ax.set_title("B) Best Probe: GPT-2 vs Pythia-410M")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 4: Cross-Model Consistency (Jokes vs. Factual Text)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figure4_cross_model.png", bbox_inches="tight")
    plt.close()
    print("Saved figure4_cross_model.png")


def plot_figure5_lora_detail():
    """Figure 5: Detailed LoRA results with accuracy vs params."""
    lora = load_json(RESULTS_DIR / "lora_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ranks = [r["rank"] for r in lora]
    accs = [r["best_val_acc"] for r in lora]
    params = [r["n_trainable"] for r in lora]

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ranks)))
    ax.bar(range(len(ranks)), accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("A) LoRA Accuracy by Rank")
    ax.set_ylim(0.95, 1.005)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (r, a) in enumerate(zip(ranks, accs)):
        ax.text(i, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)

    ax = axes[1]
    ax.plot(params, accs, "o-", color="purple", markersize=8, linewidth=2)
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("B) Accuracy vs. Trainable Parameters")
    ax.set_xscale("log")
    ax.set_ylim(0.95, 1.005)
    ax.grid(True, alpha=0.3)
    for p, a, r in zip(params, accs, ranks):
        ax.annotate(f"r={r}", (p, a), textcoords="offset points", xytext=(8, -5), fontsize=8)

    plt.suptitle("Figure 5: LoRA Fine-tuning for Humor Detection (GPT-2, Jokes vs. Factual)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figure5_lora_detail.png", bbox_inches="tight")
    plt.close()
    print("Saved figure5_lora_detail.png")


if __name__ == "__main__":
    print("Generating all figures...")
    plot_figure1_main_results()
    plot_figure2_hard_tasks()
    plot_figure3_comparison()
    plot_figure4_pythia()
    plot_figure5_lora_detail()
    print("\nAll figures saved to results/plots/")
