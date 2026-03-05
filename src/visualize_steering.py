"""
Visualization for activation steering evaluation.

Plots humor gain (GPT judge win rate vs. BoN) against average perplexity
for each steering alpha, with the base model and best-of-N as reference points.

Requires:
  - results/{slug}_steering.json       (from experiment_steering.py)
  - results/{slug}_steering_judge.json (from evaluate_steering_judge.py)

Usage:
    python src/visualize_steering.py --model google/gemma-3-4b-it
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "font.family": "serif",
})


def model_slug(model_id):
    return model_id.replace("/", "-")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_steering(model_id):
    slug = model_slug(model_id)
    steering_path = RESULTS_DIR / f"{slug}_steering.json"
    judge_path = RESULTS_DIR / f"{slug}_steering_judge.json"

    for p in (steering_path, judge_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    steering = load_json(steering_path)
    judge = load_json(judge_path)

    alphas = [str(a) for a in steering["alphas"]]
    win_rates = judge["win_rate_by_alpha"]   # alpha_key -> float (steered win rate vs BoN)
    mean_ppls = steering["mean_perplexity_by_alpha"]  # alpha_key -> float

    bon_ppl = steering["best_of_n"]["mean_perplexity"]
    base_ppl = mean_ppls.get("0", None)

    fig, ax = plt.subplots(figsize=(6, 5))

    # --- Plot steered points (non-zero alphas) ---
    for alpha_key in alphas:
        alpha_int = int(alpha_key)
        if alpha_int == 0:
            continue
        if alpha_key not in win_rates or alpha_key not in mean_ppls:
            continue

        ppl = mean_ppls[alpha_key]
        wr = win_rates[alpha_key]
        color = "#2166ac" if alpha_int > 0 else "#d6604d"
        ax.scatter(ppl, wr, color=color, s=80, zorder=3)
        ax.annotate(
            f"α={alpha_key}",
            xy=(ppl, wr),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    # Connect steered points in alpha order
    sorted_alphas = sorted(
        [k for k in alphas if k in win_rates and k in mean_ppls and k != "0"],
        key=lambda x: int(x)
    )
    xs = [mean_ppls[k] for k in sorted_alphas]
    ys = [win_rates[k] for k in sorted_alphas]
    ax.plot(xs, ys, color="gray", linewidth=0.8, linestyle="--", zorder=2)

    # --- Best-of-N reference (anchored at win_rate=0.5 by definition) ---
    ax.scatter(bon_ppl, 0.5, marker="*", s=200, color="darkorange",
               zorder=4, label=f"Best-of-N (N={steering['best_of_n']['n']})")
    ax.annotate("BoN", xy=(bon_ppl, 0.5), xytext=(5, 3),
                textcoords="offset points", fontsize=8, color="darkorange")

    # --- Base model (alpha=0): use judge win rate if available, else anchor at 0.5 ---
    if base_ppl is not None:
        base_wr = win_rates.get("0", 0.5)
        ax.scatter(base_ppl, base_wr, marker="D", s=80, color="gray",
                   zorder=4, label="Base (α=0)")
        ax.annotate("Base", xy=(base_ppl, base_wr), xytext=(5, 3),
                    textcoords="offset points", fontsize=8, color="gray")

    # --- Reference line at win_rate=0.5 ---
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", zorder=1)

    # --- Quadrant annotation ---
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(
        xlim[0] + 0.02 * (xlim[1] - xlim[0]),
        ylim[1] - 0.04 * (ylim[1] - ylim[0]),
        "← funnier, more coherent",
        fontsize=8,
        color="green",
        va="top",
        style="italic",
    )

    # --- Proxy legend entries for alpha color coding ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markersize=8, label="Steered toward humor (α > 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6604d",
               markersize=8, label="Steered away from humor (α < 0)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="darkorange",
               markersize=10, label=f"Best-of-N (N={steering['best_of_n']['n']})"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markersize=8, label="Base (α=0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.9)

    ax.set_xlabel("Mean Perplexity (↓ more coherent)")
    ax.set_ylabel("GPT Judge Win Rate vs. BoN\n(↑ funnier)")
    ax.set_title(f"Humor Gain vs. Coherence — {model_id.split('/')[-1]}")
    ax.grid(True, alpha=0.3)

    out_path = PLOTS_DIR / f"figure9_steering_{slug}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize steering humor gain vs perplexity")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID")
    args = parser.parse_args()

    plot_steering(model_id=args.model)
