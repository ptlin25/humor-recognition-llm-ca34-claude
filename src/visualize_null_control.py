"""
Visualization for null-control experiment results.

Generates a 4-panel figure from <model>_null_control.json:
  A) Full-rank probe accuracy by layer (real vs shuffled vs non-humor-random)
  B) Mean-diff accuracy by layer (same three conditions)
  C) Rank sweep at the best layer — accuracy vs probe rank (log scale)
  D) Rank x Layer heatmap for the real condition

Usage:
    python src/visualize_null_control.py
    python src/visualize_null_control.py --model google/gemma-3-4b-it
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
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "font.family": "serif",
})

CONDITION_STYLE = {
    "real":          {"color": "crimson",    "marker": "o", "label": "Real labels (HaHackathon)"},
    "shuffled":      {"color": "steelblue",  "marker": "s", "label": "Shuffled labels (humor text)"},
    "non_humor_rnd": {"color": "forestgreen","marker": "^", "label": "Random labels (non-humor text)"},
}


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "-")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def get_mean_diff_accs(probe_by_layer: list) -> list[float]:
    return [layer["mean_diff_acc"] for layer in probe_by_layer]


def get_full_rank_accs(probe_by_layer: list) -> list[float]:
    """Accuracy for the largest rank (full-rank probe) at each layer."""
    return [layer["probes"][-1]["accuracy"] for layer in probe_by_layer]


def get_rank_sweep_at_layer(probe_by_layer: list, layer_idx: int) -> tuple[list, list]:
    """Return (ranks, accuracies) for the given layer index."""
    probes = probe_by_layer[layer_idx]["probes"]
    ranks = [p["rank"] for p in probes]
    accs  = [p["accuracy"] for p in probes]
    return ranks, accs


def best_full_rank_layer(probe_by_layer: list) -> int:
    full_rank_accs = get_full_rank_accs(probe_by_layer)
    return int(np.argmax(full_rank_accs))


def build_rank_layer_matrix(probe_by_layer: list) -> tuple[np.ndarray, list, list]:
    """
    Returns (matrix, layers, ranks) where matrix[i, j] = accuracy at layer i, rank j.
    """
    layers = [l["layer"] for l in probe_by_layer]
    ranks  = [p["rank"] for p in probe_by_layer[0]["probes"]]
    matrix = np.array([
        [p["accuracy"] for p in layer["probes"]]
        for layer in probe_by_layer
    ])
    return matrix, layers, ranks


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_null_control(model_id: str, data: dict) -> Path:
    conditions = data["conditions"]
    n_layers = data["n_layers"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    (ax_full, ax_md), (ax_rank, ax_heat) = axes

    layers_x = list(range(n_layers + 1))  # 0..n_layers inclusive

    # ------------------------------------------------------------------ A --
    # Full-rank accuracy by layer
    for cond_key, style in CONDITION_STYLE.items():
        if cond_key not in conditions:
            continue
        accs = get_full_rank_accs(conditions[cond_key]["probe_by_layer"])
        ax_full.plot(layers_x, accs,
                     color=style["color"], marker=style["marker"],
                     label=style["label"], markersize=3, linewidth=1.5)

    ax_full.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Chance (0.5)")
    ax_full.set_xlabel("Layer")
    ax_full.set_ylabel("Accuracy")
    ax_full.set_title("A) Full-Rank Probe Accuracy by Layer")
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ B --
    # Mean-diff accuracy by layer
    for cond_key, style in CONDITION_STYLE.items():
        if cond_key not in conditions:
            continue
        accs = get_mean_diff_accs(conditions[cond_key]["probe_by_layer"])
        ax_md.plot(layers_x, accs,
                   color=style["color"], marker=style["marker"],
                   label=style["label"], markersize=3, linewidth=1.5)

    ax_md.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Chance (0.5)")
    ax_md.set_xlabel("Layer")
    ax_md.set_ylabel("Accuracy")
    ax_md.set_title("B) Mean-Diff (Rank-1) Accuracy by Layer")
    ax_md.legend()
    ax_md.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ C --
    # Rank sweep pinned to real condition's best layer for all conditions
    real_best_layer = best_full_rank_layer(conditions["real"]["probe_by_layer"])
    for cond_key, style in CONDITION_STYLE.items():
        if cond_key not in conditions:
            continue
        pbl = conditions[cond_key]["probe_by_layer"]
        ranks, accs = get_rank_sweep_at_layer(pbl, real_best_layer)

        hidden = data["hidden_size"]
        x_labels = [str(r) if r != hidden else f"full\n({hidden})" for r in ranks]
        x_pos = list(range(len(ranks)))

        ax_rank.plot(x_pos, accs,
                     color=style["color"], marker=style["marker"],
                     label=style["label"],
                     markersize=5, linewidth=1.8)

    ax_rank.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Chance (0.5)")
    ax_rank.set_xticks(x_pos)
    ax_rank.set_xticklabels(x_labels, fontsize=9)
    ax_rank.set_xlabel("Probe Rank")
    ax_rank.set_ylabel("Accuracy")
    ax_rank.set_title(f"C) Rank Sweep at Layer {real_best_layer} (real's best)\n(How many dims encode humor?)")
    ax_rank.legend(fontsize=8)
    ax_rank.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ D --
    # Rank x Layer heatmap (real condition only)
    if "real" in conditions:
        matrix, layer_ticks, rank_vals = build_rank_layer_matrix(
            conditions["real"]["probe_by_layer"]
        )
        hidden = data["hidden_size"]
        rank_labels = [str(r) if r != hidden else f"full\n({hidden})" for r in rank_vals]

        im = ax_heat.imshow(
            matrix, aspect="auto", origin="upper",
            cmap="RdYlGn", vmin=0.5, vmax=matrix.max()
        )
        fig.colorbar(im, ax=ax_heat, label="Accuracy")

        ax_heat.set_xticks(range(len(rank_vals)))
        ax_heat.set_xticklabels(rank_labels, fontsize=8)
        ax_heat.set_xlabel("Probe Rank")
        ax_heat.set_ylabel("Layer")
        ax_heat.set_title("D) Real Labels: Accuracy Heatmap\n(Layer × Rank)")

        # Mark best cell
        best_row, best_col = np.unravel_index(np.argmax(matrix), matrix.shape)
        ax_heat.plot(best_col, best_row, "w*", markersize=12,
                     label=f"Best: {matrix[best_row, best_col]:.3f}\n(layer {layer_ticks[best_row]}, rank {rank_vals[best_col]})")
        ax_heat.legend(fontsize=8, loc="upper left")

    # ------------------------------------------------------------------ layout --
    model_label = model_id.replace("/", " / ")
    fig.suptitle(
        f"Null Control Experiment — {model_label}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    out_path = PLOTS_DIR / f"null_control_{model_slug(model_id)}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_id: str) -> None:
    slug = model_slug(model_id)
    result_path = RESULTS_DIR / f"{slug}_null_control.json"

    if not result_path.exists():
        print(f"Error: {result_path} not found.")
        print(f"  Run experiment_null_control.py first, or check the model ID.")
        return

    data = load_json(result_path)
    print(f"Loaded results for {data['model']} ({data['n_layers']} layers, "
          f"hidden={data['hidden_size']})")
    for cond, info in data["conditions"].items():
        n_train = info.get("n_train", "?")
        n_test  = info.get("n_test", "?")
        print(f"  {cond}: train={n_train}, test={n_test}")

    plot_null_control(model_id, data)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize null-control experiment results")
    parser.add_argument(
        "--model", type=str, default="google/gemma-3-4b-it",
        help="HuggingFace model ID (must match slug in results/)"
    )
    args = parser.parse_args()
    main(args.model)
