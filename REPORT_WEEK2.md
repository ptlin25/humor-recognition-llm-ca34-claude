# Week 2: Scaling, Non-Linearity, and Steering Humor in Gemma 3

## 1. Executive Summary

Building on week 1's finding that humor style is rank-1 separable in GPT-2 but humor quality is not, we extend our analysis to Gemma 3 (4B and 12B) with four new experiments: (A) layer-by-layer accuracy curves, (C) non-linear MLP probing, (D) scaling to 12B parameters, and (E) activation steering.

**Key findings**:
- Humor is **linearly encoded** in Gemma 3 — an MLP probe is no better than logistic regression at every layer for both 4B and 12B.
- Humor signal **peaks at mid-layers** (layer 18 of 35 for Gemma 4B) then degrades, unlike style signal which is flat from the start.
- **Scale does not help**: Gemma 3 12B (48 layers, 3840 hidden dim) achieves 65.8% on HaHackathon — essentially identical to 4B's 65.6%.
- **Activation steering produces coherent but not noticeably funnier outputs**. The humor direction exists geometrically but is too weak to redirect generation. Evaluation of steering quality is ongoing.

---

## 2. Motivation

The TA's feedback on week 1 identified five areas to address:
1. Clarify which layer results are reported for
2. Add a random label baseline (partner's contribution — see section 3)
3. Test non-linear (MLP) probing
4. Try a larger model (~10B)
5. Begin activation steering

This report covers items 1, 3, 4, and 5. Item 2 was handled by a partner.

---

## 3. Background: Random Label Baseline (Partner's Contribution)

To validate that our 65.6% HaHackathon accuracy reflects real signal and not dataset artifacts, a partner ran probing with shuffled labels on the same texts:

| Condition | Mean-Diff Acc | Full-Rank LR Acc |
|-----------|--------------|-----------------|
| Real labels | 0.509 | 0.660 |
| Shuffled labels (same humor texts) | 0.501 | 0.545 |
| Random labels (non-humor texts) | 0.667 | 0.600 |

The shuffled-label condition (54.5%) is the correct control — same texts, randomized labels. The 65.6% vs 54.5% gap (~11pp) confirms the real humor signal is genuine and not an artifact of the dataset. The non-humor random baseline (60–67%) is elevated due to latent stylistic separation between text corpora, not humor signal.

---

## 4. Experiment A: Layer Curves (Figure 8)

### What We Did

Plotted full-rank probe accuracy at every layer (0–34) for all three tasks using the existing week 1 results. No new Modal run was needed — the data already existed.

### Results

| Task | Best Layer | Best Accuracy | Layer 0 Accuracy |
|------|-----------|--------------|-----------------|
| Easy (jokes vs factual) | ~8 | 75.3% | 57.9% |
| Hard (high vs low Reddit) | 9 | 55.2% | ~50% |
| HaHackathon (binary humor) | **18** | **65.6%** | 57.9% |

**Easy task**: Flat across all 35 layers at ~75%. Style information is captured from the very first layer and does not improve. The model immediately encodes joke register.

**Hard task**: Flat at near-chance (~55%) across all layers. No layer contains meaningful quality signal.

**HaHackathon**: Rises from 57.9% (layer 0) → 65.6% (layer 18) → falls back to ~62% at layer 34. Humor signal builds gradually through the middle layers, suggesting higher-level processing is involved relative to pure style.

The layer 18 peak for HaHackathon motivated our choice of steering target.

### Figure 8
See `results/plots/figure8_layer_curves_google-gemma-3-4b-it.png`.

---

## 5. Experiment C: Non-Linear (MLP) Probe

### What We Did

Added an MLPClassifier probe (1 hidden layer, 256 units, early stopping) alongside the existing logistic regression probe at every layer, for all three tasks. Run on Gemma 3 4B via Modal (A10G).

### Setup

```
StandardScaler → MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True)
```
No PCA. Operates on full hidden dim (2560 for 4B).

### Results at HaHackathon Best Layer (Layer 18)

| Probe | Accuracy |
|-------|---------|
| Linear (LR) | 65.6% |
| Non-linear (MLP) | 65.3% |
| Gap (MLP − LR) | **−0.3pp** |

MLP is marginally *worse* than linear at layer 18 and at every other layer. The pattern holds across all 35 layers — MLP never beats LR by more than ~1%.

### Interpretation

Humor is **linearly encoded** in Gemma 3's activation space. There is no nonlinear structure that a shallow MLP can exploit. This has two implications:
1. The representation is geometrically simple — humor information lives along a single direction.
2. We can use the LR coefficient vector as a valid steering direction (no need to extract gradients from the MLP).

---

## 6. Experiment D: Gemma 3 12B Probing

### Setup

- Model: `google/gemma-3-12b-it` (48 layers, hidden dim 3840, ~24GB in bfloat16)
- GPU: A100-40GB (A10G is too tight for 12B inference)
- Tasks: easy, hard, hahackathon with MLP probe enabled
- Estimated cost: ~$8–10

### Results

| Task | Best Layer | Best LR Acc | Best MLP Acc |
|------|-----------|------------|-------------|
| Easy | 8 | 75.1% | ~74% |
| Hard | 13 | 54.7% | ~54% |
| HaHackathon | **28** | **65.8%** | **65.6%** |

**Versus Gemma 4B:**

| Model | HaHackathon Best | Hard Best | Easy Best |
|-------|-----------------|-----------|-----------|
| Gemma 3 4B | 65.6% (layer 18) | 55.2% (layer 9) | 75.3% |
| Gemma 3 12B | 65.8% (layer 28) | 54.7% (layer 13) | 75.1% |
| Δ | **+0.2pp** | −0.5pp | −0.2pp |

### Interpretation

Scale does not help. A 3× increase in parameters gains essentially nothing on humor signal. The ceiling appears to be around 65–66% for this task with linear probing, regardless of model size.

The best layer shifts deeper in the 12B model (layer 28 vs 18), proportionally similar (28/48 ≈ 58% vs 18/35 ≈ 51%), suggesting the "humor layer" scales with depth roughly proportionally.

MLP ≤ LR at every layer for 12B as well — the linearity finding generalizes across scale.

---

## 7. Experiment E: Activation Steering

### Setup

- Model: `google/gemma-3-4b-it`
- Steering layer: 18 (0-indexed hook on `model.model.language_model.layers[17]`)
- Direction: LR coefficient vector back-projected to original activation space, normalized to unit vector
- Alphas: [−20, −10, 0, 10, 20, 30]
- Prompts: 5 neutral prompts (e.g., "The thing about Mondays is", "Scientists have recently discovered that")
- Completions: 3 per (prompt, alpha) = 90 total
- Generation: temperature=0.8, top_p=0.9, max_new_tokens=80

**Why LR direction, not mean-diff**: Mean-diff accuracy at layer 18 is only 54.3% (barely above chance), even though full-rank LR achieves 65.6%. The humor signal at layer 18 is not along the centroid-difference axis — it requires the full high-dimensional structure that LR learns.

**Note on architecture**: Gemma 3 4B is a multimodal model. The transformer layers are at `model.model.language_model.layers`, not `model.model.layers` as in Gemma 2. This required a dynamic layer-finding fix.

### Qualitative Results

Outputs are coherent across all alpha values. Example for "The thing about Mondays is":

**alpha=−20**: "...that they're just…Monday. It's a feeling, not a specific event. They're the default, the baseline..."

**alpha=0**: "...that they just never seem to get easier, do they? I know I'm not the only one feeling this way..."

**alpha=30**: "...that they always feel like the beginning of a long, slow decline. It's the day after the weekend, the day after the fun... But hey, at least there's coffee."

The outputs shift slightly in tone at high alpha but do not become noticeably funnier. The steering direction does not produce dramatic qualitative differences.

### Evaluation Status

Quantitative evaluation of funniness is pending. The TA suggested using a large LLM (GPT-4 level) for pairwise judgment, comparing steered outputs against an unsteered best-of-N baseline. We are determining whether to use an external API for this evaluation before proceeding.

---

## 8. Overall Story

Across all four experiments, a consistent picture emerges:

**What Gemma 3 represents about humor:**
- *Style* (joke vs factual text): strongly and linearly encoded from layer 0, ~75% accuracy
- *Quality* (funny vs unfunny): essentially not encoded, ~55% at any layer
- *Binary humor* (HaHackathon): weakly encoded, peaks at ~65.6% at layer 18, linearly

**What doesn't help:**
- Non-linear probes: MLP = LR, humor is geometrically simple
- More parameters: 12B ≈ 4B, the ceiling is in the training signal not the model capacity
- Steering: outputs stay coherent but the humor direction is too weak to redirect generation

**The core finding**: LLMs learn to recognize the *form* of humor (joke-shaped text) but not the *content* (what makes something actually funny). This holds across model sizes and is not a limitation of linear probing — it reflects what the model has internalized.

---

## 9. Next Steps

1. **Steering evaluation**: Pairwise LLM judge comparison of steered vs. unsteered outputs, with best-of-N baseline as the TA suggested
2. **Visualization**: Update figures to include week 2 MLP vs LR comparison and 12B vs 4B comparison
3. **Blog post**: Write up the full week 2 narrative for the class blog

---

## 10. Figures

- `results/plots/figure6_new_model_google-gemma-3-4b-it.png` — GPT-2 vs Gemma 4B rank-1 accuracy by normalized layer
- `results/plots/figure7_cross_transfer_google-gemma-3-4b-it.png` — Cross-dataset transfer heatmap
- `results/plots/figure8_layer_curves_google-gemma-3-4b-it.png` — Full-rank accuracy by absolute layer (new)

## 11. References

1. Meaney, J.A., et al. (2021). SemEval 2021 Task 7: HaHackathon. arXiv:2105.03402
2. Tigges, C., et al. (2023). Linear Representations of Sentiment in Large Language Models. arXiv:2310.15154
3. Zou, A., et al. (2023). Representation Engineering. arXiv:2310.01405
4. Engels, J., et al. (2024). Not All Language Model Features Are Linear. arXiv:2405.14860
