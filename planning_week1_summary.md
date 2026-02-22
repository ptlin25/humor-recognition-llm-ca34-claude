# Week 1 Extension Plan — Team Summary

## Goal

The GPT-2 / Pythia-410M baseline is complete and locked. This week we extend the project in three directions:

1. **New model probing**: Re-run the same probing experiments on Gemma 3 (an instruction-tuned model 10–30x larger than the baseline). Core question: does the rank-1 / style-capture finding hold in a more capable model?

2. **HaHackathon as a second dataset**: The baseline only used the Short Jokes (Kaggle) dataset. HaHackathon provides human-annotated humor and offensiveness ratings on a different text domain, making it a proper second benchmark.

3. **Cross-dataset transfer probing**: Train probes on one dataset, test zero-shot on the other. This tells us whether the "humor direction" is domain-general or dataset-specific.

---

## What We Are NOT Touching

- All files in `src/experiment_*.py` (the baseline experiments)
- All `results/*.json` files
- All `results/plots/figure[1–5]_*.png`
- `REPORT.md`

These are the GPT-2 baseline — leave completely unchanged.

---

## Models

| Model | HF ID | GPU | Use |
|-------|-------|-----|-----|
| Gemma 3 1B Instruct | `google/gemma-3-1b-it` | T4 (16 GB) | Fast iteration / first run |
| Gemma 3 4B Instruct | `google/gemma-3-4b-it` | A10G (24 GB) | **Main paper results** |
| Qwen3 4B | `Qwen/Qwen3-4B` *(verify)* | A10G (24 GB) | Optional cross-family check |

**All model inference runs on Modal.** Even the 1B model is too slow on CPU for activation extraction across thousands of texts. Local machines are used only for data verification and code structure testing (mock mode).

---

## Datasets

| Dataset | Location | Status |
|---------|----------|--------|
| Short Jokes (Kaggle) | `datasets/short_jokes/` | ✅ In repo |
| Reddit Jokes | `datasets/one_million_reddit_jokes/` | ✅ In repo |
| HaHackathon | `datasets/hahackathon/train.csv`, `dev.csv`, `test.csv` | ⬆️ Upload manually |

HaHackathon files will be ignored by git (covered by `datasets/.gitignore`).

---

## New Files This Week

```
src/utils.py                     ← Shared code: extract_activations, probe_at_ranks, mean_diff_accuracy
src/data_hahackathon.py          ← Loads the 3 HaHackathon CSVs, returns train/test splits
src/experiment_new_model.py      ← Probes Gemma on easy task, hard task, HaHackathon binary
src/experiment_cross_transfer.py ← Trains probe on D1, tests zero-shot on D2 (and vice versa)
src/visualize_new.py             ← Comparison plots (GPT-2 vs Gemma, cross-transfer heatmap)
modal/run_probing.py             ← Modal app that runs the above on GPU
```

---

## What We Expect to Find

| Experiment | Expected result |
|-----------|----------------|
| Gemma easy task, rank-1 | ≥ 95% — style capture holds even in capable models |
| Gemma hard task, rank-1 | ~60% — humor quality still not linearly separable |
| Transfer: easy probe → HaHackathon | Near chance — the "register direction" doesn't generalize to quality annotations |
| Transfer: HaHackathon probe → easy task | High (≥ 85%) — real humor signal generalizes downward |
| Transfer: HaHackathon probe → hard task | Low — domain mismatch |

If Gemma shows different patterns from GPT-2, that's a positive finding worth reporting. If it replicates, that strengthens the baseline finding's generalizability.

---

## Workflow

### Local (no GPU needed)
```bash
# Verify HaHackathon data loads correctly
python src/data_hahackathon.py

# Test probe + save pipeline with random activations (no model)
python src/experiment_new_model.py --mock --model google/gemma-3-4b-it
python src/experiment_cross_transfer.py --mock --model google/gemma-3-4b-it
python src/visualize_new.py
```

### Modal (all real model runs)
```bash
# First run: 1B for fast feedback (~15 min)
modal run modal/run_probing.py --model google/gemma-3-1b-it

# Main results: 4B (~30 min)
modal run modal/run_probing.py --model google/gemma-3-4b-it

# Optional: Qwen3 4B for cross-family comparison
modal run modal/run_probing.py --model Qwen/Qwen3-4B
```

---

## Questions to Resolve Before Running

1. **Qwen3 4B HF model ID**: Is it `Qwen/Qwen3-4B` or `Qwen/Qwen3-4B-Instruct`? Check HuggingFace.
2. **HaHackathon column names**: Confirm the CSV columns are `text`, `is_humor`, `humor_rating`, `offense_rating` before running `data_hahackathon.py`.
