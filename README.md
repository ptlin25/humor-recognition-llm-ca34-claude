# How Low Rank is Humor Recognition in LLMs?

An empirical investigation into the dimensionality of humor recognition in the hidden representations of large language models.

## Key Findings

- **Humor-style detection (jokes vs. non-jokes) is rank-1**: A single linear direction in GPT-2's hidden space achieves 99.8% classification accuracy. LoRA at rank 0 (classification head only, 1,536 parameters) achieves 98.3%.
- **This captures text register, not humor understanding**: When controlling for stylistic confounds (jokes vs. unfunny joke attempts), accuracy drops to ~60% regardless of probe rank.
- **Humor quality discrimination is not linearly low-rank**: High-score vs. low-score Reddit jokes are barely separable (~61%) even with full-rank linear probes across all layers.
- **Results are consistent across model scales**: Both GPT-2 (124M) and Pythia-410M (410M) show identical patterns.
- **Sentiment (SST-2) falls between**: Rank-1 sentiment probe achieves 64%, full-rank achieves 77%---harder than humor-style but easier than humor-quality.

## Project Structure

```
.
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature review
├── resources.md                 # Catalog of gathered resources
├── src/
│   ├── data_preparation.py      # Dataset loading and preparation
│   ├── experiment_activations.py # Activation extraction, PCA, linear probing
│   ├── experiment_lora.py       # LoRA fine-tuning at varying ranks
│   ├── experiment_sentiment_comparison.py # Sentiment comparison
│   ├── experiment_hard_dataset.py # Controlled confound experiments
│   ├── experiment_pythia.py     # Cross-model scale comparison
│   ├── experiment_generalization.py # Cross-dataset generalization (Dataset A + HaHackathon)
│   ├── visualize_all.py         # Comprehensive figure generation
│   └── visualize_results.py     # Additional visualizations
├── results/
│   ├── prepared_data.json       # Processed dataset
│   ├── activation_analysis_results.json  # PCA and probing results
│   ├── lora_results.json        # LoRA fine-tuning results
│   ├── hard_dataset_results.json # Controlled confound results
│   ├── pythia_results.json      # Pythia-410M results
│   ├── sentiment_comparison_results.json
│   ├── generalization_results.json # Cross-dataset transfer results
│   └── plots/                   # All visualization figures
│       ├── figure1_main_results.png
│       ├── figure2_hard_tasks.png
│       ├── figure3_comparison.png
│       ├── figure4_cross_model.png
│       └── figure5_lora_detail.png
├── datasets/                    # Downloaded datasets (not in git)
├── papers/                      # Downloaded research papers
└── code/                        # Reference implementations
```

## How to Reproduce

```bash
# Set up environment (Python 3.11 required — Modal client is incompatible with 3.14+)
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch numpy scikit-learn matplotlib pandas tqdm datasets transformers modal

# Place HaHackathon CSVs before running (see note below)

# Run via Modal (uploads datasets, runs all experiments on cloud GPU, saves results)
modal setup                                        # authenticate once
modal secret create huggingface HF_TOKEN=<token>  # for Gemma (gated model)
uv run --python 3.11 modal run modal_run.py

# Run a single experiment
uv run --python 3.11 modal run modal_run.py::main --experiment activations --skip-upload True

# Download results
modal volume get humor-experiments results/ ./results/

# Generate figures locally
export USER=researcher  # May be needed in some environments
python src/visualize_all.py
```

**Requirements**: Python 3.11, CUDA GPU provisioned automatically by Modal (A10G).

**Note on local execution**: experiments can also be run locally (requires a CUDA GPU with ≥24 GB VRAM):
```bash
python src/data_preparation.py
python src/experiment_activations.py
python src/experiment_lora.py
python src/experiment_sentiment_comparison.py
python src/experiment_hard_dataset.py
python src/experiment_pythia.py
python src/experiment_generalization.py
python src/visualize_all.py
```

**HaHackathon dataset**: Download `train.csv` and `dev.csv` from the [SemEval 2021 Task 7 repository](https://competitions.codalab.org/competitions/27446) and place them in `datasets/hahackathon/`. The CSV must have columns: `id`, `text`, `is_humor`, `humor_rating`, `humor_controversy`, `offense_rating`.

## Citation

This research builds on methodology from:
- Tigges et al. (2023) "Linear Representations of Sentiment in LLMs"
- Aghajanyan et al. (2020) "Intrinsic Dimensionality Explains LM Fine-Tuning"
- Peyrard et al. (2021) "Laughing Heads: Can Transformers Detect What Makes a Sentence Funny?"

See [REPORT.md](REPORT.md) for the full research report with detailed results and analysis.
