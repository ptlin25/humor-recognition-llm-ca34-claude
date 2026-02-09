# Code Repositories

This directory contains cloned code repositories relevant to the research project "How Low Rank is Humor Recognition in LLMs?"

## Repositories

### 1. laughing-head (`laughing-head/`)

- **Source**: https://github.com/epfl-dlab/laughing-head
- **Authors**: Peyrard, Borges, Gligoric, West (EPFL)
- **Paper**: "Laughing Heads: Can Transformers Detect What Makes a Sentence Funny?" (2021)
- **Purpose**: Humor detection analysis with BERT, including attention head analysis and the Unfun.me dataset processing pipeline
- **Key Contents**:
  - BERT fine-tuning for humor detection on Unfun.me paired data
  - Attention pattern analysis across all 144 BERT heads
  - Jensen-Shannon divergence computation between funny/serious attention patterns
  - Unfun.me dataset loader and preprocessing
- **Relevance**: Provides dataset preprocessing code for Unfun.me minimal pairs and attention analysis methodology. The Unfun.me data is accessible through this repo.

### 2. eliciting-latent-sentiment (`eliciting-latent-sentiment/`)

- **Source**: https://github.com/curt-tigges/eliciting-latent-sentiment
- **Authors**: Tigges, Hollinsworth, Geiger, Nanda
- **Paper**: "Linear Representations of Sentiment in Large Language Models" (2023)
- **Purpose**: Finding linear sentiment directions in LLM activation spaces
- **Key Contents**:
  - Multiple methods for finding concept directions: Mean Difference, K-means, Logistic Regression, PCA, DAS
  - Causal validation via activation patching and directional ablation
  - Analysis across GPT-2 and Pythia model families
  - Evaluation on SST (Stanford Sentiment Treebank) and ToyMovieReview datasets
- **Relevance**: **Primary reference implementation.** This code should be adapted to find humor directions instead of sentiment directions. The methodology (5 direction-finding methods + causal validation) is directly applicable.

### 3. pyreft (`pyreft/`)

- **Source**: https://github.com/stanfordnlp/pyreft
- **Authors**: Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts (Stanford NLP)
- **Paper**: "ReFT: Representation Finetuning for Language Models" (2024)
- **Purpose**: Library for Representation Fine-Tuning, including LoReFT (Low-rank Linear Subspace ReFT)
- **Key Contents**:
  - LoReFT implementation for low-rank representation interventions
  - Support for various ReFT methods (linear, low-rank, distributed)
  - Integration with HuggingFace Transformers
  - Examples and evaluation scripts
- **Relevance**: Use LoReFT at varying ranks to measure the effective rank of humor in LLM representations. Provides a complementary measurement to LoRA rank experiments.

## Recommended Adaptation Strategy

### For Finding Humor Directions
1. Start with `eliciting-latent-sentiment/` code
2. Replace sentiment datasets (SST, ToyMovieReview) with humor datasets (Short Jokes + non-jokes, or Unfun.me pairs)
3. Adapt the 5 direction-finding methods to extract humor directions
4. Run causal validation (activation patching, directional ablation) on the humor directions
5. Compare cosine similarity between humor directions found by different methods

### For Measuring Low-Rankness
1. Use `pyreft/` to run LoReFT at ranks 1, 2, 4, 8, 16, 32, 64
2. Compare humor LoReFT performance curve against sentiment as baseline
3. Identify minimum rank achieving 90% of full fine-tuning performance (d90 equivalent)

### For Dataset Preprocessing
1. Use `laughing-head/` for Unfun.me data loading and preprocessing
2. Adapt the paired-data evaluation framework for probing experiments

## Re-cloning

To re-clone the repositories:

```bash
git clone https://github.com/epfl-dlab/laughing-head code/laughing-head
git clone https://github.com/curt-tigges/eliciting-latent-sentiment code/eliciting-latent-sentiment
git clone https://github.com/stanfordnlp/pyreft code/pyreft
```
