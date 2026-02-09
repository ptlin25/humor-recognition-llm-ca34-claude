# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "How Low Rank is Humor Recognition in LLMs?", including papers, datasets, and code repositories.

## Papers

Total papers downloaded: 32

### Humor Recognition Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Laughing Heads | Peyrard et al. | 2021 | papers/2105.09142_laughing_heads.pdf | Single attention head detects humor; 78% paired accuracy on aligned dataset |
| SemEval 2021 Task 7: HaHackathon | Meaney et al. | 2021 | papers/2105.13602_semeval2021_hahackathon.pdf | Standard humor detection benchmark; 10K annotated texts |
| Humor Detection: Transformer Gets the Last Laugh | Weller & Seppi | 2019 | papers/1909.00252_humor_detection_transformer.pdf | 98.6% on Short Jokes; 93.1% on Puns; Reddit ratings |
| ColBERT: BERT for Computational Humor | Annamoradnejad & Zoghi | 2020 | papers/2004.12765_colbert_humor.pdf | Parallel BERT architecture for humor; 200K dataset |
| Getting Serious about Humor | Horvitz et al. | 2024 | papers/2403.00794_getting_serious_humor.pdf | LLMs can "unfun" jokes; synthetic unfunned pairs |
| This Joke is [MASK] | Li et al. | 2022 | papers/2209.12118_joke_mask_prompting.pdf | Prompting for humor; influence functions show offense-humor link |
| Uncertainty and Surprisal | Xie et al. | 2020 | papers/2012.12007_uncertainty_surprisal_punchline.pdf | GPT-2 surprisal features for humor detection |
| Humor Knowledge Enriched Transformer | Hasan et al. | 2021 | papers/2103.09188_humor_knowledge_transformer.pdf | Multimodal humor with external knowledge; 77-79% accuracy |
| Large Dataset and Language Model Fun-Tuning | Blinov et al. | 2019 | papers/1904.06130_large_dataset_humor_recognition.pdf | 300K+ Russian humor dataset; 0.91 F1 |
| Can Pre-trained LMs Understand Chinese Humor? | Chen et al. | 2023 | papers/2302.02895_chinese_humor_plms.pdf | Comprehensive Chinese humor evaluation framework |
| Transfer Learning for Humor Detection | Arora et al. | 2022 | papers/2210.00710_transfer_learning_humor.pdf | Multi-task architecture for humor transfer |
| LRG: Humor Grading with BERT | Mahurkar & Patil | 2020 | papers/2005.02439_lrg_humor_grading.pdf | BERT self-attention analysis for humor |
| Do LLMs Understand Social Knowledge? (SocKET) | Choi et al. | 2023 | papers/2305.14938_socket_benchmark.pdf | 58 social NLP tasks including humor |

### Low-Rank / Probing / Interpretability Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Linear Representations of Sentiment | Tigges et al. | 2023 | papers/2310.15154_linear_sentiment.pdf | **Key methodology**: single direction captures sentiment; causal validation |
| Intrinsic Dimensionality | Aghajanyan et al. | 2020 | papers/2012.13255_intrinsic_dimensionality.pdf | **Key theory**: NLP tasks have very low intrinsic dimension (d90~200) |
| LoRA | Hu et al. | 2021 | papers/2106.09685_lora.pdf | Low-rank adaptation; rank 1-8 sufficient for many tasks |
| Representation Engineering | Zou et al. | 2023 | papers/2310.01405_representation_engineering.pdf | Reading/controlling LLM representations via linear directions |
| Geometry of Truth | Marks & Tegmark | 2023 | papers/2310.06824_geometry_of_truth.pdf | Truth linearly represented; generalizes across statement types |
| Discovering Latent Knowledge | Burns et al. | 2022 | papers/2212.03827_discovering_latent_knowledge.pdf | Unsupervised extraction of concept directions |
| Space and Time Representations | Gurnee & Tegmark | 2023 | papers/2310.02207_space_time_representations.pdf | Linear probes for geographic/temporal concepts in LLMs |
| Probing with Control Tasks | Hewitt & Liang | 2019 | papers/1909.03368_probing_control_tasks.pdf | Control task methodology for valid probing |
| ReFT: Representation Fine-Tuning | Wu et al. | 2024 | papers/2402.14700_representation_finetuning.pdf | LoReFT: 10-50x fewer params than LoRA |
| Not All Features Are Linear | Engels et al. | 2024 | papers/2405.14860_nonlinear_features.pdf | Multi-dimensional non-linear feature structures |
| Towards Monosemanticity (SAEs) | Anthropic | 2023 | papers/2309.12263_sparse_autoencoders_features.pdf | Sparse autoencoders for interpretable features |
| Feature Sparsity in LMs | N/A | 2023 | papers/2310.07837_feature_sparsity.pdf | Measuring sparsity of learned features |
| Vector Arithmetic in LMs | N/A | 2023 | papers/2305.16130_vector_arithmetic_lms.pdf | LMs implement Word2Vec-style arithmetic |
| Understanding LLM Representations | N/A | 2023 | papers/2310.17191_understanding_llm_representations.pdf | Analysis of LLM internal representations |
| Probing via Prompting | N/A | 2023 | papers/2306.03819_probing_via_prompting.pdf | Using prompts as probes for representations |

### LoRA Variants

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| DyLoRA | Valipour et al. | 2022 | papers/2205.05638_dylora.pdf | Dynamic rank selection for LoRA |
| VeRA | Kopiczko et al. | 2023 | papers/2309.15223_vera.pdf | Vector-based random matrix adaptation |
| LoRA-XS | N/A | 2024 | papers/2310.18168_lora_xs.pdf | Extremely small parameter LoRA |
| AutoLoRA | N/A | 2024 | papers/2310.11454_autolora.pdf | Auto-tuning LoRA ranks via meta-learning |

See papers/README.md for detailed descriptions.

## Datasets

Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Short Jokes | HuggingFace (ysharma/short_jokes) | 231,657 texts | Humor corpus (jokes only) | datasets/short_jokes/ | Short jokes from Kaggle; need negative examples |
| One Million Reddit Jokes | HuggingFace (SocialGrep/one-million-reddit-jokes) | 1,000,000 posts | Humor corpus with scores | datasets/one_million_reddit_jokes/ | Reddit posts with scores; useful for graded humor |
| Offensive Humor | HuggingFace (metaeval/offensive-humor) | 102,863 texts | Humor type classification | datasets/offensive_humor/ | Joke types + scores from Reddit |

See datasets/README.md for download instructions and detailed descriptions.

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| laughing-head | https://github.com/epfl-dlab/laughing-head | Humor detection analysis with BERT | code/laughing-head/ | Includes Unfun.me data processing and attention analysis |
| eliciting-latent-sentiment | https://github.com/curt-tigges/eliciting-latent-sentiment | Linear sentiment direction finding | code/eliciting-latent-sentiment/ | **Key reference code** for finding linear directions in LLM activations |
| pyreft | https://github.com/stanfordnlp/pyreft | Representation Fine-Tuning library | code/pyreft/ | LoReFT implementation for low-rank representation interventions |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (Semantic Scholar-backed) with diligent mode for four search queries covering humor recognition, low-rank representations, LoRA variants, and linear probing
2. Manual arXiv searches for key methodology papers on representation engineering and interpretability
3. HuggingFace Hub search for humor datasets
4. GitHub search for implementations of key papers

### Selection Criteria
- Papers directly relevant to humor recognition in LLMs (priority 1)
- Papers establishing methodology for linear/low-rank representation analysis (priority 2)
- Papers on LoRA and intrinsic dimensionality as measurement tools (priority 3)
- Datasets with binary humor labels or humor ratings (essential for probing)
- Code repositories that can be adapted for our methodology

### Challenges Encountered
- arXiv ID 2108.12066 does not correspond to "Laughing Heads" (it's a plasma physics paper) - corrected to 2105.09142
- Several HuggingFace dataset names have changed or require trust_remote_code which is deprecated
- The HaHackathon dataset is not directly available on HuggingFace in standard format; available via the original task repository
- Some paper-finder searches timed out; supplemented with manual searches

### Gaps and Workarounds
- **Unfun.me dataset**: Not directly on HuggingFace but available through the laughing-head repository
- **HaHackathon dataset**: May need to download from original SemEval task page
- **Non-humor text for negative examples**: Can be sourced from news headlines, Wikipedia, or OpenWebText via standard HuggingFace datasets

## Recommendations for Experiment Design

1. **Primary dataset(s)**:
   - Short Jokes (231K) paired with non-joke text for binary humor classification
   - Use Reddit joke scores for graded humor recognition
   - Unfun.me pairs (from laughing-head repo) for controlled minimal-pair probing

2. **Baseline methods**:
   - Full fine-tuning (upper bound)
   - Linear probe on frozen representations (measure of linear separability)
   - LoRA at ranks 1, 2, 4, 8, 16, 32, 64 (measure of low-rank adaptability)
   - LoReFT at various ranks (measure of representation-level low-rankness)
   - Random direction baseline (control)

3. **Evaluation metrics**:
   - d90 (intrinsic dimension) - primary metric for "how low rank"
   - Minimum effective LoRA rank
   - Linear probe accuracy vs. full fine-tuning
   - Singular value spectrum of humor-related activation differences
   - Cross-task comparison (humor vs. sentiment vs. other tasks)

4. **Code to adapt/reuse**:
   - `eliciting-latent-sentiment` for finding linear humor directions (adapt for humor)
   - `pyreft` for LoReFT-based rank measurement
   - `laughing-head` for dataset preprocessing and attention analysis
