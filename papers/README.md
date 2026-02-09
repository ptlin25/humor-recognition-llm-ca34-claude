# Papers

This directory contains 32 academic papers relevant to the research question: **"How Low Rank is Humor Recognition in LLMs?"**

Papers are organized into three categories: humor recognition, low-rank/probing/interpretability, and LoRA variants.

## Humor Recognition Papers (13 papers)

| File | Title | Authors | Year | arXiv |
|------|-------|---------|------|-------|
| `2105.09142_laughing_heads.pdf` | Laughing Heads: Can Transformers Detect What Makes a Sentence Funny? | Peyrard et al. | 2021 | 2105.09142 |
| `2105.13602_semeval2021_hahackathon.pdf` | SemEval 2021 Task 7: HaHackathon | Meaney et al. | 2021 | 2105.13602 |
| `1909.00252_humor_detection_transformer.pdf` | Humor Detection: A Transformer Gets the Last Laugh | Weller & Seppi | 2019 | 1909.00252 |
| `2004.12765_colbert_humor.pdf` | ColBERT: Using BERT Sentence Embedding for Computational Humor | Annamoradnejad & Zoghi | 2020 | 2004.12765 |
| `2403.00794_getting_serious_humor.pdf` | Getting Serious about Humor: Crafting Humor Datasets with Unfunny LLMs | Horvitz et al. | 2024 | 2403.00794 |
| `2209.12118_joke_mask_prompting.pdf` | This Joke is [MASK]: Recognizing Humor and Offense with Prompting | Li et al. | 2022 | 2209.12118 |
| `2012.12007_uncertainty_surprisal_punchline.pdf` | Uncertainty and Surprisal Jointly Deliver the Punchline | Xie et al. | 2020 | 2012.12007 |
| `2103.09188_humor_knowledge_transformer.pdf` | Humor Knowledge Enriched Transformer | Hasan et al. | 2021 | 2103.09188 |
| `1904.06130_large_dataset_humor_recognition.pdf` | Large Dataset and Language Model Fun-Tuning for Humor Recognition | Blinov et al. | 2019 | 1904.06130 |
| `2302.02895_chinese_humor_plms.pdf` | Can Pre-trained Language Models Understand Chinese Humor? | Chen et al. | 2023 | 2302.02895 |
| `2210.00710_transfer_learning_humor.pdf` | Transfer Learning for Humor Detection | Arora et al. | 2022 | 2210.00710 |
| `2005.02439_lrg_humor_grading.pdf` | LRG at SemEval-2020: Assessing Humor in Edited Headlines Using BERT | Mahurkar & Patil | 2020 | 2005.02439 |
| `2305.14938_socket_benchmark.pdf` | Do LLMs Understand Social Knowledge? Evaluating with SocKET Benchmark | Choi et al. | 2023 | 2305.14938 |

### Key Papers for This Research

- **Laughing Heads** (2105.09142): Discovered a single attention head in BERT specializing in humor detection. Directly shows humor recognition is concentrated in few model components, suggesting low-rank structure.
- **Getting Serious about Humor** (2403.00794): LLMs can "unfun" jokes to create minimal pairs — ideal for probing linear representations.
- **HaHackathon** (2105.13602): Standard benchmark with 10K annotated texts for humor detection evaluation.

## Low-Rank / Probing / Interpretability Papers (16 papers)

| File | Title | Authors | Year | arXiv |
|------|-------|---------|------|-------|
| `2310.15154_linear_sentiment.pdf` | Linear Representations of Sentiment in Large Language Models | Tigges et al. | 2023 | 2310.15154 |
| `2012.13255_intrinsic_dimensionality.pdf` | Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning | Aghajanyan et al. | 2020 | 2012.13255 |
| `2106.09685_lora.pdf` | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | 2106.09685 |
| `2310.01405_representation_engineering.pdf` | Representation Engineering: A Top-Down Approach to AI Transparency | Zou et al. | 2023 | 2310.01405 |
| `2310.06824_geometry_of_truth.pdf` | The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets | Marks & Tegmark | 2023 | 2310.06824 |
| `2212.03827_discovering_latent_knowledge.pdf` | Discovering Latent Knowledge in Language Models Without Supervision | Burns et al. | 2022 | 2212.03827 |
| `2310.02207_space_time_representations.pdf` | Language Models Represent Space and Time | Gurnee & Tegmark | 2023 | 2310.02207 |
| `1909.03368_probing_control_tasks.pdf` | Designing and Interpreting Probes with Control Tasks | Hewitt & Liang | 2019 | 1909.03368 |
| `2402.14700_representation_finetuning.pdf` | ReFT: Representation Finetuning for Language Models | Wu et al. | 2024 | 2402.14700 |
| `2405.14860_nonlinear_features.pdf` | Not All Language Model Features Are Linear | Engels et al. | 2024 | 2405.14860 |
| `2309.12263_sparse_autoencoders_features.pdf` | Towards Monosemanticity: Decomposing Language Models with Dictionary Learning | Anthropic | 2023 | 2309.12263 |
| `2310.07837_feature_sparsity.pdf` | Feature Sparsity in Language Models | N/A | 2023 | 2310.07837 |
| `2305.16130_vector_arithmetic_lms.pdf` | Vector Arithmetic in Language Models | N/A | 2023 | 2305.16130 |
| `2310.17191_understanding_llm_representations.pdf` | Understanding LLM Representations | N/A | 2023 | 2310.17191 |
| `2306.03819_probing_via_prompting.pdf` | Probing via Prompting | N/A | 2023 | 2306.03819 |

### Key Papers for This Research

- **Linear Representations of Sentiment** (2310.15154): **Primary methodological template.** Shows sentiment is captured by a single linear direction; five methods (Mean Diff, K-means, LR, PCA, DAS) all converge. We apply this methodology to humor.
- **Intrinsic Dimensionality** (2012.13255): **Primary theoretical framework.** d90 metric measures minimum dimensions needed. RoBERTa-Large achieves 90% MRPC performance with only 200 parameters.
- **LoRA** (2106.09685): Demonstrates that fine-tuning weight updates are inherently low-rank (rank 1-8 sufficient). Provides LoRA rank as a practical low-rank measurement tool.
- **ReFT** (2402.14700): LoReFT achieves competitive results with 10-50x fewer parameters than LoRA, measuring representation-level low-rankness.

## LoRA Variant Papers (3 papers)

| File | Title | Authors | Year | arXiv |
|------|-------|---------|------|-------|
| `2205.05638_dylora.pdf` | DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation | Valipour et al. | 2022 | 2205.05638 |
| `2309.15223_vera.pdf` | VeRA: Vector-based Random Matrix Adaptation | Kopiczko et al. | 2023 | 2309.15223 |
| `2310.18168_lora_xs.pdf` | LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters | N/A | 2024 | 2310.18168 |
| `2310.11454_autolora.pdf` | AutoLoRA: Automatically Tuning Multi-Task LoRA Ranks via Meta-Learning | N/A | 2024 | 2310.11454 |

## Chunked Papers

The `pages/` subdirectory contains chunked versions of key papers (3 pages per chunk) for easier reading. Manifest files (`*_manifest.txt`) list the chunk breakdown.

## Download Source

All papers were downloaded from arXiv (https://arxiv.org/pdf/{arxiv_id}).
