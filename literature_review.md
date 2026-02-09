# Literature Review: How Low Rank is Humor Recognition in LLMs?

## Research Area Overview

This research sits at the intersection of three fields: (1) computational humor recognition, (2) mechanistic interpretability of LLMs, and (3) low-rank representations in neural networks. The central question is whether humor recognition in LLMs can be captured by a low-dimensional subspace of the model's hidden representations - analogous to how sentiment has been shown to be linearly represented along a single direction.

## Key Papers

### A. Humor Recognition and Detection

#### 1. Laughing Heads: Can Transformers Detect What Makes a Sentence Funny?
- **Authors**: Peyrard, Borges, Gligoric, West (EPFL)
- **Year**: 2021 (IJCAI)
- **arXiv**: 2105.09142
- **Key Contribution**: First mechanistic analysis of how transformers recognize humor. Discovered the "laughing head" - a single attention head (head 10-6 in BERT) that specializes in attending to the funny token in sentences, achieving 37% accuracy at identifying the humor-bearing word (5x random baseline).
- **Methodology**: Fine-tuned BERT on Unfun.me dataset (23K minimal pairs of funny/serious headlines). Analyzed attention patterns across all 144 heads. Used Jensen-Shannon divergence to measure attention distance between funny and serious sentences.
- **Datasets**: Unfun.me (23,113 paired funny/serious headlines with minimal edits)
- **Key Results**: 78% paired accuracy; humor detection happens in last transformer layers (semantic, not lexical/syntactic); single attention head explains most of the attention divergence between funny and serious processing.
- **Code**: https://github.com/epfl-dlab/laughing-head
- **Relevance**: **CRITICAL** - Directly shows that humor recognition in BERT is concentrated in a small number of model components, suggesting low-rank structure. The "laughing head" finding is strong evidence for a low-dimensional humor representation.

#### 2. SemEval 2021 Task 7: HaHackathon
- **Authors**: Meaney, Wilson, Chiruzzo, Lopez, Magdy
- **Year**: 2021
- **arXiv**: 2105.13602
- **Key Contribution**: Standard benchmark for humor detection. 10,000 texts annotated by 20 annotators aged 18-70 for humor and offense.
- **Datasets**: HaHackathon dataset (10K texts with humor/offense ratings)
- **Results**: Top systems used pre-trained language models with task-adaptive training and adversarial training. F1 up to 0.97 for humor detection.
- **Relevance**: Primary benchmark dataset for humor detection. Provides both binary humor labels and continuous humor ratings.

#### 3. Uncertainty and Surprisal Jointly Deliver the Punchline
- **Authors**: Xie, Li, Pu
- **Year**: 2020
- **arXiv**: 2012.12007
- **Key Contribution**: Models humor using incongruity theory - the set-up builds semantic uncertainty, the punchline disrupts expectations. Uses GPT-2 to compute uncertainty and surprisal as features for humor detection.
- **Methodology**: Splits jokes into set-up and punchline, computes perplexity-based features.
- **Relevance**: Suggests humor might be partially captured by simple distributional statistics (surprisal/uncertainty), which would imply low-rank representation.

#### 4. ColBERT: Using BERT Sentence Embedding for Computational Humor
- **Authors**: Annamoradnejad, Zoghi
- **Year**: 2020
- **arXiv**: 2004.12765
- **Key Contribution**: Parallel BERT architecture that processes set-up and punchline separately, then combines. Shows that humor can be detected by comparing representations of joke components.
- **Relevance**: Architecture design implicitly suggests humor is detectable from the relationship between a small number of representation vectors.

#### 5. What do Humor Classifiers Learn?
- **Authors**: Inácio, Wick-Pedro, Gonçalo Oliveira
- **Year**: 2023
- **Key Contribution**: Analyzed what BERT-based humor classifiers actually learn. Found classifiers rely mostly on stylistic aspects (punctuation, question words) rather than deep humor understanding. Content features achieve 99.64% F1 but may not capture true humor.
- **Relevance**: Caution that high humor detection performance may not reflect genuine humor understanding, and probing representations for humor-specific features is important.

#### 6. Getting Serious about Humor: Crafting Humor Datasets with Unfunny LLMs
- **Authors**: Horvitz, Chen, Aditya, Srivastava, West, Yu, McKeown
- **Year**: 2024
- **arXiv**: 2403.00794
- **Key Contribution**: Shows LLMs can "unfun" jokes (remove humor), creating aligned funny/non-funny pairs. GPT-4's synthetic unfunned data is highly rated by humans. Extends to code-mixed English-Hindi humor.
- **Relevance**: Provides methodology for creating controlled humor datasets (minimal pairs), which is ideal for probing linear representations.

#### 7. This Joke is [MASK]: Recognizing Humor and Offense with Prompting
- **Authors**: Li, Zhao, Xie, Maronikolakis, Pu, Schütze
- **Year**: 2022
- **arXiv**: 2209.12118
- **Key Contribution**: Shows prompting performs as well as fine-tuning for humor recognition when data is abundant, and excels in low-resource settings. Uses influence functions to show models rely on offense to determine humor.
- **Relevance**: Demonstrates humor can be detected through prompting (i.e., through the model's existing representations), supporting the idea of a pre-existing humor direction.

#### 8. Humor Detection: A Transformer Gets the Last Laugh
- **Authors**: Weller, Seppi
- **Year**: 2019
- **arXiv**: 1909.00252
- **Key Contribution**: Early work showing transformers achieve near-human performance on humor detection (98.6% on Short Jokes, 93.1% on Puns). Built a new dataset from Reddit ratings.
- **Citations**: 140
- **Relevance**: Established that transformers can effectively detect humor, motivating the question of how they do it internally.

### B. Linear and Low-Rank Representations in LLMs

#### 9. Linear Representations of Sentiment in Large Language Models
- **Authors**: Tigges, Hollinsworth, Geiger, Nanda
- **Year**: 2023
- **arXiv**: 2310.15154
- **Key Contribution**: **Most directly relevant methodology paper.** Shows sentiment is represented linearly in LLMs - a single direction in activation space captures positive/negative sentiment. Found the "summarization motif" where sentiment aggregates on punctuation/name tokens. Ablating this direction removes 76% of SST classification accuracy.
- **Methodology**: Five methods to find sentiment direction (Mean Difference, K-means, Logistic Regression, DAS, PCA) - all converge to the same direction (cosine similarity 79-99%). Used causal interventions (activation patching, directional ablation) to verify directions are causally relevant. Best generalization at intermediate layers.
- **Models**: GPT-2, Pythia family (160M to 2.8B)
- **Relevance**: **CRITICAL** - This is the methodological template for our research. We can apply the exact same methodology to humor instead of sentiment: find a linear humor direction, test it causally, measure its dimensionality.

#### 10. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
- **Authors**: Aghajanyan, Zettlemoyer, Gupta (Facebook)
- **Year**: 2020
- **arXiv**: 2012.13255
- **Key Contribution**: **Foundational for the "low rank" question.** Shows NLP tasks have very low intrinsic dimension when fine-tuning pre-trained models. RoBERTa-Large can reach 90% of full fine-tuning performance on MRPC with only 200 parameters. Pre-training implicitly minimizes intrinsic dimension. Larger models have lower intrinsic dimension.
- **Methodology**: Subspace optimization via Fastfood transform. Structure-Aware Intrinsic Dimension (SAID) method. Binary search for d90 (smallest dimension achieving 90% of full performance).
- **Key Results**: d90 for MRPC with RoBERTa-Large = 207 (SAID method). Pre-training monotonically decreases intrinsic dimension. Harder tasks have higher intrinsic dimension (ANLI >> Yelp Polarity).
- **Relevance**: **CRITICAL** - Provides the theoretical framework and measurement methodology for our research. We should measure the intrinsic dimensionality of humor recognition and compare it to other NLP tasks.

#### 11. LoRA: Low-Rank Adaptation of Large Language Models
- **Authors**: Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen
- **Year**: 2021
- **arXiv**: 2106.09685
- **Key Contribution**: Shows that the weight updates during fine-tuning have low intrinsic rank. LoRA freezes pre-trained weights and injects trainable low-rank decomposition matrices, achieving comparable performance with far fewer trainable parameters.
- **Methodology**: Decomposes weight update as ΔW = BA where B∈R^{d×r}, A∈R^{r×k}, r << min(d,k).
- **Key Results**: Rank r=1-8 sufficient for many tasks. Reduces trainable parameters by up to 10,000x.
- **Relevance**: **HIGH** - LoRA's success at low rank directly supports our hypothesis. We can use LoRA rank as a measure of how low-rank humor recognition fine-tuning can be.

#### 12. Representation Engineering: A Top-Down Approach to AI Transparency
- **Authors**: Zou, Phan, Chen, Campbell, Guo, Ren, Pan, Yin, Mazeika, Dombrowski, Goel, Li, Byun, Wang, Mallen, Basart, Koyber, Li, Song, Song, Zhu, Hendrycks, Boyd-Graber
- **Year**: 2023
- **arXiv**: 2310.01405
- **Key Contribution**: Introduces RepE (Representation Engineering) for reading and controlling LLM representations. Identifies linear directions for concepts like honesty, fairness, harmlessness. Shows these can be used to steer model behavior.
- **Relevance**: Provides additional methodology for finding and manipulating concept directions in LLM representations, applicable to humor.

#### 13. The Geometry of Truth
- **Authors**: Marks, Tegmark
- **Year**: 2023
- **arXiv**: 2310.06824
- **Key Contribution**: Shows that truth/falsehood is linearly represented in LLM activations. Multiple probe methods converge on similar directions. Truth directions generalize across diverse statement types.
- **Relevance**: Another demonstration that abstract concepts (truth, like humor) can be linearly represented, supporting our hypothesis.

#### 14. ReFT: Representation Finetuning for Language Models
- **Authors**: Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts
- **Year**: 2024
- **arXiv**: 2402.14700
- **Key Contribution**: Fine-tunes by learning interventions on hidden representations rather than weights. LoReFT (Low-rank Linear Subspace ReFT) achieves competitive results with 10-50x fewer parameters than LoRA.
- **Relevance**: **HIGH** - LoReFT's success at very low rank directly applies to our question. Using LoReFT for humor could measure the effective rank of humor in representations.

#### 15. Discovering Latent Knowledge in Language Models Without Supervision
- **Authors**: Burns, Ye, Klein, Steinhardt
- **Year**: 2022
- **arXiv**: 2212.03827
- **Key Contribution**: Contrast Consistent Search (CCS) finds latent knowledge directions without labels. Shows LLMs have internal representations of truth that can be extracted unsupervised.
- **Relevance**: Methodology for finding concept directions without labeled data, potentially applicable to humor.

#### 16. Not All Language Model Features Are Linear
- **Authors**: Engels, Liao, Tegmark
- **Year**: 2024
- **arXiv**: 2405.14860
- **Key Contribution**: Shows some features in LLMs are represented as multi-dimensional, non-linear structures (e.g., circular features for periodic concepts like days/months). Challenges pure linear representation hypothesis.
- **Relevance**: Important caveat - humor may not be purely linear. Could require higher-dimensional (but still low-rank) representation structure.

### C. Probing and Mechanistic Interpretability

#### 17. Designing and Interpreting Probes with Control Tasks
- **Authors**: Hewitt, Liang
- **Year**: 2019
- **arXiv**: 1909.03368
- **Key Contribution**: Introduces "control tasks" to ensure probing classifiers measure genuine linguistic knowledge rather than probe memorization capacity. Defines selectivity as the difference between linguistic and control task accuracy.
- **Relevance**: Essential methodology for properly probing humor representations - we need control tasks to ensure any humor direction we find is genuine.

#### 18. Do LLMs Understand Social Knowledge? SocKET Benchmark
- **Authors**: Choi, Pei, Kumar, Shu, Jurgens
- **Year**: 2023
- **arXiv**: 2305.14938
- **Key Contribution**: 58 NLP tasks testing social knowledge including humor and sarcasm. Shows potential for task transfer among social knowledge tasks. Pre-trained models have some innate but limited social language understanding.
- **Relevance**: Provides context for humor as part of broader social understanding in LLMs.

## Common Methodologies

### For Finding Linear Directions
- **Mean Difference**: Compute centroid difference between positive/negative examples (used in Tigges et al., Marks et al.)
- **K-means**: Cluster activations into 2 groups, take direction between centroids
- **Logistic Regression / Linear Probing**: Train linear classifier, use weight vector as direction
- **PCA**: First principal component of activations
- **DAS (Distributed Alignment Search)**: Learn direction that maximizes causal intervention effect

All methods tend to converge to similar directions (cosine similarity >80%), suggesting a genuine underlying direction.

### For Measuring Low-Rankness
- **Intrinsic Dimensionality (d90)**: Find smallest subspace dimension achieving 90% of full performance (Aghajanyan et al.)
- **LoRA Rank**: Minimum rank of LoRA adapter achieving target performance
- **Linear Probe Accuracy**: How well a linear classifier captures the concept
- **Directional Ablation**: Remove a single direction and measure performance drop
- **Activation Patching**: Swap activations along a direction and measure behavior change

### For Causal Validation
- **Activation Patching**: Swap activations between clean/corrupted examples
- **Directional Ablation**: Zero out component along specific direction
- **Activation Addition**: Add multiples of direction to steer model behavior

## Standard Baselines

1. **Full fine-tuning** of BERT/RoBERTa for humor detection
2. **Linear probing** of frozen LLM representations
3. **LoRA fine-tuning** at various ranks (r=1,2,4,8,16,...)
4. **Random direction** baseline (control for linear probing)
5. **GPT-2 perplexity** baseline (surprisal-based humor detection)

## Evaluation Metrics

- **Accuracy / F1-score**: For humor detection classification
- **d90 (intrinsic dimension)**: For measuring low-rankness
- **Cosine similarity**: Between directions found by different methods
- **Logit difference / logit flip rate**: For causal intervention experiments
- **Explained variance**: By top-k principal components of humor-related activations

## Datasets in the Literature

| Dataset | Used In | Task | Size |
|---------|---------|------|------|
| Unfun.me | Laughing Heads | Paired humor detection | 23K pairs |
| HaHackathon (SemEval 2021 Task 7) | Multiple | Binary humor + rating | 10K texts |
| Short Jokes | Weller & Seppi 2019 | Humor detection | 231K jokes |
| Reddit Jokes | Multiple | Humor detection | 1M+ posts |
| Humicroedit / FunLines | Multiple | Humor editing/generation | ~15K pairs |
| Stanford Sentiment Treebank | Linear Sentiment | Sentiment classification | 10.6K sentences |
| ToyMovieReview | Linear Sentiment | Sentiment probing | ~170 templates |

## Gaps and Opportunities

1. **No existing work directly measures the rank of humor recognition in LLMs** - This is the central gap our research addresses.
2. **Linear representation studies focus on sentiment, truth, and safety** - humor is an unexplored target for this methodology.
3. **Humor interpretability work (Laughing Heads) uses attention analysis, not representation probing** - deeper mechanistic understanding via probing/ablation is needed.
4. **Intrinsic dimensionality was measured for standard NLP tasks but not humor** - humor may be higher-dimensional than sentiment due to its subjective, context-dependent nature.
5. **No comparison of humor's representational complexity vs. other figurative language tasks** (sarcasm, irony, metaphor).

## Recommendations for Our Experiment

### Recommended Datasets
1. **Primary**: Short Jokes dataset (231K, well-studied, binary humor labels) + non-joke text for negative examples
2. **Secondary**: Unfun.me aligned pairs (minimal pairs ideal for probing)
3. **Validation**: HaHackathon (standard benchmark with human ratings)

### Recommended Models
1. **GPT-2 small** (85M) - well-studied, good for initial probing
2. **Pythia family** (70M-2.8B) - multiple scales, same architecture, used in Linear Sentiment paper
3. **Llama-2/3** (7B+) - larger modern models, if compute allows

### Recommended Methodology
1. **Collect humor/non-humor activation datasets** from LLM hidden states
2. **Find humor direction(s)** using multiple methods (Mean Diff, PCA, LR, K-means, DAS)
3. **Measure rank**:
   - Singular value decomposition of humor-related activation differences
   - Intrinsic dimensionality via subspace training (d90)
   - LoRA/LoReFT fine-tuning at varying ranks
   - Number of principal components needed to explain variance
4. **Causal validation**: Activation patching, directional ablation
5. **Compare to baselines**: Sentiment (expected rank ~1), random (expected high rank)

### Recommended Metrics
- **d90**: Intrinsic dimension for humor classification
- **Minimum LoRA rank** achieving 90% of full fine-tuning
- **Linear probe accuracy** (single direction vs multi-dimensional)
- **Ablation impact**: Performance drop when removing top-k humor directions
- **Cross-task comparison**: Humor d90 vs. sentiment d90 vs. other tasks

### Methodological Considerations
- Use **control tasks** (Hewitt & Liang, 2019) to validate probing results
- Test across **multiple model scales** to see if humor rank changes with model size
- Consider that humor may be **multi-dimensional** (not purely linear) - test rank >1
- Account for **humor subtypes** (puns, absurdity, obscenity have different mechanisms)
- Humor is more **subjective** than sentiment - may have higher variance in representations
