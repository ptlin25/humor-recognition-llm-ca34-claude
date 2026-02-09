# Research Plan: How Low Rank is Humor Recognition in LLMs?

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding how LLMs internally represent humor has both scientific and practical significance. If humor recognition is low-rank, it means (1) humor can be efficiently fine-tuned with minimal parameters (LoRA at tiny ranks), (2) humor can be steered or suppressed via simple representation interventions, and (3) humor, despite being a complex cognitive phenomenon involving incongruity, surprise, and social context, may be reducible to a small number of latent features in neural networks.

### Gap in Existing Work
Based on the literature review, linear/low-rank representations have been studied for sentiment (Tigges et al., 2023), truth (Marks & Tegmark, 2023), and safety concepts (Zou et al., 2023), but **no existing work directly measures the rank of humor recognition in LLMs**. The "Laughing Heads" paper (Peyrard et al., 2021) showed humor concentrates in a single attention head in BERT, but didn't measure the dimensionality of the representation space. Intrinsic dimensionality studies (Aghajanyan et al., 2020) measured standard NLP tasks but not humor.

### Our Novel Contribution
We provide the first systematic measurement of the rank/dimensionality of humor recognition in LLM hidden representations, using multiple complementary methods:
1. PCA of humor-vs-non-humor activation differences to measure explained variance by rank
2. Linear probing at varying dimensions (1D to full) across layers
3. LoRA fine-tuning at varying ranks to find the minimum effective rank
4. Comparison against sentiment (expected rank ~1) as calibration

### Experiment Justification
- **Experiment 1 (Activation Collection & PCA)**: Directly answers "how many dimensions span the humor subspace" by examining the singular value spectrum of humor-related activations.
- **Experiment 2 (Multi-dimensional Linear Probes)**: Tests whether humor classification accuracy saturates at low rank, providing a functional measure of rank.
- **Experiment 3 (LoRA at Varying Ranks)**: Measures rank from a fine-tuning perspective - what's the minimum rank adapter that achieves competitive humor detection?
- **Experiment 4 (Comparison with Sentiment)**: Calibrates our findings against a known low-rank concept (sentiment ≈ rank 1) to contextualize humor's rank.

## Research Question
Is humor recognition in LLMs captured by a low-dimensional (low-rank) subspace of the hidden representations, and if so, what is its effective rank compared to other concepts like sentiment?

## Hypothesis Decomposition
1. **H1**: Hidden representations of LLMs contain linearly separable humor information (testable via linear probing accuracy > random baseline).
2. **H2**: The humor subspace is low-rank: a small number of principal components (e.g., <10) explain >90% of the variance in humor-related activation differences.
3. **H3**: LoRA fine-tuning at low ranks (r ≤ 8) achieves >90% of full fine-tuning performance for humor detection.
4. **H4**: Humor has higher effective rank than sentiment (which is ~1D) but is still low-rank relative to the full representation space.

## Proposed Methodology

### Approach
We use GPT-2 small (124M parameters) as our primary model, leveraging TransformerLens for activation extraction. We collect hidden representations for humor (Short Jokes dataset) and non-humor text (news-style factual sentences), then analyze the dimensionality of the humor-discriminating subspace through PCA, linear probing, and LoRA fine-tuning.

### Models
- **Primary**: GPT-2 small (124M) via TransformerLens - well-studied, fast, fits in memory
- **Scale test**: Pythia-410M to check if rank changes with model size

### Datasets
- **Humor**: Short Jokes dataset (231K jokes) - sample 2,000 for activation analysis, 5,000 for LoRA
- **Non-humor**: OpenWebText or news sentences matched in length
- **Sentiment comparison**: Stanford Sentiment Treebank (available in reference code)

### Experimental Steps

1. **Data Preparation** (30 min)
   - Load Short Jokes, sample and clean
   - Generate non-humor text from factual/news sources
   - Create balanced humor/non-humor dataset
   - Create train/val/test splits (60/20/20)

2. **Activation Collection** (30 min)
   - Run texts through GPT-2 small
   - Extract hidden states at every layer (last token position)
   - Store activations for humor and non-humor texts separately

3. **PCA / SVD Analysis** (20 min)
   - Compute mean difference between humor/non-humor activations per layer
   - Run PCA on the activation difference matrix
   - Plot singular value spectrum and cumulative explained variance
   - Identify the effective rank (number of components for 90%/95%/99% variance)

4. **Linear Probing at Varying Dimensions** (30 min)
   - Train linear probes with constrained rank (1D, 2D, 4D, 8D, 16D, full)
   - Measure accuracy at each rank per layer
   - Plot accuracy vs rank curves to find saturation point

5. **LoRA Fine-tuning** (60 min)
   - Fine-tune GPT-2 with LoRA at ranks 1, 2, 4, 8, 16, 32, 64
   - Measure humor classification accuracy at each rank
   - Find minimum rank achieving 90% of full fine-tuning

6. **Sentiment Comparison** (30 min)
   - Repeat PCA and linear probing for sentiment (SST-2)
   - Compare rank profiles between humor and sentiment

### Baselines
- Random direction baseline (shuffled labels for probing)
- Full-rank linear probe (upper bound for linear separability)
- Full fine-tuning (upper bound for LoRA comparison)

### Evaluation Metrics
- **Classification accuracy & F1**: For probing and LoRA experiments
- **Cumulative explained variance**: From PCA/SVD analysis
- **Effective rank (d90)**: Dimensions needed for 90% of peak performance
- **Singular value ratio**: σ₁/σ_k to measure concentration

### Statistical Analysis Plan
- Bootstrap confidence intervals (1000 resamples) for probe accuracies
- 3 random seeds for LoRA experiments, report mean ± std
- Wilcoxon signed-rank test for humor vs sentiment rank comparisons

## Expected Outcomes
- **If humor is very low-rank (rank 1-3)**: Similar to sentiment, humor would be a near-linear concept in LLM space. LoRA at r=1-2 would suffice.
- **If humor is moderately low-rank (rank 4-16)**: Humor is more complex than sentiment but still compressible. LoRA at r=4-8 would be needed.
- **If humor is high-rank (>16)**: Humor requires many latent dimensions, suggesting it's a fundamentally multi-faceted concept that resists simple linear representation.

## Timeline and Milestones
- Phase 0-1 (Planning + Review): 30 min ✓
- Phase 2 (Environment + Data): 30 min
- Phase 3 (Activation collection + PCA): 45 min
- Phase 4 (Probing + LoRA experiments): 90 min
- Phase 5 (Analysis + Visualization): 30 min
- Phase 6 (Documentation): 30 min

## Potential Challenges
1. **Memory**: GPT-2 activations for 2000 texts × 12 layers × 768 dims ≈ manageable
2. **Non-humor text quality**: Need to ensure non-humor text is matched in style/length, not trivially distinguishable
3. **Humor subjectivity**: Humor is more subjective than sentiment; may lead to noisier representations
4. **Confounds**: Jokes often have distinctive style (short, setup-punchline) that could be detected by surface features rather than humor understanding

## Success Criteria
1. Successfully extract and analyze activation patterns for humor vs non-humor
2. Produce clear rank estimates from at least 2 complementary methods (PCA + probing)
3. Compare humor's rank to sentiment's rank quantitatively
4. Generate visualizations showing the singular value spectrum and accuracy-vs-rank curves
