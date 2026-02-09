# Datasets

This directory contains humor recognition datasets for the research project "How Low Rank is Humor Recognition in LLMs?"

## Downloaded Datasets

### 1. Short Jokes (`short_jokes/`)

- **Source**: HuggingFace — `ysharma/short_jokes`
- **Original source**: Kaggle Short Jokes dataset
- **Size**: 231,657 texts
- **Task**: Humor corpus (jokes only — no negative examples included)
- **Format**: HuggingFace `datasets` format (Arrow)
- **Fields**: `Joke` (text of the joke)
- **Notes**: Contains only positive examples (jokes). To use for binary humor detection, pair with non-joke text sourced from news headlines, Wikipedia sentences, or OpenWebText.

**Sample** (from `short_jokes_samples.json`):
```json
{"Joke": "What do you call a belt made of watches? A waist of time."}
```

### 2. One Million Reddit Jokes (`one_million_reddit_jokes/`)

- **Source**: HuggingFace — `SocialGrep/one-million-reddit-jokes`
- **Size**: 1,000,000 Reddit posts
- **Task**: Humor corpus with community scores
- **Format**: HuggingFace `datasets` format (Arrow)
- **Fields**: `id`, `title`, `selftext`, `score`, `num_comments`, `subreddit`, etc.
- **Notes**: Reddit posts from joke-related subreddits with upvote scores. Useful for graded humor recognition (scores as proxy for humor quality). Contains both setup (title) and punchline (selftext) structure.

**Sample** (from `one_million_reddit_jokes_samples.json`):
```json
{"title": "Why did the chicken...", "selftext": "Punchline text", "score": 42}
```

### 3. Offensive Humor (`offensive_humor/`)

- **Source**: HuggingFace — `metaeval/offensive-humor`
- **Size**: 102,863 texts
- **Task**: Humor type classification with ratings
- **Format**: HuggingFace `datasets` format (Arrow)
- **Fields**: `text`, `humor_rating`, `joke_type`, `offensiveness_rating`
- **Notes**: Jokes with both humor and offensiveness ratings. Useful for studying the humor-offense correlation noted in Li et al. (2022).

**Sample** (from `offensive_humor_samples.json`):
```json
{"text": "Example joke text", "humor_rating": 3.5, "offensiveness_rating": 1.2}
```

## Download Instructions

To re-download the datasets, use the following Python code:

```python
from datasets import load_dataset

# Short Jokes
short_jokes = load_dataset("ysharma/short_jokes")
short_jokes.save_to_disk("datasets/short_jokes")

# One Million Reddit Jokes
reddit_jokes = load_dataset("SocialGrep/one-million-reddit-jokes")
reddit_jokes.save_to_disk("datasets/one_million_reddit_jokes")

# Offensive Humor
offensive = load_dataset("metaeval/offensive-humor")
offensive.save_to_disk("datasets/offensive_humor")
```

Requires: `pip install datasets`

## Additional Datasets (Not Downloaded)

These datasets are referenced in the literature but not directly downloaded. They may be useful for future experiments:

| Dataset | Source | Notes |
|---------|--------|-------|
| Unfun.me | Available via `code/laughing-head/` repo | 23K minimal pairs of funny/serious headlines; ideal for probing |
| HaHackathon (SemEval 2021 Task 7) | [CodaLab competition page](https://competitions.codalab.org/competitions/27446) | 10K texts with humor/offense ratings; standard benchmark |
| Humicroedit / FunLines | Various sources | ~15K humor editing pairs |

## Git Configuration

Large data files are excluded from git via `.gitignore`. Only documentation files (`README.md`), sample files (`*_samples.json`), and download scripts are tracked.

## Recommended Usage for Experiments

1. **Binary humor detection**: Use Short Jokes (positive) paired with non-joke text from OpenWebText or news headlines (negative)
2. **Graded humor recognition**: Use Reddit Jokes with upvote scores as continuous labels
3. **Minimal-pair probing**: Use Unfun.me pairs from the laughing-head repo for controlled experiments
4. **Humor subtype analysis**: Use Offensive Humor for studying humor types and the humor-offense relationship
