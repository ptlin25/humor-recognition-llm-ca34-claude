"""
Evaluate activation steering quality using GPT as a pairwise humor judge.

For each alpha and each steered completion, compares it head-to-head against
the best-of-N baseline completion using an LLM judge. Reports win rates per alpha.

Requires:
  - results/{slug}_steering.json  (from experiment_steering.py, must include best_of_n)
  - OPENAI_API_KEY environment variable

Usage:
    python src/evaluate_steering_judge.py --model google/gemma-3-4b-it
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42
random.seed(SEED)

JUDGE_MODEL = "gpt-5.2"
N_JUDGE_REPEATS = 3  # votes per pairwise comparison (majority wins)

JUDGE_PROMPT = """\
You are evaluating humor quality.

Prompt: "{prompt}"

Completion A: "{completion_a}"

Completion B: "{completion_b}"

Which completion is funnier given the prompt?
Reply with exactly "A" or "B". Do not explain."""


def model_slug(model_id):
    return model_id.replace("/", "-")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def judge_pair(client, prompt, completion_a, completion_b):
    """
    Ask GPT to pick the funnier completion. Returns "A" or "B".
    A/B assignment is passed in by the caller (caller handles randomization).
    """
    message = JUDGE_PROMPT.format(
        prompt=prompt,
        completion_a=completion_a,
        completion_b=completion_b,
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": message}],
    )
    answer = response.choices[0].message.content.strip().upper()
    if answer and answer[0] in ("A", "B"):
        return answer[0]
    return "A"  # fallback


def majority_vote(votes):
    """Return 'A' or 'B' based on majority of votes list."""
    a_count = votes.count("A")
    b_count = votes.count("B")
    return "A" if a_count >= b_count else "B"


def run_judge(model_id):
    slug = model_slug(model_id)
    steering_path = RESULTS_DIR / f"{slug}_steering.json"
    if not steering_path.exists():
        print(f"ERROR: {steering_path} not found. Run experiment_steering.py first.")
        sys.exit(1)

    data = load_json(steering_path)
    if "best_of_n" not in data or not data["best_of_n"]:
        print("ERROR: steering JSON missing 'best_of_n' key. Re-run experiment_steering.py.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    prompts = data["prompts"]
    bon_gens = data["best_of_n"]["generations"]  # prompt -> best completion text
    alphas = [str(a) for a in data["alphas"]]

    results = {
        "model": model_id,
        "JUDGE_MODEL": JUDGE_MODEL,
        "n_judge_repeats": N_JUDGE_REPEATS,
        "win_rate_by_alpha": {},
        "comparisons": {},
    }

    rng = random.Random(SEED)

    for alpha_key in alphas:
        results["comparisons"][alpha_key] = {}
        steered_wins_total = 0
        bon_wins_total = 0

        print(f"\nalpha={alpha_key}")
        steered_gens = data["generations"][alpha_key]  # prompt -> [comp1, comp2, ...]

        for prompt in prompts:
            bon_completion = bon_gens[prompt]
            prompt_wins = {"steered_wins": 0, "bon_wins": 0}

            for steered_completion in steered_gens[prompt]:
                # Randomize which is A vs B to avoid position bias
                if rng.random() < 0.5:
                    a_text, b_text = steered_completion, bon_completion
                    steered_is_a = True
                else:
                    a_text, b_text = bon_completion, steered_completion
                    steered_is_a = False

                votes = []
                for _ in range(N_JUDGE_REPEATS):
                    winner = judge_pair(client, prompt, a_text, b_text)
                    votes.append(winner)
                    time.sleep(0.1)  # avoid hitting rate limits

                final = majority_vote(votes)
                if (final == "A" and steered_is_a) or (final == "B" and not steered_is_a):
                    prompt_wins["steered_wins"] += 1
                    steered_wins_total += 1
                else:
                    prompt_wins["bon_wins"] += 1
                    bon_wins_total += 1

            total = prompt_wins["steered_wins"] + prompt_wins["bon_wins"]
            prompt_wins["win_rate"] = prompt_wins["steered_wins"] / total if total else 0.5
            results["comparisons"][alpha_key][prompt] = prompt_wins
            print(f"  [{prompt[:50]}] steered_wins={prompt_wins['steered_wins']}  "
                  f"bon_wins={prompt_wins['bon_wins']}")

        total_all = steered_wins_total + bon_wins_total
        win_rate = steered_wins_total / total_all if total_all else 0.5
        results["win_rate_by_alpha"][alpha_key] = win_rate
        print(f"  alpha={alpha_key} overall win_rate={win_rate:.3f}")

    output_path = RESULTS_DIR / f"{slug}_steering_judge.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJudge results saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT pairwise humor judge for steering evaluation")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID (used to locate steering JSON)")
    args = parser.parse_args()

    run_judge(model_id=args.model)
