"""
Evaluate activation steering quality using GPT as a pairwise humor judge.

Uses the OpenAI Batch API to submit all comparisons at once (50% cost reduction,
no rate limits). Polls until complete, then reconstructs win rates.

For each alpha, compares steered completions head-to-head against the best-of-N
baseline. Also judges random-direction completions if present in the steering JSON.

Requires:
  - results/{slug}_steering.json  (from experiment_steering.py, must include best_of_n)
  - OPENAI_API_KEY environment variable

Usage:
    python src/evaluate_steering_judge.py --model google/gemma-3-4b-it
"""
import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42
JUDGE_MODEL = "gpt-5.2"
POLL_INTERVAL = 30  # seconds between batch status checks

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


def build_requests(data, prompts, bon_gens, alphas, rng):
    """
    Build all batch request dicts and record A/B assignments.

    custom_id format: "{kind}|{alpha_key}|{prompt_idx}|{comp_idx}"
      kind = "probe" (LR steering direction) or "rand" (random direction)

    Returns:
      requests  - list of dicts ready to serialize as JSONL
      assignments - custom_id -> bool (True if the non-BoN completion is A)
    """
    requests = []
    assignments = {}

    def add_request(kind, alpha_key, pi, ci, completion, bon_completion, prompt):
        is_a = rng.random() < 0.5
        a_text = completion if is_a else bon_completion
        b_text = bon_completion if is_a else completion
        custom_id = f"{kind}|{alpha_key}|{pi}|{ci}"
        assignments[custom_id] = is_a
        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": JUDGE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        prompt=prompt,
                        completion_a=a_text,
                        completion_b=b_text,
                    ),
                }],
            },
        })

    for alpha_key in alphas:
        steered_gens = data["generations"][alpha_key]
        for pi, prompt in enumerate(prompts):
            for ci, completion in enumerate(steered_gens[prompt]):
                add_request("probe", alpha_key, pi, ci, completion, bon_gens[prompt], prompt)

    if "random_direction" in data:
        for alpha_key in alphas:
            rand_gens = data["random_direction"]["generations"][alpha_key]
            for pi, prompt in enumerate(prompts):
                for ci, completion in enumerate(rand_gens[prompt]):
                    add_request("rand", alpha_key, pi, ci, completion, bon_gens[prompt], prompt)

    return requests, assignments


def submit_batch(client, requests):
    """Upload JSONL and submit batch. Returns batch object."""
    jsonl_bytes = "\n".join(json.dumps(r) for r in requests).encode()
    file_obj = client.files.create(
        file=("batch_requests.jsonl", io.BytesIO(jsonl_bytes), "application/json"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id}  ({len(requests)} requests)")
    return batch


def poll_batch(client, batch_id):
    """Poll until batch completes. Returns output file content as string."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.status}  "
              f"completed={counts.completed}/{counts.total}  "
              f"failed={counts.failed}")
        if batch.status == "completed":
            content = client.files.content(batch.output_file_id).text
            return content
        if batch.status in ("failed", "expired", "cancelled"):
            print(f"ERROR: batch {batch_id} ended with status={batch.status}")
            sys.exit(1)
        time.sleep(POLL_INTERVAL)


def parse_output(output_text, assignments, prompts, alphas):
    """
    Parse batch output JSONL. Returns:
      probe_wins[alpha_key][pi] = {"steered_wins": int, "bon_wins": int}
      rand_wins[alpha_key][pi]  = {"steered_wins": int, "bon_wins": int}
    """
    probe_wins = {a: {pi: {"steered_wins": 0, "bon_wins": 0} for pi in range(len(prompts))}
                  for a in alphas}
    rand_wins = {a: {pi: {"steered_wins": 0, "bon_wins": 0} for pi in range(len(prompts))}
                 for a in alphas}

    for line in output_text.strip().splitlines():
        result = json.loads(line)
        custom_id = result["custom_id"]
        if result.get("error"):
            print(f"  WARNING: request {custom_id} failed: {result['error']}")
            continue

        content = result["response"]["body"]["choices"][0]["message"]["content"]
        answer = content.strip().upper()
        answer = answer[0] if answer and answer[0] in ("A", "B") else "A"

        parts = custom_id.split("|")
        kind, alpha_key, pi = parts[0], parts[1], parts[2]
        pi = int(pi)
        is_a = assignments[custom_id]
        won = (answer == "A" and is_a) or (answer == "B" and not is_a)

        bucket = probe_wins if kind == "probe" else rand_wins
        if won:
            bucket[alpha_key][pi]["steered_wins"] += 1
        else:
            bucket[alpha_key][pi]["bon_wins"] += 1

    return probe_wins, rand_wins


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
    bon_gens = data["best_of_n"]["generations"]
    alphas = [str(a) for a in data["alphas"]]
    rng = random.Random(SEED)

    # Build and submit batch
    requests, assignments = build_requests(data, prompts, bon_gens, alphas, rng)
    batch = submit_batch(client, requests)

    # Poll for completion
    print(f"\nPolling every {POLL_INTERVAL}s...")
    output_text = poll_batch(client, batch.id)

    # Parse results
    probe_wins, rand_wins = parse_output(output_text, assignments, prompts, alphas)

    # Aggregate into final results dict
    results = {
        "model": model_id,
        "JUDGE_MODEL": JUDGE_MODEL,
        "batch_id": batch.id,
        "win_rate_by_alpha": {},
        "comparisons": {},
        "random_direction_win_rate_by_alpha": {},
        "random_direction_comparisons": {},
    }

    for alpha_key in alphas:
        results["comparisons"][alpha_key] = {}
        total_steered = total_bon = 0
        for pi, prompt in enumerate(prompts):
            pw = probe_wins[alpha_key][pi]
            total = pw["steered_wins"] + pw["bon_wins"]
            pw["win_rate"] = pw["steered_wins"] / total if total else 0.5
            results["comparisons"][alpha_key][prompt] = pw
            total_steered += pw["steered_wins"]
            total_bon += pw["bon_wins"]
            print(f"  alpha={alpha_key} [{prompt[:50]}] "
                  f"steered={pw['steered_wins']} bon={pw['bon_wins']}")
        total_all = total_steered + total_bon
        wr = total_steered / total_all if total_all else 0.5
        results["win_rate_by_alpha"][alpha_key] = wr
        print(f"  alpha={alpha_key} probe win_rate={wr:.3f}")

    if "random_direction" in data:
        for alpha_key in alphas:
            results["random_direction_comparisons"][alpha_key] = {}
            total_rand = total_bon = 0
            for pi, prompt in enumerate(prompts):
                rw = rand_wins[alpha_key][pi]
                total = rw["steered_wins"] + rw["bon_wins"]
                rw["win_rate"] = rw["steered_wins"] / total if total else 0.5
                results["random_direction_comparisons"][alpha_key][prompt] = rw
                total_rand += rw["steered_wins"]
                total_bon += rw["bon_wins"]
            total_all = total_rand + total_bon
            wr = total_rand / total_all if total_all else 0.5
            results["random_direction_win_rate_by_alpha"][alpha_key] = wr
            print(f"  alpha={alpha_key} random_direction win_rate={wr:.3f}")

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
