"""
Experiment E: Activation Steering for humor.

Extracts the LR probe direction at the best HaHackathon layer (layer 18),
registers a forward hook on that layer, and generates text with varying
steering strengths (alpha). Scores completions with the trained probe.

Usage:
    # Local pipeline test (no GPU):
    python src/experiment_steering.py --mock

    # Real run (on Modal — see modal/run_steering.py):
    python src/experiment_steering.py --model google/gemma-3-4b-it
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Layer indices (1-indexed with embedding = layer 0, matching probe_by_layer keys)
STEERING_LAYER = 18          # index into outputs.hidden_states
HOOK_LAYER_IDX = STEERING_LAYER - 1  # 0-indexed into model.model.layers

ALPHAS = [-20, -10, 0, 10, 20, 30]
N_COMPLETIONS = 5
N_BON = 10  # completions to sample for best-of-N baseline

NEUTRAL_PROMPTS = [
    "Today I went to the store and",
    "The weather outside is",
    "Scientists have recently discovered that",
    "My coworker told me that",
    "The thing about Mondays is",
    "On my way to work this morning,",
    "The most surprising thing about getting older is",
    "My doctor told me I should",
    "At the family dinner last night,",
    "The problem with technology these days is",
    "My neighbor knocked on my door and",
    "The last time I tried to cook,",
    "Everyone at the office was talking about",
    "I finally decided to start exercising and",
    "The thing nobody tells you about parenthood is",
    "Last weekend I tried something new and",
    "My phone battery died right when",
    "The instructions said it was simple, but",
    "I asked my friend for advice and",
    "When I opened my email this morning,",
    "The meeting was supposed to take thirty minutes but",
    "I haven't slept well since",
    "The grocery store was completely out of",
    "My cat stared at the wall for an hour and",
    "I tried to explain the situation, but",
    "According to the internet,",
    "The last time I flew on an airplane,",
    "I thought I was being productive until",
    "The worst part about winter is",
    "I signed up for a class and",
]


def model_slug(model_id):
    return model_id.replace("/", "-")


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------

def _find_layers(model):
    """
    Locate the transformer decoder layer list, handling Gemma 3's nested structure.

    Gemma3ForCausalLM: model.model is Gemma3Model (multimodal wrapper),
    which stores the text transformer at model.model.language_model with .layers.
    Fallback covers plain architectures (GPT-2, Gemma 2, etc.) where model.model.layers exists.
    """
    inner = model.model
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model.layers
    if hasattr(inner, "layers"):
        return inner.layers
    raise AttributeError(
        f"Cannot find transformer layers in {type(inner).__name__}. "
        f"Available attrs: {[k for k in inner._modules.keys()]}"
    )


def make_steering_hook(direction_tensor, alpha):
    """
    Forward hook for model.model.layers[HOOK_LAYER_IDX].
    Adds alpha * direction to the hidden state at every token position.

    direction_tensor: torch.Tensor (hidden_dim,), unit vector, on same device as model.
    alpha: float — positive steers toward humor, negative steers away.
    """
    def hook(module, module_input, module_output):
        if isinstance(module_output, tuple):
            hidden_states = module_output[0]
            rest = module_output[1:]
        else:
            hidden_states = module_output
            rest = ()
        # shape: (batch, seq_len, hidden_dim)
        steering = alpha * direction_tensor.to(hidden_states.device, dtype=hidden_states.dtype)
        steered = hidden_states + steering.unsqueeze(0).unsqueeze(0)
        if rest or isinstance(module_output, tuple):
            return (steered,) + rest
        return steered
    return hook


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def fit_probe(train_acts, train_labels):
    """Fit StandardScaler + LogisticRegression. Returns (scaler, lr)."""
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_acts)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(train_s, train_labels)
    return scaler, lr


def get_lr_direction_from_probe(scaler, lr):
    """
    Back-project LR coef from standardized space to original activation space.
    Returns unit vector (hidden_dim,).
    """
    direction = lr.coef_[0] / scaler.scale_
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


# ---------------------------------------------------------------------------
# Activation extraction (single layer, no PCA)
# ---------------------------------------------------------------------------

def extract_layer_acts(model, tokenizer, texts, layer_idx, batch_size=8, max_length=128):
    """
    Extract last-token hidden states at layer_idx only.
    Returns np.array of shape (N, hidden_dim).
    """
    device = next(model.parameters()).device
    model.eval()
    all_acts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]  # (B, seq, hidden)
        last_pos = inputs["attention_mask"].sum(dim=1) - 1
        for b in range(hidden.shape[0]):
            pos = last_pos[b].item()
            all_acts.append(hidden[b, pos, :].cpu().to(torch.float32).numpy())

    return np.stack(all_acts)


# ---------------------------------------------------------------------------
# Score completions with trained probe
# ---------------------------------------------------------------------------

def score_texts(model, tokenizer, texts, scaler, lr, batch_size=8):
    """
    Extract layer STEERING_LAYER activations (no hook) and return LR predictions.
    Returns list of int (0 or 1) per text.
    """
    acts = extract_layer_acts(model, tokenizer, texts, STEERING_LAYER, batch_size=batch_size)
    acts_s = scaler.transform(acts)
    preds = lr.predict(acts_s)
    return preds.tolist()


def score_texts_proba(model, tokenizer, texts, scaler, lr, batch_size=8):
    """
    Extract layer STEERING_LAYER activations (no hook) and return soft humor probabilities.
    Returns list of float per text.
    """
    acts = extract_layer_acts(model, tokenizer, texts, STEERING_LAYER, batch_size=batch_size)
    acts_s = scaler.transform(acts)
    return lr.predict_proba(acts_s)[:, 1].tolist()


# ---------------------------------------------------------------------------
# Perplexity computation (no hook)
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, prompts, completions, batch_size=4):
    """
    Compute perplexity of each completion conditioned on its prompt.

    Labels for prompt tokens are set to -100 so they are excluded from the loss.
    Returns a list of float (one PPL per completion).
    """
    device = next(model.parameters()).device
    model.eval()
    ppls = []

    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion
        enc_full = tokenizer(full_text, return_tensors="pt").to(device)
        enc_prompt = tokenizer(prompt, return_tensors="pt")
        prompt_len = enc_prompt["input_ids"].shape[1]

        labels = enc_full["input_ids"].clone()
        labels[:, :prompt_len] = -100  # ignore prompt tokens in loss

        with torch.no_grad():
            outputs = model(**enc_full, labels=labels)
        ppl = torch.exp(outputs.loss).item()
        ppls.append(ppl)

    return ppls


# ---------------------------------------------------------------------------
# Main steering experiment
# ---------------------------------------------------------------------------

def run_steering_experiment(model_id, mock=False):
    """
    Run activation steering experiment.

    For each alpha in ALPHAS and each prompt in NEUTRAL_PROMPTS:
      - Register hook on model.model.layers[HOOK_LAYER_IDX]
      - Generate N_COMPLETIONS completions
      - Remove hook
      - Score completions with trained probe (no hook)

    Returns results dict (also saved to results/{slug}_steering.json).
    """
    print("=" * 60)
    print(f"EXPERIMENT E: Activation Steering — {model_id}")
    print(f"  Steering layer: {STEERING_LAYER} (hook on layers[{HOOK_LAYER_IDX}])")
    print(f"  Alphas: {ALPHAS}  |  Mock: {mock}  |  Device: {DEVICE}")
    print("=" * 60)

    results = {
        "model": model_id,
        "steering_layer": STEERING_LAYER,
        "steering_direction": "lr_coef",
        "alphas": ALPHAS,
        "direction_norm": 1.0,
        "prompts": NEUTRAL_PROMPTS,
        "generations": {},
        "probe_humor_scores": {},
        "mean_probe_score_by_alpha": {},
        "perplexity_by_alpha": {},
        "mean_perplexity_by_alpha": {},
        "best_of_n": {},
    }

    if mock:
        rng = np.random.default_rng(SEED)
        for alpha in ALPHAS:
            alpha_key = str(alpha)
            results["generations"][alpha_key] = {}
            results["probe_humor_scores"][alpha_key] = {}
            results["perplexity_by_alpha"][alpha_key] = {}
            for prompt in NEUTRAL_PROMPTS:
                completions = [
                    f"[mock alpha={alpha}] completion {i+1} for: {prompt[:30]}"
                    for i in range(N_COMPLETIONS)
                ]
                # Simulate: higher alpha → more likely to score as humor, higher PPL
                p_humor = float(np.clip(0.5 + alpha * 0.01, 0.0, 1.0))
                scores = [int(rng.random() < p_humor) for _ in range(N_COMPLETIONS)]
                base_ppl = 80.0 + abs(alpha) * 2.0
                ppls = [float(rng.normal(base_ppl, 5.0)) for _ in range(N_COMPLETIONS)]
                results["generations"][alpha_key][prompt] = completions
                results["probe_humor_scores"][alpha_key][prompt] = scores
                results["perplexity_by_alpha"][alpha_key][prompt] = ppls
            all_scores = [
                s for p in NEUTRAL_PROMPTS
                for s in results["probe_humor_scores"][alpha_key][p]
            ]
            all_ppls = [
                v for p in NEUTRAL_PROMPTS
                for v in results["perplexity_by_alpha"][alpha_key][p]
            ]
            results["mean_probe_score_by_alpha"][alpha_key] = float(np.mean(all_scores))
            results["mean_perplexity_by_alpha"][alpha_key] = float(np.mean(all_ppls))
        # Mock best-of-N
        bon_gens, bon_ppls = {}, {}
        for prompt in NEUTRAL_PROMPTS:
            bon_gens[prompt] = f"[mock bon] best completion for: {prompt[:30]}"
            bon_ppls[prompt] = float(rng.normal(78.0, 4.0))
        results["best_of_n"] = {
            "n": N_BON,
            "generations": bon_gens,
            "perplexity": bon_ppls,
            "mean_perplexity": float(np.mean(list(bon_ppls.values()))),
        }
        print("Mock run complete.")
        _save_results(model_id, results)
        return results

    # --- Real run ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    model.eval()

    # --- Load HaHackathon training data and fit probe ---
    print("\nLoading HaHackathon data and fitting probe at layer", STEERING_LAYER)
    from data_hahackathon import load_hahackathon
    data = load_hahackathon(binary=True)
    train_texts = data["train"]["texts"]
    train_labels = data["train"]["labels"]

    t0 = time.time()
    train_acts = extract_layer_acts(model, tokenizer, train_texts, STEERING_LAYER)
    print(f"  Activation extraction: {time.time() - t0:.1f}s  shape={train_acts.shape}")

    scaler, lr = fit_probe(train_acts, train_labels)
    direction = get_lr_direction_from_probe(scaler, lr)
    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    train_preds = lr.predict(scaler.transform(train_acts))
    train_acc = float((train_preds == np.array(train_labels)).mean())
    print(f"  Probe train accuracy at layer {STEERING_LAYER}: {train_acc:.3f}")

    # --- Generation loop ---
    print(f"\nGenerating {len(ALPHAS)} × {len(NEUTRAL_PROMPTS)} × {N_COMPLETIONS} completions...")

    for alpha in ALPHAS:
        alpha_key = str(alpha)
        results["generations"][alpha_key] = {}
        results["probe_humor_scores"][alpha_key] = {}
        results["perplexity_by_alpha"][alpha_key] = {}
        print(f"\n  alpha={alpha}")

        for prompt in NEUTRAL_PROMPTS:
            # Register steering hook
            hook_handle = _find_layers(model)[HOOK_LAYER_IDX].register_forward_hook(
                make_steering_hook(direction_tensor, alpha)
            )

            # Generate N_COMPLETIONS in one batch
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids_batch = inputs["input_ids"].repeat(N_COMPLETIONS, 1)
            attn_mask_batch = inputs["attention_mask"].repeat(N_COMPLETIONS, 1)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attn_mask_batch,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            hook_handle.remove()

            # Decode completions
            completions = [
                tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
                for i in range(N_COMPLETIONS)
            ]
            full_texts = [
                tokenizer.decode(output_ids[i], skip_special_tokens=True)
                for i in range(N_COMPLETIONS)
            ]

            # Score completions with probe (no hook)
            scores = score_texts(model, tokenizer, full_texts, scaler, lr)

            # Perplexity of completions (no hook)
            ppls = compute_perplexity(model, tokenizer, [prompt] * N_COMPLETIONS, completions)

            results["generations"][alpha_key][prompt] = completions
            results["probe_humor_scores"][alpha_key][prompt] = scores
            results["perplexity_by_alpha"][alpha_key][prompt] = ppls

            humor_frac = sum(scores) / len(scores)
            print(f"    [{prompt[:40]}] humor_frac={humor_frac:.2f}  ppl={np.mean(ppls):.1f}"
                  f"| {completions[0][:60]!r}...")

        # Mean probe score / perplexity across all prompts for this alpha
        all_scores = [
            s for p in NEUTRAL_PROMPTS
            for s in results["probe_humor_scores"][alpha_key][p]
        ]
        all_ppls = [
            v for p in NEUTRAL_PROMPTS
            for v in results["perplexity_by_alpha"][alpha_key][p]
        ]
        results["mean_probe_score_by_alpha"][alpha_key] = float(np.mean(all_scores))
        results["mean_perplexity_by_alpha"][alpha_key] = float(np.mean(all_ppls))
        print(f"  alpha={alpha}  mean_probe_score={results['mean_probe_score_by_alpha'][alpha_key]:.3f}"
              f"  mean_ppl={results['mean_perplexity_by_alpha'][alpha_key]:.1f}")

    # --- Best-of-N baseline (no hook, alpha=0) ---
    print(f"\nGenerating best-of-N baseline (N={N_BON}, no hook)...")
    bon_gens, bon_ppls = {}, {}
    for prompt in NEUTRAL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids_batch = inputs["input_ids"].repeat(N_BON, 1)
        attn_mask_batch = inputs["attention_mask"].repeat(N_BON, 1)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attn_mask_batch,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        bon_completions = [
            tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
            for i in range(N_BON)
        ]
        bon_full_texts = [
            tokenizer.decode(output_ids[i], skip_special_tokens=True)
            for i in range(N_BON)
        ]

        # Rank by soft probe score; pick the best
        proba_scores = score_texts_proba(model, tokenizer, bon_full_texts, scaler, lr)
        best_idx = int(np.argmax(proba_scores))
        best_completion = bon_completions[best_idx]

        # Perplexity of selected completion
        best_ppl = compute_perplexity(model, tokenizer, [prompt], [best_completion])[0]
        bon_gens[prompt] = best_completion
        bon_ppls[prompt] = best_ppl
        print(f"  [{prompt[:40]}] bon_ppl={best_ppl:.1f}  {best_completion[:60]!r}...")

    results["best_of_n"] = {
        "n": N_BON,
        "generations": bon_gens,
        "perplexity": bon_ppls,
        "mean_perplexity": float(np.mean(list(bon_ppls.values()))),
    }

    # --- Random direction baseline (same alphas, random unit vector) ---
    print("\nGenerating random-direction baseline...")
    rng_rand = np.random.default_rng(SEED + 1)
    random_vec = rng_rand.standard_normal(direction.shape)
    random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-8)
    random_direction_tensor = torch.tensor(random_vec, dtype=torch.float32)

    results["random_direction"] = {
        "generations": {},
        "perplexity_by_alpha": {},
        "mean_perplexity_by_alpha": {},
    }

    for alpha in ALPHAS:
        alpha_key = str(alpha)
        results["random_direction"]["generations"][alpha_key] = {}
        results["random_direction"]["perplexity_by_alpha"][alpha_key] = {}
        print(f"\n  alpha={alpha} (random direction)")

        for prompt in NEUTRAL_PROMPTS:
            hook_handle = _find_layers(model)[HOOK_LAYER_IDX].register_forward_hook(
                make_steering_hook(random_direction_tensor, alpha)
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids_batch = inputs["input_ids"].repeat(N_COMPLETIONS, 1)
            attn_mask_batch = inputs["attention_mask"].repeat(N_COMPLETIONS, 1)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attn_mask_batch,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            hook_handle.remove()

            completions = [
                tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
                for i in range(N_COMPLETIONS)
            ]
            ppls = compute_perplexity(model, tokenizer, [prompt] * N_COMPLETIONS, completions)

            results["random_direction"]["generations"][alpha_key][prompt] = completions
            results["random_direction"]["perplexity_by_alpha"][alpha_key][prompt] = ppls
            print(f"    [{prompt[:40]}] ppl={np.mean(ppls):.1f}")

        all_ppls = [
            v for p in NEUTRAL_PROMPTS
            for v in results["random_direction"]["perplexity_by_alpha"][alpha_key][p]
        ]
        results["random_direction"]["mean_perplexity_by_alpha"][alpha_key] = float(np.mean(all_ppls))

    print("\nSteering experiment complete.")
    print("Alpha → mean probe score / mean perplexity:")
    for alpha in ALPHAS:
        print(f"  {alpha:4d}: probe={results['mean_probe_score_by_alpha'][str(alpha)]:.3f}"
              f"  ppl={results['mean_perplexity_by_alpha'][str(alpha)]:.1f}")
    print(f"  BoN (N={N_BON}): ppl={results['best_of_n']['mean_perplexity']:.1f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _save_results(model_id, results)
    return results


def _save_results(model_id, results):
    slug = model_slug(model_id)
    output_path = PROJECT_ROOT / "results" / f"{slug}_steering.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation steering experiment")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode (no GPU, for pipeline testing)")
    args = parser.parse_args()

    run_steering_experiment(model_id=args.model, mock=args.mock)
