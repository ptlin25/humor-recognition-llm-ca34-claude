"""
Modal app: Activation steering experiment for humor.

Does NOT need _ensure_datasets() — only uses HaHackathon CSVs (baked into image).
Saves to results/{slug}_steering.json.

Usage:
    modal run modal/run_steering.py::main
    modal run modal/run_steering.py::main --model google/gemma-3-4b-it

Estimated cost: ~$1 (~45 min on A10G at $1.10/hr)

Prerequisites:
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxxx
"""
import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

app = modal.App("humor-activation-steering")

hf_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "accelerate",
        "scikit-learn",
        "numpy",
        "tqdm",
        "pandas",
    )
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/repo/src")
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "hahackathon"),
        remote_path="/repo/datasets/hahackathon",
    )
)


@app.function(
    image=image,
    gpu="a10g",
    volumes={"/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=5400,  # 90 minutes
)
def run_steering(model_id: str) -> dict:
    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    sys.path.insert(0, "/repo/src")

    print(f"=== Activation Steering: {model_id} ===")

    from experiment_steering import run_steering_experiment  # type: ignore

    results = run_steering_experiment(model_id=model_id, mock=False)
    print("=== Done. ===")
    return results


@app.local_entrypoint()
def main(model: str = "google/gemma-3-4b-it") -> None:
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    slug = model.replace("/", "-")

    print(f"Running activation steering: {model}")
    results = run_steering.remote(model)

    path = results_dir / f"{slug}_steering.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Saved → {path}")
