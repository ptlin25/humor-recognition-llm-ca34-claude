"""
Modal app: Gemma 3 12B full probing with MLP probe.

Requires A100-40GB (Gemma 12B in bfloat16 ≈ 24GB; A10G is too tight).
You must accept the model terms at huggingface.co/google/gemma-3-12b-it
before running.

Saves to results/{slug}_results.json (same format as 4B results).

Usage:
    modal run modal/run_larger_model.py::main
    modal run modal/run_larger_model.py::main --model google/gemma-3-12b-it

Estimated cost: ~$8-10 (110-150 min on A100-40GB at $3.70/hr)

Prerequisites:
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxxx
    # Accept model terms at huggingface.co/google/gemma-3-12b-it
"""
import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

app = modal.App("humor-probing-larger-model")

hf_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "accelerate",
        "scikit-learn",
        "numpy",
        "datasets",
        "tqdm",
        "pandas",
    )
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/repo/src")
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "hahackathon"),
        remote_path="/repo/datasets/hahackathon",
    )
)


def _ensure_datasets() -> None:
    """Download HF datasets to the volume on first run, symlink to expected paths."""
    from datasets import DatasetDict, load_dataset  # type: ignore

    VOLUME_DS_DIR = Path("/hf-cache/datasets")
    REPO_DS_DIR = Path("/repo/datasets")
    VOLUME_DS_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        ("short_jokes", "ysharma/short_jokes", None),
        ("one_million_reddit_jokes", "SocialGrep/one-million-reddit-jokes", "train[:80000]"),
    ]

    for name, hf_id, split_spec in specs:
        vol_path = VOLUME_DS_DIR / name
        repo_path = REPO_DS_DIR / name

        if not (vol_path / "train").exists():
            print(f"Downloading {hf_id} ...")
            if split_spec:
                ds = load_dataset(hf_id, split=split_spec)
                DatasetDict({"train": ds}).save_to_disk(str(vol_path))
            else:
                load_dataset(hf_id).save_to_disk(str(vol_path))
            print(f"  {name}: done.")
        else:
            print(f"  {name}: cached.")

        if not repo_path.exists() and not repo_path.is_symlink():
            os.symlink(str(vol_path), str(repo_path))


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={"/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=10800,  # 3 hours — 12B is slow to download + probe
)
def run_larger_model(model_id: str) -> dict:
    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    sys.path.insert(0, "/repo/src")

    print(f"=== Larger Model Probing: {model_id} ===")
    _ensure_datasets()

    from experiment_new_model import run_experiment  # type: ignore

    results = run_experiment(
        model_id=model_id,
        tasks=["easy", "hard", "hahackathon"],
        run_mlp=True,
    )
    print("=== Done. ===")
    return results


@app.local_entrypoint()
def main(model: str = "google/gemma-3-12b-it") -> None:
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    slug = model.replace("/", "-")

    print(f"Running larger model probing: {model}")
    print("NOTE: Ensure you have accepted HF terms for this model.")
    results = run_larger_model.remote(model)

    path = results_dir / f"{slug}_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Saved → {path}")
