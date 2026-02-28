"""
Modal app: Gemma 3 4B week-2 probing run.

Runs all four tasks (easy, hard, hahackathon, hahackathon_random) with
MLP probe enabled. Saves to results/{slug}_week2_results.json so the
existing week-1 results are not overwritten.

Usage:
    modal run modal/run_gemma4b_week2.py::main
    modal run modal/run_gemma4b_week2.py::main --model google/gemma-3-4b-it

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

app = modal.App("humor-probing-gemma4b-week2")

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
    gpu="a10g",
    volumes={"/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
)
def run_week2(model_id: str) -> dict:
    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    sys.path.insert(0, "/repo/src")

    print(f"=== Gemma 4B Week-2 Probing: {model_id} ===")
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
def main(model: str = "google/gemma-3-4b-it") -> None:
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    slug = model.replace("/", "-")

    print(f"Running week-2 probing: {model}")
    results = run_week2.remote(model)

    path = results_dir / f"{slug}_week2_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Saved → {path}")
