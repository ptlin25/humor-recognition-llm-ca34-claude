"""
Modal app for running humor probing experiments on GPU.

Leave your terminal open — results are written to local results/ when done.

Usage:
    modal run modal/run_probing.py::main --model google/gemma-3-1b-it
    modal run modal/run_probing.py::main --model google/gemma-3-4b-it
    modal run modal/run_probing.py::main --model google/gemma-3-4b-it --skip-transfer

Prerequisites:
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxxx  # for gated models (Gemma)
"""
import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

app = modal.App("humor-probing")

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

GPU_TYPE = "a10g"


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
    gpu=GPU_TYPE,
    volumes={"/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
)
def run_probing(model_id: str, skip_transfer: bool = False) -> dict:
    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    sys.path.insert(0, "/repo/src")

    print(f"=== Starting probing: {model_id} ===")
    _ensure_datasets()

    from experiment_new_model import run_experiment as run_new_model  # type: ignore
    output = {"new_model": run_new_model(model_id=model_id)}

    if not skip_transfer:
        from experiment_cross_transfer import run_experiment as run_cross_transfer  # type: ignore
        output["cross_transfer"] = run_cross_transfer(model_id=model_id)

    print("=== Done. ===")
    return output


@app.local_entrypoint()
def main(model: str = "google/gemma-3-4b-it", skip_transfer: bool = False) -> None:
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    slug = model.replace("/", "-")

    print(f"Running: {model}")
    output = run_probing.remote(model, skip_transfer)

    path = results_dir / f"{slug}_results.json"
    path.write_text(json.dumps(output["new_model"], indent=2))
    print(f"Saved → {path}")

    if "cross_transfer" in output:
        path = results_dir / f"{slug}_cross_transfer.json"
        path.write_text(json.dumps(output["cross_transfer"], indent=2))
        print(f"Saved → {path}")
