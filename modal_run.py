"""Modal app for humor-recognition LLM experiments.

Quickstart
----------
1. pip install modal
2. modal setup                                          # authenticate
3. modal run modal_run.py                              # full pipeline
   modal run modal_run.py::main --experiment activations  # single experiment

Available --experiment values:
  all | data | activations | lora | sentiment | hard | pythia | visualize

Results are stored in Modal volume "humor-experiments".
Download with:
  modal volume get humor-experiments results/ ./results/
"""
import io
import zipfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App / infrastructure
# ---------------------------------------------------------------------------

LOCAL_ROOT = Path(__file__).parent

app = modal.App("humor-recognition")

# Persistent volume: datasets, HF model cache, experiment results
volume = modal.Volume.from_name("humor-experiments", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "tqdm",
        "datasets",
        "transformers",
        "accelerate",
        "huggingface-hub",
    )
    .add_local_dir("src", remote_path="/project/src")
)

GPU = "T4"  # sufficient for GPT-2 and Pythia-410M


# ---------------------------------------------------------------------------
# Helper: wire volume dirs into the project tree at runtime
# ---------------------------------------------------------------------------

def _setup():
    """Symlink /data/{datasets,results} into /project/ as expected by experiment scripts.

    Experiment scripts compute PROJECT_ROOT = Path(__file__).parent.parent = /project,
    then read from /project/datasets/ and write to /project/results/. Symlinking those
    into the volume makes all reads/writes persistent across runs.
    """
    import os
    import sys
    from pathlib import Path

    os.makedirs(f"{VOLUME_PATH}/datasets", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/results/plots", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/hf_cache", exist_ok=True)

    for name in ("datasets", "results"):
        link = Path(f"/project/{name}")
        if not link.exists() and not link.is_symlink():
            link.symlink_to(f"{VOLUME_PATH}/{name}")

    # Redirect HF downloads to the volume so models persist across runs
    os.environ["HF_HOME"] = f"{VOLUME_PATH}/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{VOLUME_PATH}/hf_cache"
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    sys.path.insert(0, "/project/src")


# ---------------------------------------------------------------------------
# Dataset upload (run once, or when local datasets change)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600)
def upload_datasets(zip_bytes: bytes):
    """Unpack a zip of the local datasets/ directory into the volume."""
    from pathlib import Path

    dest_root = Path(VOLUME_PATH)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for name in names:
            dest = dest_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
    volume.commit()
    print(f"Uploaded {len(names)} files to volume.")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600)
def run_data_preparation():
    _setup()
    import json
    from data_preparation import prepare_dataset

    data = prepare_dataset()
    out = f"{VOLUME_PATH}/results/prepared_data.json"
    with open(out, "w") as f:
        json.dump(data, f)
    volume.commit()
    print(f"Saved {out}")


@app.function(
    image=image, gpu=GPU, volumes={VOLUME_PATH: volume},
timeout=7200,
)
def run_activations():
    _setup()
    from experiment_activations import run_experiment
    run_experiment()
    volume.commit()


@app.function(
    image=image, gpu=GPU, volumes={VOLUME_PATH: volume},
timeout=14400,
)
def run_lora():
    _setup()
    from experiment_lora import run_lora_experiment
    run_lora_experiment()
    volume.commit()


@app.function(
    image=image, gpu=GPU, volumes={VOLUME_PATH: volume},
timeout=7200,
)
def run_sentiment():
    _setup()
    from experiment_sentiment_comparison import run_sentiment_comparison
    run_sentiment_comparison()
    volume.commit()


@app.function(
    image=image, gpu=GPU, volumes={VOLUME_PATH: volume},
timeout=7200,
)
def run_hard():
    _setup()
    from experiment_hard_dataset import run_hard_experiments
    run_hard_experiments()
    volume.commit()


@app.function(
    image=image, gpu=GPU, volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def run_pythia():
    _setup()
    from experiment_pythia import run_pythia_experiment
    run_pythia_experiment()
    volume.commit()


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600)
def run_visualize():
    _setup()
    import visualize_all
    visualize_all.plot_figure1_main_results()
    visualize_all.plot_figure2_hard_tasks()
    visualize_all.plot_figure3_comparison()
    visualize_all.plot_figure4_pythia()
    visualize_all.plot_figure5_lora_detail()
    volume.commit()
    print("All figures saved to results/plots/.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(experiment: str = "all", skip_upload: bool = False):
    """Run the full experiment pipeline (or a single step).

    Args:
        experiment:   all | data | activations | lora | sentiment |
                      hard | pythia | visualize
        skip_upload:  Skip uploading datasets/ to the volume.
                      Safe to use after the first run.
    """
    if not skip_upload:
        datasets_dir = LOCAL_ROOT / "datasets"
        if not datasets_dir.exists() or not any(datasets_dir.iterdir()):
            print(
                "WARNING: datasets/ is empty — skipping upload.\n"
                "  Populate datasets/short_jokes/ and datasets/hahackathon/ before running."
            )
        else:
            print("Zipping and uploading datasets/ to volume...")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in datasets_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(LOCAL_ROOT))
            buf.seek(0)
            upload_datasets.remote(buf.read())
            print("Upload complete.\n")

    steps = {
        "data":        run_data_preparation,
        "activations": run_activations,
        "lora":        run_lora,
        "sentiment":   run_sentiment,
        "hard":        run_hard,
        "pythia":      run_pythia,
        "visualize":   run_visualize,
    }
    pipeline = list(steps.keys()) if experiment == "all" else [experiment]

    for step in pipeline:
        if step not in steps:
            print(f"Unknown experiment '{step}'. Choose from: {list(steps)}")
            return
        print(f"--- Running {step} ---")
        steps[step].remote()

    print(
        "\nDone! Download results with:\n"
        "  modal volume get humor-experiments results/ ./results/"
    )
