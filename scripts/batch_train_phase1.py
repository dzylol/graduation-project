#!/usr/bin/env python3
"""
Phase 1 Batch Training Script
Runs all 81 experiments for model structure exploration.

Usage:
    podman run --device nvidia.com/gpu=all -e NVIDIA_VISIBLE_DEVICES=all \
        -v /home/qfh/graduation-project:/workspace \
        localhost/bimamba-train:latest \
        python3.11 /workspace/scripts/batch_train_phase1.py
"""

import itertools
import subprocess
import json
import os
import sys
import time
from datetime import datetime

# Safe batch sizes for 16GB VRAM (RTX 5060 Ti) - conservative for training with backward pass
GPU_BATCH_SIZE_MAP = {
    (128, 2): 16,
    (128, 4): 16,
    (128, 6): 16,
    (256, 2): 16,
    (256, 4): 16,
    (256, 6): 8,  # Reduced due to NaN issues
    (512, 2): 8,  # Reduced due to OOM
    (512, 4): 8,  # Reduced due to OOM
    (512, 6): 8,
}


def get_batch_size(d_model, n_layers):
    return GPU_BATCH_SIZE_MAP.get((d_model, n_layers), 16)


# Phase 1 configurations
POOLING_OPTIONS = ["mean", "max", "cls"]
D_MODEL_OPTIONS = [128, 256, 512]
N_LAYERS_OPTIONS = [2, 4, 6]
DATASETS = ["ESOL", "BBBP", "ClinTox"]
EPOCHS = 100
LEARNING_RATE = 1e-3
DROPOUT = 0.1


def generate_experiments():
    """Generate all experiment configurations"""
    for dataset in DATASETS:
        for pooling, d_model, n_layers in itertools.product(
            POOLING_OPTIONS, D_MODEL_OPTIONS, N_LAYERS_OPTIONS
        ):
            exp = {
                "dataset": dataset,
                "pooling": pooling,
                "d_model": d_model,
                "n_layers": n_layers,
                "batch_size": get_batch_size(d_model, n_layers),
                "learning_rate": LEARNING_RATE,
                "dropout": DROPOUT,
                "epochs": EPOCHS,
                "seed": 42,
            }
            yield exp


def cleanup_gpu():
    """Clean up GPU memory"""
    try:
        subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True, timeout=10)
    except:
        pass
    time.sleep(2)


def run_experiment(exp, experiment_data_dir):
    output_dir = f"{experiment_data_dir}/checkpoints/{exp['dataset']}"
    os.makedirs(output_dir, exist_ok=True)

    db_path = f"{experiment_data_dir}/experiments.db"

    task_type = "regression" if exp["dataset"] == "ESOL" else "classification"

    cmd = [
        "python3.11",
        "train.py",
        "--dataset",
        exp["dataset"],
        "--d_model",
        str(exp["d_model"]),
        "--n_layers",
        str(exp["n_layers"]),
        "--pooling",
        exp["pooling"],
        "--learning_rate",
        str(exp["learning_rate"]),
        "--batch_size",
        str(exp["batch_size"]),
        "--dropout",
        str(exp["dropout"]),
        "--epochs",
        str(exp["epochs"]),
        "--seed",
        str(exp["seed"]),
        "--device",
        "cuda",
        "--output_dir",
        output_dir,
        "--db_path",
        db_path,
        "--task_type",
        task_type,
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace")
    elapsed = time.time() - start_time

    success = result.returncode == 0

    # Extract metrics from output
    val_loss = None
    if success:
        # Try to find validation loss from output
        for line in result.stdout.split("\n"):
            if "验证损失" in line or "val_loss" in line.lower():
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "损失" in p or "loss" in p.lower():
                            val_loss = float(parts[i + 1].replace(":", ""))
                            break
                except:
                    pass

    return {
        "exp": exp,
        "success": success,
        "elapsed": elapsed,
        "val_loss": val_loss,
        "stdout": result.stdout[-500:] if result.stdout else "",
        "stderr": result.stderr[-500:] if result.stderr else "",
    }


def main():
    experiment_data_dir = "/workspace/experiment-data"
    os.makedirs(experiment_data_dir, exist_ok=True)

    experiments = list(generate_experiments())
    total = len(experiments)

    print(f"=" * 60)
    print(f"Phase 1: Model Structure Exploration")
    print(f"Total experiments: {total}")
    print(f"=" * 60)

    results_file = f"{experiment_data_dir}/phase1_results.json"
    results = []

    start_time = time.time()

    for i, exp in enumerate(experiments, 1):
        exp_id = (
            f"{exp['dataset']}_{exp['pooling']}_d{exp['d_model']}_l{exp['n_layers']}"
        )
        print(f"\n[{i}/{total}] Running: {exp_id}")
        print(
            f"    d_model={exp['d_model']}, n_layers={exp['n_layers']}, "
            f"pooling={exp['pooling']}, batch_size={exp['batch_size']}"
        )

        result = run_experiment(exp, experiment_data_dir)
        results.append(result)

        if result["success"]:
            print(f"    ✅ Success ({result['elapsed']:.1f}s)")
            if result["val_loss"] is not None:
                print(f"    val_loss: {result['val_loss']:.6f}")
        else:
            print(f"    ❌ Failed ({result['elapsed']:.1f}s)")
            print(f"    stderr: {result['stderr'][-200:]}")
            cleanup_gpu()

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'=' * 60}")
    print(f"Phase 1 Complete!")
    print(f"Successful: {successful}/{total}")
    print(f"Total time: {total_time / 3600:.1f} hours")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 60}")

    # Save final summary
    summary = {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600,
    }
    with open(f"{experiment_data_dir}/phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
