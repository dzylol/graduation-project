#!/usr/bin/env python
"""Run ESOL training with different pooling strategies."""

import subprocess
import sys

# Training configurations for pooling ablation
configs = [
    {"name": "ESOL_pooling_mean", "pooling": "mean"},
    {"name": "ESOL_pooling_max", "pooling": "max"},
    {"name": "ESOL_pooling_cls", "pooling": "cls"},
]

base_cmd = [
    "python",
    "train.py",
    "--dataset",
    "ESOL",
    "--data_dir",
    "./dataset/ESOL",
    "--train_file",
    "train.csv",
    "--val_file",
    "val.csv",
    "--test_file",
    "test.csv",
    "--model_type",
    "mamba_ssm",
    "--task_type",
    "regression",
    "--d_model",
    "256",
    "--n_layers",
    "4",
    "--epochs",
    "100",
    "--batch_size",
    "32",
    "--learning_rate",
    "1e-3",
    "--device",
    "cuda",
    "--no_db",
]

for cfg in configs:
    cmd = base_cmd + ["--pooling", cfg["pooling"], "--experiment_name", cfg["name"]]
    print(f"\n{'=' * 60}")
    print(f"Training: {cfg['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, cwd="/home/qfh/graduation-project")
    if result.returncode != 0:
        print(f"FAILED: {cfg['name']}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {cfg['name']}")

print("\n" + "=" * 60)
print("All pooling ablation experiments complete!")
print("=" * 60)
