"""
Training visualization utilities.

Functions for plotting training curves and metric comparisons.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    logs: List[Dict[str, float]],
    save_path: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    title: str = "Training Curves",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        logs: list of epoch logs with metrics
        save_path: path to save the figure
        metrics: list of metrics to plot (default: all found)
        title: plot title
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    if not logs:
        raise ValueError("No training logs provided")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = [log.get("epoch", i + 1) for i, log in enumerate(logs)]

    if metrics is None:
        metrics = []
        for log in logs:
            for key in log.keys():
                if key not in ["epoch"] and key not in metrics:
                    metrics.append(key)

    train_metrics = [m for m in metrics if "train" in m.lower()]
    val_metrics = [m for m in metrics if "val" in m.lower() or "test" in m.lower()]

    ax1 = axes[0]
    for metric in train_metrics:
        values = [log.get(metric, 0) for log in logs]
        ax1.plot(epochs, values, marker="o", label=metric, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for metric in val_metrics:
        values = [log.get(metric, 0) for log in logs]
        ax2.plot(epochs, values, marker="s", label=metric, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    return fig


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "val_loss",
    title: str = "Metric Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Compare a metric across multiple experiments.

    Args:
        results: dict of experiment_name -> metrics dict
        metric: metric to compare
        title: plot title
        save_path: path to save the figure
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    names = list(results.keys())
    values = [results[name].get(metric, 0) for name in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(names)), values, color="steelblue", alpha=0.8)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metric comparison saved to {save_path}")

    return fig


def load_training_logs(
    exp_id: int, db_path: str = "bi_mamba_chem.db"
) -> List[Dict[str, float]]:
    """
    Load training logs from database.

    Args:
        exp_id: experiment ID
        db_path: path to database

    Returns:
        list of epoch logs
    """
    from src.db import get_db

    db = get_db(db_path)
    with db.connect() as conn:
        import sqlite3

        cursor = conn.cursor()
        cursor.execute("SELECT training_logs FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return []


def plot_experiment_training(
    exp_id: int,
    db_path: str = "bi_mamba_chem.db",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves for an experiment from database.

    Args:
        exp_id: experiment ID
        db_path: path to database
        save_path: path to save figure
        title: plot title

    Returns:
        matplotlib Figure object
    """
    logs = load_training_logs(exp_id, db_path)

    if not logs:
        raise ValueError(f"No training logs found for experiment {exp_id}")

    if title is None:
        from src.db import get_db

        db = get_db(db_path)
        with db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM experiments WHERE id = ?", (exp_id,))
            row = cursor.fetchone()
            title = f"Training Curves - {row[0] if row else exp_id}"

    return plot_training_curves(logs, save_path=save_path, title=title)
