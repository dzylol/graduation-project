"""
Experiment dashboard utilities.

Functions for creating comprehensive experiment comparison dashboards.
"""

from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os


def create_experiment_dashboard(
    experiments: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    Create a comprehensive dashboard comparing multiple experiments.

    Args:
        experiments: list of experiment dicts with metrics and training logs
        save_path: path to save the dashboard
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    n_exps = len(experiments)
    if n_exps == 0:
        raise ValueError("No experiments to display")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, min(3, n_exps + 1), figure=fig)

    fig.suptitle("Experiment Comparison Dashboard", fontsize=16, fontweight="bold")

    ax_summary = fig.add_subplot(gs[0, :2])
    plot_summary_table(ax_summary, experiments)

    ax_metrics = fig.add_subplot(gs[0, 2:])
    plot_metrics_comparison(ax_metrics, experiments)

    ax_curves = fig.add_subplot(gs[1, :])
    plot_combined_training_curves(ax_curves, experiments)

    ax_tasks = fig.add_subplot(gs[2, :])
    plot_task_comparison(ax_tasks, experiments)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")

    return fig


def plot_summary_table(ax, experiments: List[Dict[str, Any]]):
    """Plot a summary table of experiment configurations."""
    ax.axis("off")

    col_labels = ["Name", "Dataset", "Epochs", "Status"]
    table_data = []

    for exp in experiments:
        row = [
            exp.get("name", "N/A")[:20],
            exp.get("dataset", "N/A")[:15],
            str(exp.get("best_epoch", "N/A")),
            exp.get("status", "N/A"),
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.2, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E2F3")

    ax.set_title("Experiment Summary", fontsize=12, fontweight="bold", pad=20)


def plot_metrics_comparison(ax, experiments: List[Dict[str, Any]]):
    """Plot metrics comparison across experiments."""
    names = [exp.get("name", f"Exp{i}")[:15] for i, exp in enumerate(experiments)]
    metrics = experiments[0].get("metrics", {}) if experiments else {}

    metric_names = ["mae", "mse", "rmse", "r2"]
    available_metrics = [m for m in metric_names if m in metrics]

    if not available_metrics:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        ax.axis("off")
        return

    x = range(len(names))
    width = 0.8 / len(available_metrics)

    for i, metric in enumerate(available_metrics):
        values = []
        for exp in experiments:
            val = exp.get("metrics", {}).get(metric, 0)
            values.append(val)

        offset = (i - len(available_metrics) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width, label=metric.upper())

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Metrics Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_combined_training_curves(ax, experiments: List[Dict[str, Any]]):
    """Plot training curves for multiple experiments."""
    colors = plt.cm.tab10.colors

    for i, exp in enumerate(experiments):
        logs = exp.get("training_logs", [])
        if not logs:
            continue

        name = exp.get("name", f"Exp{i}")[:15]
        epochs = [log.get("epoch", j + 1) for j, log in enumerate(logs)]

        if i == 0:
            train_metric = "train_loss"
            val_metric = "val_loss"
        else:
            train_metric = "train_loss"
            val_metric = "val_loss"

        train_values = [log.get(train_metric, 0) for log in logs]
        val_values = [log.get(val_metric, 0) for log in logs]

        color = colors[i % len(colors)]
        ax.plot(
            epochs,
            train_values,
            linestyle="-",
            color=color,
            alpha=0.7,
            label=f"{name} (train)",
        )
        ax.plot(
            epochs,
            val_values,
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"{name} (val)",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_task_comparison(ax, experiments: List[Dict[str, Any]]):
    """Plot task-level comparison for multi-task experiments."""
    all_tasks = set()
    for exp in experiments:
        tasks = exp.get("tasks", [])
        all_tasks.update(tasks)

    if not all_tasks:
        ax.text(0.5, 0.5, "No tasks to display", ha="center", va="center")
        ax.axis("off")
        return

    tasks = sorted(list(all_tasks))
    n_tasks = len(tasks)
    n_exps = len(experiments)

    x = range(n_exps)
    width = 0.8 / n_tasks

    for i, task in enumerate(tasks):
        values = []
        for exp in experiments:
            val = exp.get("metrics", {}).get(f"{task}_mae", 0)
            values.append(val)

        offset = (i - n_tasks / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width, label=task)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [exp.get("name", f"Exp{i}")[:10] for i, exp in enumerate(experiments)],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("MAE")
    ax.set_title("Task-wise MAE Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def load_experiments_from_db(
    exp_ids: List[int],
    db_path: str = "bi_mamba_chem.db",
) -> List[Dict[str, Any]]:
    """
    Load experiments from database for dashboard creation.

    Args:
        exp_ids: list of experiment IDs
        db_path: path to database

    Returns:
        list of experiment dicts
    """
    from src.db import get_db

    db = get_db(db_path)
    experiments = []

    with db.connect() as conn:
        cursor = conn.cursor()
        for exp_id in exp_ids:
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
            row = cursor.fetchone()
            if row:
                exp = {
                    "id": row["id"],
                    "name": row["name"],
                    "dataset": row["dataset"],
                    "tasks": json.loads(row["tasks"] or "[]"),
                    "model_config": json.loads(row["model_config"] or "{}"),
                    "hyperparams": json.loads(row["hyperparams"] or "{}"),
                    "metrics": json.loads(row["metrics"] or "{}"),
                    "training_logs": json.loads(row["training_logs"] or "[]"),
                    "best_epoch": row["best_epoch"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                }
                experiments.append(exp)

    return experiments


def create_dashboard_from_db(
    exp_ids: List[int],
    db_path: str = "bi_mamba_chem.db",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a dashboard from experiments in database.

    Args:
        exp_ids: list of experiment IDs
        db_path: path to database
        save_path: path to save dashboard

    Returns:
        matplotlib Figure object
    """
    experiments = load_experiments_from_db(exp_ids, db_path)
    return create_experiment_dashboard(experiments, save_path=save_path)
