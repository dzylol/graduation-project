"""
Prediction visualization utilities.

Functions for plotting prediction results, scatter plots, and residual analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str = "Property",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
    metrics: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """
    Create a scatter plot of predicted vs true values.

    Args:
        y_true: true values
        y_pred: predicted values
        task_name: name of the task for title
        save_path: path to save figure
        figsize: figure size
        metrics: optional dict of metrics to display (MAE, RMSE, R2)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c="steelblue", edgecolors="none")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    ax.set_xlabel("True Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(f"{task_name}: Predicted vs True", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if metrics:
        textstr = "\n".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.95,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str = "Property",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Create residual plots for regression analysis.

    Args:
        y_true: true values
        y_pred: predicted values
        task_name: name of the task
        save_path: path to save figure
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{task_name}: Residual Analysis", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20, c="steelblue", edgecolors="none")
    ax1.axhline(y=0, color="r", linestyle="--", lw=2)
    ax1.set_xlabel("Predicted Values", fontsize=12)
    ax1.set_ylabel("Residuals", fontsize=12)
    ax1.set_title("Residual Plot")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.axvline(x=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Residuals", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Residual Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Residual plots saved to {save_path}")

    return fig


def plot_multitask_predictions(
    predictions: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Create prediction scatter plots for multiple tasks.

    Args:
        predictions: dict of task_name -> {y_true, y_pred}
        save_path: path to save figure
        figsize: figure size (auto-calculated if None)

    Returns:
        matplotlib Figure object
    """
    n_tasks = len(predictions)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Multi-task Prediction Results", fontsize=14, fontweight="bold")

    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    for i, (task_name, data) in enumerate(predictions.items()):
        ax = axes[i]
        y_true = data.get("y_true", [])
        y_pred = data.get("y_pred", [])

        if len(y_true) > 0 and len(y_pred) > 0:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            ax.scatter(
                y_true, y_pred, alpha=0.5, s=20, c="steelblue", edgecolors="none"
            )

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            ax.text(
                0.05,
                0.95,
                f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(task_name)
        ax.grid(True, alpha=0.3)

    for i in range(n_tasks, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Multi-task predictions saved to {save_path}")

    return fig


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute common regression metrics.

    Args:
        y_true: true values
        y_pred: predicted values

    Returns:
        dict of metric_name -> value
    """
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
