"""
Evaluation metrics for molecular property prediction.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    # Binary classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1]),
    }

    # ROC-AUC if probabilities provided
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics['auc'] = float(auc)
        except ValueError:
            metrics['auc'] = 0.0

    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute metrics based on task type.

    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        task_type: 'regression' or 'classification'
        y_prob: Predicted probabilities (for classification)

    Returns:
        Dictionary of metrics
    """
    if task_type == 'regression':
        return compute_regression_metrics(y_true, y_pred)
    else:
        return compute_classification_metrics(y_true, y_pred, y_prob)


def format_metrics(metrics: Dict[str, Any], task_type: str) -> str:
    """
    Format metrics for display.

    Args:
        metrics: Dictionary of metrics
        task_type: 'regression' or 'classification'

    Returns:
        Formatted string
    """
    if task_type == 'regression':
        return (
            f"RMSE: {metrics.get('rmse', 0):.4f}, "
            f"MAE: {metrics.get('mae', 0):.4f}, "
            f"R2: {metrics.get('r2', 0):.4f}"
        )
    else:
        parts = [
            f"Accuracy: {metrics.get('accuracy', 0):.4f}",
            f"Precision: {metrics.get('precision', 0):.4f}",
            f"Recall: {metrics.get('recall', 0):.4f}",
            f"F1: {metrics.get('f1', 0):.4f}",
        ]
        if 'auc' in metrics:
            parts.append(f"AUC: {metrics['auc']:.4f}")
        return ", ".join(parts)


if __name__ == "__main__":
    # Test regression metrics
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)
    print("Regression metrics:", reg_metrics)

    # Test classification metrics
    y_true_cls = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_pred_cls = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob_cls = np.array([0.2, 0.9, 0.3, 0.6, 0.1, 0.8, 0.7, 0.6])
    cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls, y_prob_cls)
    print("Classification metrics:", cls_metrics)
