"""
评估指标模块 - 用于衡量分子属性预测模型的性能

本模块提供了回归任务和分类任务的各种评估指标:
- 回归任务: MSE, RMSE, MAE, R²
- 分类任务: Accuracy, Precision, Recall, F1, AUC

使用示例:
    # 回归任务
    metrics = compute_metrics(y_true, y_pred, 'regression')
    print(f"RMSE: {metrics['rmse']:.4f}")

    # 分类任务
    metrics = compute_metrics(y_true, y_pred, 'classification', y_prob)
    print(f"AUC: {metrics['auc']:.4f}")
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
    计算回归任务的评估指标

    回归任务是指预测连续值,如预测分子的溶解度。

    指标说明:
    - MSE (Mean Squared Error, 均方误差): 预测值与真实值差异的平方的平均值。
      值越小越好,表示预测误差越小。
    - RMSE (Root Mean Squared Error, 均方根误差): MSE的平方根。
      与目标值单位相同,更易解释。值越小越好。
    - MAE (Mean Absolute Error, 平均绝对误差): 预测值与真实值绝对差异的平均值。
      对异常值更鲁棒。值越小越好。
    - R² (R-squared, 决定系数): 表示模型解释目标变量变异的程度。
      范围通常在0-1之间,值越接近1表示模型拟合效果越好。

    Args:
        y_true: 真实的目标值数组
        y_pred: 模型预测的值数组

    Returns:
        包含MSE, RMSE, MAE, R²指标的字典
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
    计算分类任务的评估指标

    分类任务是指预测离散的类别标签,如判断分子是否有毒性。

    指标说明:
    - Accuracy (准确率): 正确预测的数量除以总预测数量。
      值越接近1越好,但在类别不平衡时可能具有误导性。
    - Precision (精确率): 预测为正类中实际为正类的比例。
      值越接近1越好,高精确率意味着较少的误报。
    - Recall (召回率): 实际为正类中被正确预测的比例。
      值越接近1越好,高召回率意味着较少的漏报。
    - F1 Score (F1分数): Precision和Recall的调和平均数。
      值越接近1越好,是精确率和召回率的平衡指标。
    - AUC (Area Under ROC Curve, ROC曲线下面积): 衡量分类器区分能力的指标。
      值越接近1越好,0.5表示随机猜测,1.0表示完美分类。
    - Confusion Matrix (混淆矩阵): TN(真负), FP(假正), FN(假负), TP(真正)

    Args:
        y_true: 真实的类别标签数组 (0或1)
        y_pred: 模型预测的类别标签数组 (0或1)
        y_prob: 模型预测为正类的概率数组 (可选,用于计算AUC)

    Returns:
        包含Accuracy, Precision, Recall, F1, AUC等指标的字典
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
    根据任务类型计算相应的评估指标

    本函数是一个统一入口,根据task_type参数自动选择计算回归指标或分类指标。

    Args:
        y_true: 真实值或真实标签数组
        y_pred: 预测值或预测标签数组
        task_type: 任务类型,可选'regression'(回归)或'classification'(分类)
        y_prob: 预测为正类的概率 (仅分类任务需要,用于计算AUC)

    Returns:
        包含相应评估指标的字典
    """
    if task_type == 'regression':
        return compute_regression_metrics(y_true, y_pred)
    else:
        return compute_classification_metrics(y_true, y_pred, y_prob)


def format_metrics(metrics: Dict[str, Any], task_type: str) -> str:
    """
    将评估指标格式化为易读的字符串

    用于在控制台打印时将指标字典转换为简洁的字符串格式。

    Args:
        metrics: 包含评估指标的字典
        task_type: 任务类型,可选'regression'(回归)或'classification'(分类)

    Returns:
        格式化的指标字符串,例如:
        - 回归任务: "RMSE: 0.1234, MAE: 0.1000, R2: 0.9500"
        - 分类任务: "Accuracy: 0.9000, Precision: 0.8500, Recall: 0.8800, F1: 0.8650, AUC: 0.9200"
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
