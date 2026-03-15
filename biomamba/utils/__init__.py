"""
工具模块 - 包含评估指标和日志记录工具

本模块导出的组件:

【评估指标】
- compute_regression_metrics: 计算回归任务指标 (MSE, RMSE, MAE, R²)
- compute_classification_metrics: 计算分类任务指标 (Accuracy, Precision, Recall, F1, AUC)
- compute_metrics: 统一的指标计算接口,根据任务类型自动选择
- format_metrics: 将指标格式化为易读的字符串

【日志记录】
- Logger: 日志记录器类,支持控制台和文件双输出
- get_logger: 获取日志记录器实例的便捷函数
"""

from .metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_metrics,
    format_metrics,
)
from .logger import Logger, get_logger

__all__ = [
    'compute_regression_metrics',
    'compute_classification_metrics',
    'compute_metrics',
    'format_metrics',
    'Logger',
    'get_logger',
]
