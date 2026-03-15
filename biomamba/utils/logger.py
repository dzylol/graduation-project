"""
日志记录模块 - 用于记录训练和评估过程中的信息

Logger (日志记录器) 是编程中常用的工具,用于:
1. 记录程序运行状态和进度
2. 记录训练过程中的指标变化 (如loss, accuracy)
3. 记录警告和错误信息
4. 将日志同时输出到控制台和文件

本模块提供了:
- Logger类: 功能完整的日志记录器,支持控制台和文件双输出
- get_logger()函数: 快速获取日志记录器实例

使用示例:
    logger = get_logger(name='training', log_dir='./logs')
    logger.info("Training started")
    logger.log_metrics(epoch=1, metrics={'loss': 0.5, 'accuracy': 0.8})
    logger.save_metrics()  # 保存指标历史到JSON文件
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime


class Logger:
    """
    日志记录器类

    功能:
    - 同时向控制台和文件输出日志
    - 记录训练指标并保存历史
    - 支持不同级别的日志: DEBUG, INFO, WARNING, ERROR

    属性:
    - log_dir: 日志文件保存目录
    - name: 日志记录器名称
    - log_file: 日志文件路径
    - metrics_history: 训练指标历史记录列表
    """

    def __init__(
        self,
        log_dir: str = './logs',
        name: str = 'biomamba',
        level: int = logging.INFO,
    ):
        """
        初始化日志记录器

        初始化时会:
        1. 创建日志保存目录(如果不存在)
        2. 生成带时间戳的日志文件名
        3. 配置文件处理器(保存详细日志到文件)
        4. 配置控制台处理器(显示简洁日志)

        Args:
            log_dir: 日志文件保存的目录,默认'./logs'
            name: 日志记录器名称,默认'biomamba'
            level: 日志级别,默认logging.INFO (可选: DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = log_dir
        self.name = name

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

        # Configure logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.log_file = log_file
        self.metrics_history = []

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def log_metrics(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        prefix: str = '',
    ):
        """
        记录训练指标

        将每个epoch的训练指标记录到日志中,并保存到历史记录中。
        方便后续分析训练过程和绘制指标曲线。

        Args:
            epoch: 当前的训练轮次 (从1开始)
            metrics: 包含各项指标的字典,如 {'loss': 0.5, 'accuracy': 0.8}
            prefix: 指标名称的前缀,用于区分不同阶段的指标:
                   - 'train_': 训练集指标,如 'train_loss'
                   - 'val_': 验证集指标,如 'val_accuracy'
        """
        metric_str = ', '.join([
            f"{prefix}{k}: {v:.4f}" if isinstance(v, float) else f"{prefix}{k}: {v}"
            for k, v in metrics.items()
        ])
        self.info(f"Epoch {epoch}: {metric_str}")

        # Save to history
        self.metrics_history.append({
            'epoch': epoch,
            **metrics,
        })

    def save_metrics(self, filepath: Optional[str] = None):
        """
        保存指标历史到JSON文件

        将之前所有记录的指标保存为JSON格式的文件,方便后续分析或可视化。
        JSON是一种轻量级的数据交换格式,易于阅读和处理。

        Args:
            filepath: 保存路径,如果为None则自动生成带时间戳的文件名
        """
        if filepath is None:
            filepath = os.path.join(
                self.log_dir,
                f'{self.name}_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )

        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        self.info(f"Metrics saved to: {filepath}")

    def log_config(self, config: Dict[str, Any]):
        """
        记录配置信息

        在训练开始时记录超参数和配置,方便复现实验和调试。

        Args:
            config: 包含配置信息的字典,如 {'batch_size': 32, 'lr': 0.001}
        """
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")


def get_logger(
    log_dir: str = './logs',
    name: str = 'biomamba',
) -> Logger:
    """
    获取日志记录器实例的便捷函数

    这是创建Logger对象的推荐方式,会自动配置好所有参数。

    Args:
        log_dir: 日志文件保存的目录,默认'./logs'
        name: 日志记录器名称,默认'biomamba'

    Returns:
        配置好的Logger实例
    """
    return Logger(log_dir=log_dir, name=name)


if __name__ == "__main__":
    # Test logger
    logger = get_logger(name='test')

    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Log metrics
    logger.log_metrics(1, {'loss': 0.5, 'accuracy': 0.8})
    logger.log_metrics(2, {'loss': 0.3, 'accuracy': 0.9})

    # Save metrics
    logger.save_metrics()

    print(f"Log file: {logger.log_file}")
