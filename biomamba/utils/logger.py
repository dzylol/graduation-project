"""
Logger utility for training and evaluation.
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime


class Logger:
    """
    Simple logger for training and evaluation.
    """

    def __init__(
        self,
        log_dir: str = './logs',
        name: str = 'biomamba',
        level: int = logging.INFO,
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            name: Logger name
            level: Logging level
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
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            prefix: Prefix for metric names (e.g., 'train_', 'val_')
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
        Save metrics history to JSON file.

        Args:
            filepath: Path to save metrics (optional)
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
        Log configuration.

        Args:
            config: Configuration dictionary
        """
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")


def get_logger(
    log_dir: str = './logs',
    name: str = 'biomamba',
) -> Logger:
    """
    Get logger instance.

    Args:
        log_dir: Directory to save logs
        name: Logger name

    Returns:
        Logger instance
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
