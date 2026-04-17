"""Shared utilities for training and evaluation."""

import argparse
import logging
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get device in order: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_train_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="训练 BiMamba 分子性质预测模型")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称（如 ESOL, BBBP, ClinTox）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="数据文件目录",
    )
    parser.add_argument("--train_file", type=str, default="train.csv", help="训练数据文件名")
    parser.add_argument("--val_file", type=str, default="val.csv", help="验证数据文件名")
    parser.add_argument("--test_file", type=str, default="test.csv", help="测试数据文件名")

    # 数据划分参数
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["random", "scaffold"],
        help="数据划分策略: random 或 scaffold (默认: random)",
    )
    parser.add_argument("--split_seed", type=int, default=42, help="数据划分随机种子 (默认: 42)")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例 (默认: 0.8)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (默认: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例 (默认: 0.1)")
    parser.add_argument(
        "--single_file",
        type=str,
        default=None,
        help="单数据文件路径（如 delaney.csv），与 train_file 互斥",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="manual",
        choices=["manual", "mamba_ssm"],
        help="模型类型: manual (无外部依赖) 或 mamba_ssm (需要 mamba-ssm 包)",
    )
    parser.add_argument("--d_model", type=int, default=256, help="模型维度 (embedding/输出维度)")
    parser.add_argument("--d_mamba", type=int, default=256, help="Mamba 内部维度 (必须是 256 倍数)")
    parser.add_argument("--n_layers", type=int, default=4, help="BiMamba 层数")
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="任务类型：regression（回归）或 classification（分类）",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "smooth_l1", "huber"],
        help="回归任务损失函数：mse（默认）、smooth_l1（更鲁棒）、huber（抗异常值）",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="池化方法：mean（平均池化）、max（最大池化）、cls（CLS token）",
    )
    parser.add_argument("--num_labels", type=int, default=1, help="输出标签数量")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    parser.add_argument("--bidirectional", action="store_true", help="启用双向 SSM（默认开启）")
    parser.add_argument(
        "--no-bidirectional",
        dest="bidirectional",
        action="store_false",
        help="禁用双向 SSM（单向）",
    )
    parser.set_defaults(bidirectional=True)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="权重衰减 (Mamba 官方推荐 0.1, 用于防止 B/C 矩阵发散)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数（用于增大有效批大小）",
    )
    parser.add_argument("--warmup_epochs", type=int, default=5, help="学习率预热轮数")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪的最大范数",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="早停耐心值：验证损失连续 N 个 epoch 无改善则停止训练",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备：cuda（GPU）、mps（Apple GPU）、cpu 或 auto（自动选择）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可重复性）")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="模型保存目录",
    )
    parser.add_argument("--log_interval", type=int, default=100, help="日志输出间隔（批次）")
    parser.add_argument("--eval_interval", type=int, default=500, help="评估间隔（批次）")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存检查点间隔（批次）")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker 进程数")
    parser.add_argument(
        "--db_path",
        type=str,
        default="interactive",
        help="数据库路径（默认 interactive 会让用户选择）",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="实验名称（默认为 {dataset}_{timestamp}）",
    )
    parser.add_argument(
        "--no_db",
        action="store_true",
        help="禁用数据库记录",
    )

    return parser.parse_args()


def parse_eval_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="评估 BiMamba 分子性质预测模型")

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="数据文件目录",
    )
    parser.add_argument("--test_file", type=str, default="test.csv", help="测试数据文件名")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument(
        "--model_type",
        type=str,
        default="manual",
        choices=["manual", "mamba_ssm"],
        help="模型类型: manual (无外部依赖) 或 mamba_ssm (需要 mamba-ssm 包)",
    )
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")
    parser.add_argument("--n_layers", type=int, default=4, help="BiMamba 层数")
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="任务类型",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="池化方法",
    )
    parser.add_argument("--num_labels", type=int, default=1, help="输出标签数量")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备：cuda、mps、cpu 或 auto",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="最大评估样本数（-1 表示全部）",
    )

    return parser.parse_args()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    normalizer: Optional[object] = None,
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if args.task_type == "classification":
                labels = labels.float()

            logits, loss = model(input_ids=input_ids, labels=labels)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())

    if num_batches == 0:
        logger.warning("evaluate() got 0 batches from data_loader")
        return {"loss": 0.0, "mae": 0.0, "mse": 0.0, "rmse": 0.0}

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics: Dict[str, float] = {"loss": total_loss / num_batches}

    if args.task_type == "regression":
        mae = torch.mean(torch.abs(all_preds - all_labels)).item()
        mse = torch.mean((all_preds - all_labels) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        metrics.update({"mae": mae, "mse": mse, "rmse": rmse})

        if normalizer is not None and hasattr(normalizer, "inverse_transform"):
            preds_orig = normalizer.inverse_transform(all_preds.numpy())
            labels_orig = normalizer.inverse_transform(all_labels.numpy())
            rmse_orig = np.sqrt(np.mean((preds_orig - labels_orig) ** 2))
            mae_orig = np.mean(np.abs(preds_orig - labels_orig))
            metrics.update({"rmse_orig": rmse_orig, "mae_orig": mae_orig})
    else:
        if args.num_labels == 1:
            preds_prob = torch.sigmoid(all_preds).numpy()
            preds_label = (preds_prob > 0.5).astype(int)
            labels_np = all_labels.numpy()

            try:
                metrics.update(
                    {
                        "auc": roc_auc_score(labels_np, preds_prob),
                        "accuracy": accuracy_score(labels_np, preds_label),
                    }
                )
            except ValueError:
                metrics.update({"auc": 0.5, "accuracy": 0.0})
        else:
            preds_prob = torch.softmax(all_preds, dim=-1).numpy()
            preds_label = torch.argmax(all_preds, dim=-1).numpy()
            labels_np = all_labels.numpy()

            try:
                metrics.update(
                    {
                        "auc": roc_auc_score(labels_np, preds_prob, multi_class="ovr"),
                        "accuracy": accuracy_score(labels_np, preds_label),
                    }
                )
            except ValueError:
                metrics.update({"auc": 0.5, "accuracy": 0.0})

    return metrics
