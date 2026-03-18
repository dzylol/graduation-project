#!/usr/bin/env python3



# ============================================================================
# 导入必要的库
# ============================================================================

import argparse  # 命令行参数解析
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
from torch.utils.data import DataLoader  # 数据加载器
import logging  # 日志记录
import os  # 文件路径操作
import json  # JSON 文件处理
from typing import Dict, Any, Optional  # 类型提示
import numpy as np  # 数值计算

# 从 sklearn 导入评估指标
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

# 导入本地模块
from src.models.bimamba import BiMambaForPropertyPrediction, create_bimamba_model
from src.data.molecule_dataset import (
    MoleculeDataset,
    create_data_loaders,
    MoleculeTokenizer,
)

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# 命令行参数解析
# ============================================================================


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="评估 BiMamba 分子性质预测模型")

    # -------------------------------------------------------------------------
    # 数据参数
    # -------------------------------------------------------------------------
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
    parser.add_argument(
        "--test_file", type=str, default="test.csv", help="测试数据文件名"
    )

    # -------------------------------------------------------------------------
    # 模型参数（需要与训练时保持一致）
    # -------------------------------------------------------------------------
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
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

    # -------------------------------------------------------------------------
    # 其他参数
    # -------------------------------------------------------------------------
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


# ============================================================================
# 辅助函数
# ============================================================================


def set_seed(seed: int):
    """
    设置随机种子（确保可重复性）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """
    获取设备
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


# ============================================================================
# 评估函数
# ============================================================================


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    在测试集上评估模型

    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
        args: 命令行参数

    Returns:
        评估指标字典
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0  # 累计损失
    num_batches = 0  # 批次计数
    all_preds = []  # 收集所有预测
    all_labels = []  # 收集所有标签

    # 样本计数
    samples_processed = 0

    # 禁用梯度计算
    with torch.no_grad():
        for input_ids, labels in test_loader:
            # 检查是否达到最大样本数
            if args.max_samples > 0 and samples_processed >= args.max_samples:
                break

            # 如果需要限制最后一个批次的大小
            if args.max_samples > 0:
                remaining = args.max_samples - samples_processed
                if remaining < input_ids.size(0):
                    input_ids = input_ids[:remaining]
                    labels = labels[:remaining]

            # 数据移到设备
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 前向传播
            logits, loss = model(input_ids=input_ids, labels=labels)

            total_loss += loss.item()
            num_batches += 1

            # 保存预测和标签
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())

            samples_processed += input_ids.size(0)

    # 检查是否有数据
    if len(all_preds) == 0:
        raise ValueError("没有处理任何样本进行评估")

    # 合并所有预测和标签
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算损失指标
    metrics = {"loss": total_loss / num_batches}

    # -------------------------------------------------------------------------
    # 根据任务类型计算不同指标
    # -------------------------------------------------------------------------
    if args.task_type == "regression":
        # 回归任务指标
        preds_np = all_preds.numpy()
        labels_np = all_labels.numpy()

        mae = mean_absolute_error(labels_np, preds_np)  # 平均绝对误差
        mse = mean_squared_error(labels_np, preds_np)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差

        metrics.update({"mae": mae, "mse": mse, "rmse": rmse})

        logger.info(f"MAE (平均绝对误差): {mae:.6f}")
        logger.info(f"MSE (均方误差): {mse:.6f}")
        logger.info(f"RMSE (均方根误差): {rmse:.6f}")

    else:
        # 分类任务指标
        if args.num_labels == 1:
            # 二分类
            # sigmoid 将 logits 转换为概率 [0, 1]
            preds_prob = torch.sigmoid(all_preds).numpy()
            # 大于 0.5 为正类
            preds_label = (preds_prob > 0.5).astype(int)
            labels_np = all_labels.numpy()

            try:
                # ROC-AUC：衡量模型区分正负样本的能力
                auc = roc_auc_score(labels_np, preds_prob)
                # 准确率：预测正确的比例
                acc = accuracy_score(labels_np, preds_label)
                metrics.update({"auc": auc, "accuracy": acc})

                logger.info(f"AUC (ROC曲线下面积): {auc:.6f}")
                logger.info(f"Accuracy (准确率): {acc:.6f}")

            except ValueError as e:
                logger.warning(f"无法计算 AUC: {e}")
                metrics.update({"auc": 0.5, "accuracy": 0.0})
        else:
            # 多分类
            # softmax 将 logits 转换为概率分布
            preds_prob = torch.softmax(all_preds, dim=-1).numpy()
            # 选择概率最大的类别作为预测
            preds_label = torch.argmax(all_preds, dim=-1).numpy()
            labels_np = all_labels.numpy()

            try:
                auc = roc_auc_score(labels_np, preds_prob, multi_class="ovr")
                acc = accuracy_score(labels_np, preds_label)
                metrics.update({"auc": auc, "accuracy": acc})

                logger.info(f"AUC (ROC曲线下面积): {auc:.6f}")
                logger.info(f"Accuracy (准确率): {acc:.6f}")

            except ValueError as e:
                logger.warning(f"无法计算 AUC: {e}")
                metrics.update({"auc": 0.5, "accuracy": 0.0})

    return metrics


# ============================================================================
# 主函数
# ============================================================================


def main():
    """
    主评估函数

    完整的评估流程：
    1. 解析参数
    2. 加载模型检查点
    3. 加载测试数据
    4. 评估模型
    5. 保存结果
    """
    # -------------------------------------------------------------------------
    # 1. 解析参数
    # -------------------------------------------------------------------------
    args = parse_args()

    # -------------------------------------------------------------------------
    # 2. 设置随机种子
    # -------------------------------------------------------------------------
    set_seed(args.seed)

    # -------------------------------------------------------------------------
    # 3. 获取设备
    # -------------------------------------------------------------------------
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")

    # -------------------------------------------------------------------------
    # 4. 加载模型检查点
    # -------------------------------------------------------------------------
    logger.info(f"从 {args.checkpoint} 加载模型")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 如果检查点包含参数，使用它
    if "args" in checkpoint:
        saved_args = checkpoint["args"]
        # 命令行参数优先
        for key, value in vars(args).items():
            if value is not None:
                saved_args[key] = value
        args = argparse.Namespace(**saved_args)
        logger.info("从检查点加载参数")

    # -------------------------------------------------------------------------
    # 5. 加载测试数据
    # -------------------------------------------------------------------------
    test_path = os.path.join(args.data_dir, args.test_file)
    logger.info(f"从 {test_path} 加载测试数据")

    # 创建数据加载器（只需要测试数据）
    _, _, test_loader = create_data_loaders(
        train_path="",  # 空路径，不会被使用
        val_path="",  # 空路径，不会被使用
        test_path=test_path,
        batch_size=args.batch_size,
        task_type=args.task_type,
        max_length=args.max_length,
        num_workers=4,
    )

    if test_loader is None:
        raise ValueError(f"无法从 {test_path} 加载测试数据")

    # 从数据集获取词汇表信息
    vocab_size = test_loader.dataset.get_vocab_size()
    pad_token_id = test_loader.dataset.get_pad_token_id()
    logger.info(f"词汇表大小: {vocab_size}")

    # -------------------------------------------------------------------------
    # 6. 创建模型
    # -------------------------------------------------------------------------
    logger.info("创建 BiMamba 模型")
    model = create_bimamba_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        task_type=args.task_type,
        num_labels=args.num_labels,
        pooling=args.pooling,
        dropout=args.dropout,
        pad_token_id=pad_token_id,
    )
    model = model.to(device)

    # -------------------------------------------------------------------------
    # 7. 加载模型权重
    # -------------------------------------------------------------------------
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("加载模型状态成功")
    else:
        # 假设检查点直接就是模型状态字典
        model.load_state_dict(checkpoint)
        logger.info("加载模型状态成功（假设格式）")

    # -------------------------------------------------------------------------
    # 8. 打印模型信息
    # -------------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # 9. 评估模型
    # -------------------------------------------------------------------------
    logger.info("开始评估")
    test_metrics = evaluate(model, test_loader, device, args)

    # -------------------------------------------------------------------------
    # 10. 输出结果
    # -------------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("测试结果:")
    logger.info("=" * 50)
    for key, value in test_metrics.items():
        logger.info(f"  {key.upper()}: {value:.6f}")

    # -------------------------------------------------------------------------
    # 11. 保存结果到文件
    # -------------------------------------------------------------------------
    results_file = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"评估结果已保存到 {results_file}")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
