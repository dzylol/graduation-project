#!/usr/bin/env python3
"""
Bi-Mamba 模型训练脚本

本脚本用于训练 Bi-Mamba 分子性质预测模型。

训练流程：
1. 加载数据
2. 创建模型
3. 训练循环（多轮 epoch）
4. 评估模型
5. 保存模型

使用方法：
```bash
python train.py --dataset ESOL --epochs 100 --batch_size 32 --device cuda --model_type manual
python train.py --dataset ESOL --epochs 100 --batch_size 32 --device cuda --model_type mamba_ssm
```

作者: Bi-Mamba-Chem Team
"""

# ============================================================================
# 导入必要的库
# ============================================================================

import argparse  # 命令行参数解析
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器
import logging  # 日志记录
import os  # 文件路径操作
import json  # JSON 文件处理
from typing import Dict, Any, Optional  # 类型提示
import time  # 时间测量

from src.db import ExperimentRepository

# 导入本地模块
from src.models.bimamba import BiMambaForPropertyPrediction as BiMambaManual
from src.models.bimamba import create_bimamba_model as create_bimamba_manual
from src.models.bimamba_with_mamba_ssm import (
    BiMambaForPropertyPrediction as BiMambaMambaSSM,
)
from src.models.bimamba_with_mamba_ssm import (
    create_bimamba_model as create_bimamba_mamba_ssm,
)
from src.data.molecule_dataset import (
    MoleculeDataset,
    create_data_loaders,
    MoleculeTokenizer,
    select_database,
)

# ============================================================================
# 日志配置
# ============================================================================

# 配置日志格式和输出
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 格式
    handlers=[
        logging.FileHandler("training.log"),  # 保存到文件
        logging.StreamHandler(),  # 输出到终端
    ],
)
logger = logging.getLogger(__name__)  # 获取日志记录器


# ============================================================================
# 命令行参数解析
# ============================================================================


def parse_args():
    """
    解析命令行参数

    使用 argparse 模块解析命令行参数，方便调整训练配置。

    参数说明：
    - 数据参数：数据集路径、文件名称
    - 模型参数：维度、层数、任务类型等
    - 训练参数：轮数、批大小、学习率等
    - 其他：设备、随机种子、输出路径等
    """
    parser = argparse.ArgumentParser(description="训练 BiMamba 分子性质预测模型")

    # -------------------------------------------------------------------------
    # 数据参数
    # -------------------------------------------------------------------------
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
    parser.add_argument(
        "--train_file", type=str, default="train.csv", help="训练数据文件名"
    )
    parser.add_argument(
        "--val_file", type=str, default="val.csv", help="验证数据文件名"
    )
    parser.add_argument(
        "--test_file", type=str, default="test.csv", help="测试数据文件名"
    )

    # -------------------------------------------------------------------------
    # 模型参数
    # -------------------------------------------------------------------------
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
        help="任务类型：regression（回归）或 classification（分类）",
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

    # -------------------------------------------------------------------------
    # 训练参数
    # -------------------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
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

    # -------------------------------------------------------------------------
    # 其他参数
    # -------------------------------------------------------------------------
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
    parser.add_argument(
        "--log_interval", type=int, default=100, help="日志输出间隔（批次）"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=500, help="评估间隔（批次）"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="保存检查点间隔（批次）"
    )
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
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


# ============================================================================
# 辅助函数
# ============================================================================


def set_seed(seed: int):
    """
    设置随机种子

    确保实验可重复性。使用相同的种子，每次训练结果应该相同。

    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)  # CPU 随机种子
    torch.cuda.manual_seed_all(seed)  # GPU 随机种子
    torch.backends.cudnn.deterministic = True  # CUDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 基准测试


def get_device(device_str: str) -> torch.device:
    """
    获取设备

    根据用户指定和可用性选择最佳设备。
    优先级：CUDA GPU > Apple MPS > CPU

    Args:
        device_str: 设备字符串

    Returns:
        torch.device 对象
    """
    if device_str == "auto":
        # 自动选择
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


# ============================================================================
# 训练和评估函数
# ============================================================================


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> float:
    """
    训练一个 epoch（一个完整的数据遍历）

    训练步骤：
    1. 前向传播：计算预测值和损失
    2. 反向传播：计算梯度
    3. 参数更新：使用优化器更新模型参数

    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        epoch: 当前轮数
        args: 命令行参数

    Returns:
        平均损失值
    """
    model.train()  # 设置为训练模式
    total_loss = 0.0  # 累计损失
    num_batches = 0  # 批次计数

    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        # -------------------------------------------------------------------------
        # 1. 数据移到设备
        # -------------------------------------------------------------------------
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # -------------------------------------------------------------------------
        # 2. 前向传播
        # -------------------------------------------------------------------------
        # model() 会调用 forward() 方法
        logits, loss = model(input_ids=input_ids, labels=labels)

        # 保存原始损失值用于日志
        loss_value = loss.item()

        # -------------------------------------------------------------------------
        # 3. 反向传播
        # -------------------------------------------------------------------------
        # 梯度累积：将损失除以累积步数
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()  # 反向传播，计算梯度

        # 检查梯度是否有 NaN
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break

        # 如果梯度有 NaN，跳过这个批次
        if has_nan_grad:
            optimizer.zero_grad()
            logger.warning(f"Batch {batch_idx}: 梯度包含 NaN，跳过此批次")
            continue

        # -------------------------------------------------------------------------
        # 4. 梯度累积和参数更新
        # -------------------------------------------------------------------------
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 参数更新
            optimizer.step()

            # 学习率调度
            scheduler.step()

            # 清零梯度（重要！否则梯度会累积）
            optimizer.zero_grad()

        # 累计损失（使用原始损失值）
        total_loss += loss_value
        num_batches += 1

        # -------------------------------------------------------------------------
        # 5. 日志输出
        # -------------------------------------------------------------------------
        if batch_idx % args.log_interval == 0:
            logger.info(
                f"Epoch: {epoch + 1}/{args.epochs} | "
                f"Batch: {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss_value:.6f}"
            )

    # 返回平均损失
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    在验证集上评估模型

    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        device: 计算设备
        args: 命令行参数

    Returns:
        评估指标字典
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    num_batches = 0
    all_preds = []  # 收集所有预测
    all_labels = []  # 收集所有标签

    # 禁用梯度计算（节省内存和计算）
    with torch.no_grad():
        for input_ids, labels in val_loader:
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

    # 合并所有预测和标签
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算指标
    metrics = {"loss": total_loss / num_batches}

    # -------------------------------------------------------------------------
    # 根据任务类型计算不同指标
    # -------------------------------------------------------------------------
    if args.task_type == "regression":
        # 回归任务指标
        mae = torch.mean(torch.abs(all_preds - all_labels)).item()  # 平均绝对误差
        mse = torch.mean((all_preds - all_labels) ** 2).item()  # 均方误差
        rmse = torch.sqrt(torch.tensor(mse)).item()  # 均方根误差
        metrics.update({"mae": mae, "mse": mse, "rmse": rmse})
    else:
        # 分类任务指标
        from sklearn.metrics import roc_auc_score, accuracy_score

        if args.num_labels == 1:
            # 二分类
            # sigmoid 将 logits 转换为概率
            preds_prob = torch.sigmoid(all_preds).numpy()
            preds_label = (preds_prob > 0.5).astype(int)
            labels_np = all_labels.numpy()

            try:
                auc = roc_auc_score(labels_np, preds_prob)  # AUC
                acc = accuracy_score(labels_np, preds_label)  # 准确率
                metrics.update({"auc": auc, "accuracy": acc})
            except ValueError:
                # 处理只有一个类别的情况
                metrics.update({"auc": 0.5, "accuracy": 0.0})
        else:
            # 多分类
            preds_prob = torch.softmax(all_preds, dim=-1).numpy()
            preds_label = torch.argmax(all_preds, dim=-1).numpy()
            labels_np = all_labels.numpy()

            try:
                auc = roc_auc_score(labels_np, preds_prob, multi_class="ovr")
                acc = accuracy_score(labels_np, preds_label)
                metrics.update({"auc": auc, "accuracy": acc})
            except ValueError:
                metrics.update({"auc": 0.5, "accuracy": 0.0})

    return metrics


# ============================================================================
# 主函数
# ============================================================================


def main():
    """
    主训练函数

    完整的训练流程：
    1. 解析参数
    2. 设置设备和随机种子
    3. 加载数据
    4. 创建模型
    5. 训练循环
    6. 评估和保存
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
    # 4. 创建输出目录
    # -------------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 5. 保存训练参数
    # -------------------------------------------------------------------------
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------------------------------------------------------------
    # 6. 加载数据
    # -------------------------------------------------------------------------
    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, args.val_file) if args.val_file else None
    test_path = os.path.join(args.data_dir, args.test_file) if args.test_file else None

    logger.info(f"从 {args.data_dir} 加载数据")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=args.batch_size,
        task_type=args.task_type,
        max_length=args.max_length,
        num_workers=0,
    )

    # 从数据集获取词汇表信息
    vocab_size = train_loader.dataset.get_vocab_size()
    pad_token_id = train_loader.dataset.get_pad_token_id()
    logger.info(f"词汇表大小: {vocab_size}")

    # -------------------------------------------------------------------------
    # 7. 初始化实验追踪数据库
    # -------------------------------------------------------------------------
    exp_repo = None
    exp_id = None
    if not args.no_db:
        if args.db_path == "interactive":
            db_path = select_database()
            logger.info(f"选择数据库: {db_path}")
        else:
            db_path = args.db_path
        exp_repo = ExperimentRepository(db_path=db_path)
        exp_name = args.exp_name or f"{args.dataset}_{int(time.time())}"
        model_config = {
            "model_type": args.model_type,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "pooling": args.pooling,
            "dropout": args.dropout,
            "vocab_size": vocab_size,
        }
        hyperparams = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "max_grad_norm": args.max_grad_norm,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        exp_id = exp_repo.create(
            name=exp_name,
            dataset=args.dataset,
            tasks=[args.task_type],
            model_config=model_config,
            hyperparams=hyperparams,
        )
        logger.info(f"创建实验记录: ID={exp_id}, 名称={exp_name}")

    # -------------------------------------------------------------------------
    # 8. 创建模型
    # -------------------------------------------------------------------------
    if args.model_type == "manual":
        logger.info("创建 BiMamba 模型 (manual SSM, 无外部依赖)")
        model = create_bimamba_manual(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            task_type=args.task_type,
            num_labels=args.num_labels,
            pooling=args.pooling,
            dropout=args.dropout,
            pad_token_id=pad_token_id,
        )
    else:
        logger.info("创建 BiMamba 模型 (mamba_ssm, 使用 mamba-ssm 包)")
        model = create_bimamba_mamba_ssm(
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
    # 8. 打印模型信息
    # -------------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # 9. 创建优化器和学习率调度器
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # 学习率调度器（带预热）
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = (
        len(train_loader) * args.warmup_epochs // args.gradient_accumulation_steps
    )

    def lr_lambda(current_step):
        """
        学习率调度函数

        前 warmup_steps 步线性增加学习率，
        之后线性衰减。
        """
        if current_step < warmup_steps:
            # 预热阶段：线性增加
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 衰减阶段：线性减少
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -------------------------------------------------------------------------
    # 10. 训练循环
    # -------------------------------------------------------------------------
    logger.info("开始训练")
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, f"{args.dataset}_bi_mamba_best.pt")

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练一个 epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args
        )

        # 在验证集上评估
        val_metrics = (
            evaluate(model, val_loader, device, args) if val_loader else {"loss": 0.0}
        )

        epoch_time = time.time() - start_time

        # -------------------------------------------------------------------------
        # 11. 记录结果
        # -------------------------------------------------------------------------
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} 完成，耗时 {epoch_time:.2f}s | "
            f"训练损失: {train_loss:.6f} | "
            f"验证损失: {val_metrics['loss']:.6f}"
        )

        # 打印其他指标
        for key, value in val_metrics.items():
            if key != "loss":
                logger.info(f"  验证 {key.upper()}: {value:.6f}")

        # -------------------------------------------------------------------------
        # 11. 记录到数据库
        # -------------------------------------------------------------------------
        if exp_repo and exp_id is not None:
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics.get("loss", 0),
                "val_mae": val_metrics.get("mae", 0),
                "val_rmse": val_metrics.get("rmse", 0),
            }
            exp_repo.append_training_log(exp_id, epoch_log)

        # -------------------------------------------------------------------------
        # 12. 保存最佳模型
        # -------------------------------------------------------------------------
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "args": vars(args),
                },
                best_model_path,
            )
            logger.info(f"保存最佳模型到 {best_model_path}")

        # -------------------------------------------------------------------------
        # 13. 定期保存检查点
        # -------------------------------------------------------------------------
        if (epoch + 1) % (args.save_interval // len(train_loader)) == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"{args.dataset}_bi_mamba_epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                },
                checkpoint_path,
            )
            logger.info(f"保存检查点到 {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 14. 最终测试
    # -------------------------------------------------------------------------
    test_metrics = {}
    if test_loader:
        logger.info("在测试集上评估")
        test_metrics = evaluate(model, test_loader, device, args)
        logger.info(f"测试结果:")
        for key, value in test_metrics.items():
            logger.info(f"  {key.upper()}: {value:.6f}")

    # -------------------------------------------------------------------------
    # 15. 更新数据库记录
    # -------------------------------------------------------------------------
    if exp_repo and exp_id is not None:
        final_metrics = {
            "best_val_loss": best_val_loss,
            "test_loss": test_metrics.get("loss", 0),
            "test_mae": test_metrics.get("mae", 0),
            "test_rmse": test_metrics.get("rmse", 0),
            "test_auc": test_metrics.get("auc", 0),
        }
        exp_repo.complete(exp_id, final_metrics, best_epoch=epoch + 1)
        logger.info(f"更新实验记录: ID={exp_id}, 状态=completed")

    logger.info("训练完成！")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
