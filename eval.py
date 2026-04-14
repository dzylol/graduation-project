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
from src.models.bimamba import BiMambaForPropertyPrediction as BiMambaManual
from src.models.bimamba import create_bimamba_model as create_bimamba_manual
from src.models.bimamba_with_mamba_ssm import (
    BiMambaForPropertyPrediction as BiMambaMambaSSM,
)
from src.models.bimamba_with_mamba_ssm import (
    create_bimamba_model as create_bimamba_mamba_ssm,
)
from src.data import (
    MoleculeDataset,
    create_data_loaders,
    MoleculeTokenizer,
)
from src.shared.utils import parse_eval_args, evaluate

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
# 辅助函数
# ============================================================================


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
    args = parse_eval_args()

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
    # 根据 model_type 选择模型实现
    model_type = getattr(args, "model_type", "manual")

    if model_type == "mamba_ssm":
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
    else:
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
