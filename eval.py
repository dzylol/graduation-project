#!/usr/bin/env python3


import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
import json
from typing import Dict, Any, Optional
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    args = parse_eval_args()
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")

    logger.info(f"从 {args.checkpoint} 加载模型")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "args" in checkpoint:
        saved_args = checkpoint["args"]
        for key, value in vars(args).items():
            if value is not None:
                saved_args[key] = value
        args = argparse.Namespace(**saved_args)
        logger.info("从检查点加载参数")

    test_path = os.path.join(args.data_dir, args.test_file)
    logger.info(f"从 {test_path} 加载测试数据")

    _, _, test_loader, _ = create_data_loaders(
        train_path="",
        val_path="",
        test_path=test_path,
        batch_size=args.batch_size,
        task_type=args.task_type,
        max_length=args.max_length,
        num_workers=4,
        normalize=False,
    )

    if test_loader is None:
        raise ValueError(f"无法从 {test_path} 加载测试数据")

    vocab_size = test_loader.dataset.get_vocab_size()
    pad_token_id = test_loader.dataset.get_pad_token_id()
    logger.info(f"词汇表大小: {vocab_size}")

    model_type = getattr(args, "model_type", "manual")

    if model_type == "mamba_ssm":
        logger.info("创建 BiMamba 模型 (mamba_ssm)")
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
        logger.info("创建 BiMamba 模型 (manual SSM)")
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

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("加载模型状态成功")
    else:
        model.load_state_dict(checkpoint)
        logger.info("加载模型状态成功")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")

    logger.info("开始评估")
    test_metrics = evaluate(model, test_loader, device, args)

    logger.info("=" * 50)
    logger.info("测试结果:")
    logger.info("=" * 50)
    for key, value in test_metrics.items():
        logger.info(f"  {key.upper()}: {value:.6f}")

    results_file = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"评估结果已保存到 {results_file}")


if __name__ == "__main__":
    main()
