#!/usr/bin/env python3
"""
Bi-Mamba 模型训练脚本
"""

import argparse
import atexit
import signal
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.db import ExperimentRepository
from src.models.bimamba import BiMambaForPropertyPrediction as BiMambaManual
from src.models.bimamba import create_bimamba_model as create_bimamba_manual
from src.models.bimamba_with_mamba_ssm import (
    BiMambaForPropertyPrediction as BiMambaMambaSSM,
)
from src.models.bimamba_with_mamba_ssm import (
    create_bimamba_model as create_bimamba_mamba_ssm,
)
from src.models.vanilla_transformer import (
    VanillaTransformerForPropertyPrediction as TransformerModel,
)
from src.data import (
    MoleculeDataset,
    create_data_loaders,
    MoleculeTokenizer,
)
from src.data.split import scaffold_split_dataset, random_split_dataset
from src.data.column_mapping import detect_column_mapping
from src.data.split import select_database
from src.shared.utils import parse_train_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    scaler: Optional[GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        # BCEWithLogitsLoss expects float labels
        if args.task_type == "classification":
            labels = labels.float()

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(input_ids=input_ids, labels=labels)
            loss_value = loss.item()
            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            logits, loss = model(input_ids=input_ids, labels=labels)
            loss_value = loss.item()
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break

        if has_nan_grad:
            optimizer.zero_grad()
            logger.warning(f"Batch {batch_idx}: 梯度包含 NaN，跳过此批次")
            continue

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            num_batches += 1
            total_loss += loss_value

        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch: {epoch + 1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss_value:.6f}"
            )

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    normalizer: Optional[object] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for input_ids, labels in val_loader:
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
        logger.warning("evaluate() got 0 batches from val_loader")
        return {"loss": 0.0, "mae": 0.0, "mse": 0.0, "rmse": 0.0}

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics: Dict[str, float] = {"loss": total_loss / num_batches}

    if args.task_type == "regression":
        # Val loader returns NORMALIZED labels (via NormalizedDataset wrapper).
        # Compute metrics in normalized space first.
        mae = torch.mean(torch.abs(all_preds - all_labels)).item()
        mse = torch.mean((all_preds - all_labels) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        metrics.update({"mae": mae, "mse": mse, "rmse": rmse})

        # Also compute RMSE in original scale if normalizer provided.
        if normalizer is not None and hasattr(normalizer, "inverse_transform"):
            preds_orig = normalizer.inverse_transform(all_preds.numpy())
            labels_orig = normalizer.inverse_transform(all_labels.numpy())
            rmse_orig = np.sqrt(np.mean((preds_orig - labels_orig) ** 2))
            mae_orig = np.mean(np.abs(preds_orig - labels_orig))
            metrics.update({"rmse_orig": rmse_orig, "mae_orig": mae_orig})
    else:
        # Classification metrics
        from sklearn.metrics import roc_auc_score, accuracy_score

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


def log_experiment_to_json(experiment_data, filepath):
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        elif hasattr(obj, "__float__") and not isinstance(obj, (int, float, str, bool, type(None))):
            return float(obj)
        return obj

    with open(filepath, "w") as f:
        json.dump(convert_to_native(experiment_data), f, indent=2)


def main():
    args = parse_train_args()
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.dataset}_{args.model_type}_{timestamp}.json"
    log_filepath = os.path.join(logs_dir, log_filename)

    experiment_data = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "model_type": args.model_type,
            "task_type": args.task_type,
            "split": args.split,
            "split_seed": args.split_seed,
        },
        "training_params": {},
        "model_params": {},
        "data_info": {},
        "training_results": {},
    }

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, args.val_file) if args.val_file else None
    test_path = os.path.join(args.data_dir, args.test_file) if args.test_file else None

    if args.single_file:
        single_path = os.path.join(args.data_dir, args.single_file)
        import tempfile

        tmpdir = os.path.join(args.data_dir, f".tmp_split_{os.getpid()}")
        os.makedirs(tmpdir, exist_ok=True)

        try:
            if args.split == "scaffold":
                scaffold_split_dataset(
                    single_path,
                    output_dir=tmpdir,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
            else:
                random_split_dataset(
                    single_path,
                    output_dir=tmpdir,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
            train_path = os.path.join(tmpdir, "train.csv")
            val_path = os.path.join(tmpdir, "val.csv")
            test_path = os.path.join(tmpdir, "test.csv")

            logger.info(f"从 {args.data_dir} 加载数据（单文件划分模式）")
            train_loader, val_loader, test_loader, normalizer = create_data_loaders(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                batch_size=args.batch_size,
                task_type=args.task_type,
                dataset_name=args.dataset,
                max_length=args.max_length,
                num_workers=args.num_workers,
                normalize=(args.task_type == "regression"),
            )
        finally:
            experiment_data["data_info"]["train_samples"] = len(train_loader.dataset)
            if val_loader:
                experiment_data["data_info"]["val_samples"] = len(val_loader.dataset)
            if test_loader:
                experiment_data["data_info"]["test_samples"] = len(test_loader.dataset)
    else:
        logger.info(f"从 {args.data_dir} 加载数据")
        train_loader, val_loader, test_loader, normalizer = create_data_loaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=args.batch_size,
            task_type=args.task_type,
            dataset_name=args.dataset,
            max_length=args.max_length,
            num_workers=args.num_workers,
            normalize=(args.task_type == "regression"),
        )
        experiment_data["data_info"]["train_samples"] = len(train_loader.dataset)
        if val_loader:
            experiment_data["data_info"]["val_samples"] = len(val_loader.dataset)
        if test_loader:
            experiment_data["data_info"]["test_samples"] = len(test_loader.dataset)
    if normalizer:
        logger.info(f"Z-score 归一化: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
        experiment_data["data_info"]["normalization"] = {
            "mean": float(normalizer.mean),
            "std": float(normalizer.std),
        }

    dataset_for_vocab = train_loader.dataset.base_dataset if normalizer else train_loader.dataset
    vocab_size = dataset_for_vocab.get_vocab_size()
    pad_token_id = dataset_for_vocab.get_pad_token_id()
    logger.info(f"词汇表大小: {vocab_size}")
    experiment_data["data_info"]["vocab_size"] = vocab_size
    experiment_data["data_info"]["max_length"] = args.max_length

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
            "d_mamba": args.d_mamba,
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
    elif args.model_type == "transformer":
        logger.info("创建 VanillaTransformer 模型")
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=8,
            d_ffn=args.d_model * 2,
            max_seq_length=args.max_length,
            num_labels=args.num_labels,
            task_type=args.task_type,
            pooling=args.pooling,
            dropout=args.dropout,
            pad_token_id=pad_token_id,
        )
    else:
        logger.info("创建 BiMamba 模型 (mamba_ssm, 使用 mamba-ssm 包)")
        model = create_bimamba_mamba_ssm(
            vocab_size=vocab_size,
            d_model=args.d_model,
            d_mamba=args.d_mamba,
            n_layers=args.n_layers,
            task_type=args.task_type,
            num_labels=args.num_labels,
            pooling=args.pooling,
            dropout=args.dropout,
            pad_token_id=pad_token_id,
            bidirectional=args.bidirectional,
        )
    model = model.to(device)

    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            logger.info("启用 torch.compile 加速")
        except Exception as e:
            logger.warning(f"torch.compile 失败: {e}")

    scaler = GradScaler() if device.type == "cuda" else None
    if scaler:
        logger.info("启用混合精度训练 (AMP, bfloat16)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    experiment_data["model_params"] = {
        "d_model": args.d_model,
        "d_mamba": args.d_mamba,
        "n_layers": args.n_layers,
        "pooling": args.pooling,
        "dropout": args.dropout,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    experiment_data["training_params"] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "device": str(device),
        "seed": args.seed,
    }

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.gradient_accumulation_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("开始训练")
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, f"{args.dataset}_bi_mamba_best.pt")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args, scaler
        )

        val_metrics = (
            evaluate(model, val_loader, device, args, normalizer) if val_loader else {"loss": 0.0}
        )

        epoch_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} 完成，耗时 {epoch_time:.2f}s | "
            f"训练损失: {train_loss:.6f} | "
            f"验证损失: {val_metrics['loss']:.6f}"
        )

        for key, value in val_metrics.items():
            if key != "loss":
                logger.info(f"  验证 {key.upper()}: {value:.6f}")

        if exp_repo and exp_id is not None:
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics.get("loss", 0),
                "val_mae": val_metrics.get("mae", 0),
                "val_rmse": val_metrics.get("rmse", 0),
            }
            exp_repo.append_training_log(exp_id, epoch_log)

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

        batches_per_epoch = len(train_loader)
        save_every_n_epochs = (
            max(1, args.save_interval // batches_per_epoch) if args.save_interval > 0 else 0
        )
        if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
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

        if val_metrics["loss"] < best_val_loss:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logger.info(f"验证损失连续 {epochs_without_improvement} 个 epoch 无改善")

        if epochs_without_improvement >= args.early_stopping_patience:
            logger.info(
                f"早停触发：连续 {epochs_without_improvement} 个 epoch 验证损失无改善，停止训练"
            )
            break

    test_metrics = {}
    if test_loader:
        logger.info("在测试集上评估")
        test_metrics = evaluate(model, test_loader, device, args, normalizer)
        logger.info(f"测试结果:")
        for key, value in test_metrics.items():
            logger.info(f"  {key.upper()}: {value:.6f}")

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

    # Finalize and save experiment data to JSON
    experiment_data["training_results"]["best_val_loss"] = best_val_loss
    experiment_data["training_results"]["epochs_trained"] = epoch + 1
    experiment_data["training_results"]["test_loss"] = test_metrics.get("loss", 0)
    experiment_data["training_results"]["test_mae"] = test_metrics.get("mae", 0)
    experiment_data["training_results"]["test_mse"] = test_metrics.get("mse", 0)
    experiment_data["training_results"]["test_rmse"] = test_metrics.get("rmse", 0)
    experiment_data["training_results"]["test_rmse_orig"] = test_metrics.get("rmse_orig", 0)
    experiment_data["training_results"]["test_mae_orig"] = test_metrics.get("mae_orig", 0)
    experiment_data["training_results"]["test_auc"] = test_metrics.get("auc", 0)
    experiment_data["training_results"]["test_accuracy"] = test_metrics.get("accuracy", 0)

    log_experiment_to_json(experiment_data, log_filepath)
    logger.info("训练完成！")
    logger.info(f"实验日志已保存到 {log_filepath}")


if __name__ == "__main__":
    _cleanup_done = [False]

    def _cleanup():
        if _cleanup_done[0]:
            return
        _cleanup_done[0] = True
        logger.info("清理 DataLoader worker 进程...")
        import os
        import subprocess

        try:
            subprocess.run(["pkill", "-9", "-f", "_worker.py"], timeout=3)
        except:
            pass

    def _signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在退出...")
        _cleanup()
        exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup)

    main()
