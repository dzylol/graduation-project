#!/usr/bin/env python3
"""
Multi-task training script for Bi-Mamba molecular property prediction.

Usage:
    python train_multitask.py --dataset mydataset --tasks "solubility:regression:1.0,toxicity:classification:0.5"
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import json
from typing import Dict, Any, Optional
import time

from src.models.multitask import create_multitask_model, parse_task_string
from src.data.multitask_dataset import create_multitask_loaders
from src.db import ExperimentRepository


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("multitask_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task BiMamba Training")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--val_file", type=str, default="val.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")

    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Task config: task1:type:weight,task2:type:weight,...",
    )
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument(
        "--pooling", type=str, default="mean", choices=["mean", "max", "cls"]
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--task_strategy", type=str, default="shared", choices=["shared", "separate"]
    )

    parser.add_argument("--db_path", type=str, default="bi_mamba_chem.db")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--no_db", action="store_true")

    return parser.parse_args()


def train_epoch(
    model, train_loader, optimizer, scheduler, device, epoch, args, task_names
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        logits, loss = model(input_ids=input_ids, labels=labels)

        if loss is None:
            continue

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
            logger.warning(f"Batch {batch_idx}: NaN gradient, skipping")
            continue

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss_value
        num_batches += 1

        if batch_idx % args.log_interval == 0:
            logger.info(
                f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss_value:.6f}"
            )

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, val_loader, device, task_names):
    model.eval()
    all_losses = {task: 0.0 for task in task_names}
    all_preds = {task: [] for task in task_names}
    all_labels = {task: [] for task in task_names}
    num_batches = 0

    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            logits, loss = model(input_ids=input_ids, labels=labels)

            if loss is not None:
                num_batches += 1

            for task_name in task_names:
                if task_name in logits:
                    pred = logits[task_name].cpu()
                    all_preds[task_name].append(pred)
                if task_name in labels:
                    target = labels[task_name].cpu()
                    all_labels[task_name].append(target)

    metrics = {}
    for task_name in task_names:
        if all_preds[task_name] and all_labels[task_name]:
            preds = torch.cat(all_preds[task_name])
            targets = torch.cat(all_labels[task_name])

            mae = torch.mean(torch.abs(preds - targets)).item()
            mse = torch.mean((preds - targets) ** 2).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()

            metrics[task_name] = {"mae": mae, "mse": mse, "rmse": rmse}

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = parse_task_string(args.tasks)
    logger.info(f"Tasks: {tasks}")

    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, args.val_file) if args.val_file else None
    test_path = os.path.join(args.data_dir, args.test_file) if args.test_file else None

    logger.info(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = create_multitask_loaders(
        train_path=train_path,
        tasks=tasks,
        val_path=val_path,
        test_path=test_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    vocab_size = train_loader.dataset.get_vocab_size()
    pad_token_id = train_loader.dataset.get_pad_token_id()
    logger.info(f"Vocabulary size: {vocab_size}")

    task_names = list(tasks.keys())

    exp_repo = None
    exp_id = None
    if not args.no_db:
        exp_repo = ExperimentRepository(db_path=args.db_path)
        exp_name = args.exp_name or f"{args.dataset}_multitask_{int(time.time())}"
        model_config = {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "pooling": args.pooling,
            "task_strategy": args.task_strategy,
            "vocab_size": vocab_size,
        }
        hyperparams = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "tasks": tasks,
        }
        exp_id = exp_repo.create(
            name=exp_name,
            dataset=args.dataset,
            tasks=task_names,
            model_config=model_config,
            hyperparams=hyperparams,
        )
        logger.info(f"Created experiment: ID={exp_id}, name={exp_name}")

    model = create_multitask_model(
        vocab_size=vocab_size,
        tasks=tasks,
        d_model=args.d_model,
        n_layers=args.n_layers,
        pooling=args.pooling,
        dropout=args.dropout,
        pad_token_id=pad_token_id,
        task_strategy=args.task_strategy,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = (
        len(train_loader) * args.warmup_epochs // args.gradient_accumulation_steps
    )

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("Starting training")
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, f"{args.dataset}_multitask_best.pt")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args, task_names
        )

        val_metrics = (
            evaluate(model, val_loader, device, task_names) if val_loader else {}
        )

        epoch_time = time.time() - start_time

        avg_val_loss = (
            sum(m.get("mae", 0) for m in val_metrics.values()) / len(val_metrics)
            if val_metrics
            else 0
        )

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.2f}s | Train Loss: {train_loss:.6f}"
        )

        for task_name, task_metrics in val_metrics.items():
            logger.info(
                f"  {task_name} - MAE: {task_metrics['mae']:.6f}, RMSE: {task_metrics['rmse']:.6f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                best_model_path,
            )
            logger.info(f"Saved best model to {best_model_path}")

        if exp_repo and exp_id is not None:
            epoch_log = {"epoch": epoch + 1, "train_loss": train_loss}
            for task_name, task_metrics in val_metrics.items():
                for metric_name, metric_value in task_metrics.items():
                    epoch_log[f"{task_name}_{metric_name}"] = metric_value
            exp_repo.append_training_log(exp_id, epoch_log)

    if test_loader:
        logger.info("Evaluating on test set")
        test_metrics = evaluate(model, test_loader, device, task_names)
        for task_name, task_metrics in test_metrics.items():
            logger.info(
                f"Test {task_name} - MAE: {task_metrics['mae']:.6f}, RMSE: {task_metrics['rmse']:.6f}"
            )

    if exp_repo and exp_id is not None:
        final_metrics = {}
        for task_name, task_metrics in test_metrics.items():
            final_metrics[f"test_{task_name}_mae"] = task_metrics["mae"]
            final_metrics[f"test_{task_name}_rmse"] = task_metrics["rmse"]
        exp_repo.complete(exp_id, final_metrics, best_epoch=epoch + 1)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
