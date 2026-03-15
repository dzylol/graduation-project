"""
Training script for Bi-Mamba-Chem molecular property prediction.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import project modules
from biomamba.data import get_dataset, get_task_type
from biomamba.models import (
    BiMambaForPrediction,
    BiMambaForSequenceClassification,
    BiMambaForRegression,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Bi-Mamba-Chem for molecular property prediction'
    )

    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='ESOL',
        choices=['ESOL', 'BBBP', 'CLINTOX'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Data directory'
    )

    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='bi_mamba',
        choices=['bi_mamba', 'bi_ssm'],
        help='Model architecture'
    )
    parser.add_argument(
        '--use_ssm',
        action='store_true',
        help='Use manual SSM implementation instead of mamba-ssm'
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=256,
        help='Model dimension'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers'
    )
    parser.add_argument(
        '--d_state',
        type=int,
        default=128,
        help='SSM state dimension'
    )
    parser.add_argument(
        '--d_conv',
        type=int,
        default=4,
        help='Convolution kernel size'
    )
    parser.add_argument(
        '--expand',
        type=int,
        default=2,
        help='Expansion factor'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )

    # Architecture
    parser.add_argument(
        '--fusion',
        type=str,
        default='gate',
        choices=['concat', 'add', 'gate'],
        help='Bidirectional fusion strategy'
    )
    parser.add_argument(
        '--pool_type',
        type=str,
        default='mean',
        choices=['mean', 'max', 'cls'],
        help='Pooling type'
    )

    # Other
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='Log interval'
    )
    parser.add_argument(
        '--save_best',
        action='store_true',
        help='Save best model only'
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def compute_metrics(predictions, labels, task_type: str):
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_type: 'regression' or 'classification'

    Returns:
        Dictionary of metrics
    """
    if task_type == 'regression':
        # RMSE
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        # MAE
        mae = np.mean(np.abs(predictions - labels))
        # R2
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
    else:
        # Classification
        pred_labels = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_labels == labels)

        # ROC-AUC (for binary classification)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, predictions)
        except:
            auc = 0.0

        return {
            'accuracy': accuracy,
            'auc': auc,
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
):
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        task_type: 'regression' or 'classification'

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(input_ids)

        # Loss
        if task_type == 'classification':
            loss = criterion(outputs.squeeze(-1), labels)
        else:
            loss = criterion(outputs.squeeze(-1), labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Collect predictions
        if task_type == 'classification':
            preds = torch.sigmoid(outputs.squeeze(-1)).detach().cpu().numpy()
        else:
            preds = outputs.squeeze(-1).detach().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

        pbar.set_postfix({'loss': loss.item()})

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels, task_type)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
):
    """
    Evaluate model.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device
        task_type: 'regression' or 'classification'

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(input_ids)

        # Loss
        if task_type == 'classification':
            loss = criterion(outputs.squeeze(-1), labels)
            preds = torch.sigmoid(outputs.squeeze(-1))
        else:
            loss = criterion(outputs.squeeze(-1), labels)
            preds = outputs.squeeze(-1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels, task_type)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset, val_dataset, test_dataset, tokenizer = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        max_length=args.max_length,
    )

    task_type = get_task_type(args.dataset)
    print(f"Task type: {task_type}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model
    print(f"\nCreating Bi-Mamba model...")
    model = BiMambaForPrediction(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        use_mamba=not args.use_ssm,
        fusion=args.fusion,
        max_len=args.max_length,
        pool_type=args.pool_type,
        task_type=task_type,
    )
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    if task_type == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # Training loop
    best_val_metric = float('inf') if task_type == 'regression' else 0.0
    best_epoch = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, task_type
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, task_type
        )

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print metrics
        if task_type == 'regression':
            print(
                f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Val R2: {val_metrics['r2']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )

        # Save best model
        if task_type == 'regression':
            is_best = val_metrics['rmse'] < best_val_metric
        else:
            is_best = val_metrics['auc'] > best_val_metric

        if is_best:
            best_val_metric = val_metrics['rmse'] if task_type == 'regression' else val_metrics['auc']
            best_epoch = epoch + 1

            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'{args.dataset}_bi_mamba_best.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  -> Best model saved! (Val {task_type}: {best_val_metric:.4f})")

    print("-" * 80)

    # Final evaluation on test set
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, f'{args.dataset}_bi_mamba_best.pt')
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, task_type)

    print(f"\nFinal Test Results (epoch {best_epoch}):")
    if task_type == 'regression':
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  R2: {test_metrics['r2']:.4f}")
    else:
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {test_metrics['auc']:.4f}")

    print(f"\nCheckpoint saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
