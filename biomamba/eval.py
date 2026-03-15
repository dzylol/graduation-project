"""
Evaluation script for Bi-Mamba-Chem molecular property prediction.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import project modules
from biomamba.data import get_dataset, get_task_type
from biomamba.models import BiMambaForPrediction
from biomamba.utils.metrics import compute_metrics, format_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Bi-Mamba-Chem for molecular property prediction'
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

    # Model checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    # Model architecture (must match checkpoint)
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
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
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
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
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
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for predictions'
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


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_type: str,
):
    """
    Make predictions.

    Args:
        model: Model
        dataloader: DataLoader
        device: Device
        task_type: 'regression' or 'classification'

    Returns:
        Tuple of (predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Predicting'):
        input_ids, labels = batch
        input_ids = input_ids.to(device)

        # Forward
        outputs = model(input_ids)

        # Get predictions
        if task_type == 'classification':
            preds = torch.sigmoid(outputs.squeeze(-1))
        else:
            preds = outputs.squeeze(-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, val_dataset, test_dataset, tokenizer = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        max_length=args.max_length,
    )

    task_type = get_task_type(args.dataset)
    print(f"Task type: {task_type}")

    # Select split
    if args.split == 'train':
        eval_dataset = train_dataset
    elif args.split == 'val':
        eval_dataset = val_dataset
    else:
        eval_dataset = test_dataset

    print(f"Evaluating on {args.split} split: {len(eval_dataset)} samples")

    # Create dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model
    print(f"\nCreating model...")
    model = BiMambaForPrediction(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        use_mamba=True,
        fusion=args.fusion,
        max_len=args.max_length,
        pool_type=args.pool_type,
        task_type=task_type,
    )
    model = model.to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights from full checkpoint")

    # Make predictions
    print(f"\nMaking predictions...")
    predictions, labels = predict(model, eval_loader, device, task_type)

    # Compute metrics
    if task_type == 'classification':
        pred_labels = (predictions > 0.5).astype(int)
        metrics = compute_metrics(labels, pred_labels, task_type, predictions)
    else:
        metrics = compute_metrics(labels, predictions, task_type)

    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({args.split} split)")
    print("=" * 50)
    print(format_metrics(metrics, task_type))

    # Save predictions if requested
    if args.output is not None:
        output_path = args.output
        np.savez(
            output_path,
            predictions=predictions,
            labels=labels,
        )
        print(f"\nPredictions saved to: {output_path}")

    print("=" * 50)


if __name__ == "__main__":
    main()
