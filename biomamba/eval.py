"""
评估脚本 - 用于在测试集上评估训练好的Bi-Mamba模型

使用方式:
    python eval.py --checkpoint checkpoints/ESOL_bi_mamba_best.pt --dataset ESOL

本脚本会:
1. 加载训练好的模型检查点
2. 在指定的数据集(训练集/验证集/测试集)上进行预测
3. 计算并显示评估指标
4. (可选)保存预测结果到文件
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import project modules
from data import get_dataset, get_task_type
from models import BiMambaForPrediction
from utils.metrics import compute_metrics, format_metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='评估Bi-Mamba-Chem分子属性预测模型'
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
        help='Device (auto, cpu, cuda, mps)'
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
    """
    设置随机种子以确保结果可复现

    随机种子使得每次运行程序时,随机操作(如参数初始化、数据打乱)产生相同的结果,
    这对于调试和对比实验非常重要。

    Args:
        seed: 随机种子值,通常使用42
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_str: str) -> torch.device:
    """
    根据字符串获取PyTorch设备

    Args:
        device_str: 设备字符串,可选'auto', 'cpu', 'cuda', 'mps'

    Returns:
        PyTorch设备对象
    """
    if device_str == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def detect_and_select_device() -> str:
    """
    自动检测可用的计算设备并让用户选择

    本函数会检测:
    - CUDA: NVIDIA GPU (如RTX 3090, A100等)
    - MPS: Apple Silicon GPU (如M1, M2, M3芯片)
    - CPU: 通用处理器

    Returns:
        选中的设备字符串 ('cpu', 'cuda', 或 'mps')
    """
    print("\n" + "=" * 60)
    print("Detecting available compute devices...")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    devices = []
    device_info = {}

    devices.append('cpu')
    device_info['cpu'] = 'CPU'

    if cuda_available:
        devices.append('cuda')
        device_info['cuda'] = f"CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})"

    if mps_available:
        devices.append('mps')
        device_info['mps'] = "MPS (Apple Silicon GPU)"

    print("\nAvailable devices:")
    for i, dev in enumerate(devices):
        print(f"  {i + 1}. {device_info[dev]}")

    if len(devices) == 1:
        return devices[0]

    print("\n  0. auto (fastest)")
    while True:
        try:
            choice = input("\nSelect device [0-{}]: ".format(len(devices))).strip()
            choice = int(choice) if choice else 0
            if 0 <= choice <= len(devices):
                return ['mps', 'cuda', 'cpu'][choice] if choice == 0 else devices[choice - 1]
        except ValueError:
            pass


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_type: str,
):
    """
    对数据集进行预测

    遍历数据加载器中的所有批次,使用训练好的模型进行预测。

    Args:
        model: 训练好的PyTorch模型
        dataloader: 数据加载器
        device: 计算设备
        task_type: 任务类型,'regression'(回归)或'classification'(分类)

    Returns:
        predictions: 模型预测值数组
        labels: 真实标签/值数组
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
    """
    评估主函数

    评估流程:
    1. 解析命令行参数
    2. 设置随机种子
    3. 选择计算设备
    4. 加载模型检查点
    5. 加载数据集
    6. 创建模型并加载权重
    7. 在指定数据上运行预测
    8. 计算评估指标
    9. 显示并保存结果
    """
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device - interactive selection
    if args.device == 'auto':
        device_str = detect_and_select_device()
    else:
        device_str = args.device

    device = get_device(device_str)
    print(f"Using device: {device}")

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

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
