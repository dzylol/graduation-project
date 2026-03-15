"""
Bi-Mamba-Chem 训练脚本

本脚本用于训练 Bi-Mamba 分子属性预测模型。

训练流程:
=========
1. 加载数据集 (ESOL, BBBP, ClinTox)
2. 创建模型 (Bi-Mamba)
3. 训练多个 epoch
4. 验证模型性能
5. 保存最佳模型
6. 在测试集上评估

什么是 Epoch?
------------
一个 epoch = 看完一次整个训练数据集

示例:
-----
假设训练集有 1000 个分子
- 1 epoch = 模型学习了 1000 个分子
- 100 epochs = 模型学习了 1000 × 100 = 100000 次

通常需要几十到几百个 epoch 才能让模型收敛。
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

# 导入项目模块
from data import get_dataset, get_task_type
from models import (
    BiMambaForPrediction,
    BiMambaForSequenceClassification,
    BiMambaForRegression,
)


def parse_args():
    """
    解析命令行参数

    这样设计的好处:
    --------------
    不用修改代码,直接通过命令行参数调整训练设置

    示例:
    -----
    python train.py --dataset ESOL --epochs 100 --batch_size 32
    python train.py --dataset BBBP --d_model 512 --n_layers 6
    """
    parser = argparse.ArgumentParser(
        description='训练 Bi-Mamba-Chem 分子属性预测模型'
    )

    # ====== 数据集相关参数 ======
    parser.add_argument(
        '--dataset',
        type=str,
        default='ESOL',
        choices=['ESOL', 'BBBP', 'CLINTOX'],
        help='数据集名称: ESOL(溶解度回归), BBBP(血脑屏障分类), CLINTOX(毒性分类)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='数据目录'
    )

    # ====== 模型相关参数 ======
    parser.add_argument(
        '--model',
        type=str,
        default='bi_mamba',
        choices=['bi_mamba', 'bi_ssm'],
        help='模型架构'
    )
    parser.add_argument(
        '--use_ssm',
        action='store_true',
        help='是否使用手动实现的 SSM (不安装 mamba-ssm 时使用)'
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=256,
        help='模型隐藏维度: 每个 token 的向量表示长度'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='模型层数: 有多少个 Bi-Mamba Block'
    )
    parser.add_argument(
        '--d_state',
        type=int,
        default=128,
        help='SSM 状态维度: 模型记忆单元的数量'
    )
    parser.add_argument(
        '--d_conv',
        type=int,
        default=4,
        help='卷积核大小: 局部上下文的大小'
    )
    parser.add_argument(
        '--expand',
        type=int,
        default=2,
        help='扩展因子: 内部维度的扩展倍数'
    )

    # ====== 训练相关参数 ======
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮数: 完整遍历训练集的次数'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批大小: 每次同时处理的分子数量'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='数据加载的并行进程数'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='梯度累积步数: 用于增大有效批大小'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='学习率: 每次参数更新的步长'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='权重衰减: L2 正则化,防止过拟合'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout 概率: 随机丢弃神经元的比例'
    )

    # ====== 模型架构参数 ======
    parser.add_argument(
        '--fusion',
        type=str,
        default='gate',
        choices=['concat', 'add', 'gate'],
        help='双向融合策略: concat(拼接), add(相加), gate(门控推荐)'
    )
    parser.add_argument(
        '--pool_type',
        type=str,
        default='mean',
        choices=['mean', 'max', 'cls'],
        help='池化方式: mean(平均), max(最大), cls(CLS token)'
    )

    # ====== 其他参数 ======
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='最大序列长度: SMILES 分词后的最大长度'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子: 保证结果可复现'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='设备: auto(自动), cpu(只用CPU), cuda(NVIDIA GPU), mps(Apple GPU)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='模型保存目录'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='日志打印间隔'
    )
    parser.add_argument(
        '--save_best',
        action='store_true',
        help='是否只保存最佳模型'
    )

    return parser.parse_args()


def set_seed(seed: int):
    """
    设置随机种子

    为什么需要随机种子?
    ------------------
    深度学习涉及很多随机操作:
    - 参数初始化
    - Dropout
    - 数据 shuffle

    设置相同的种子可以让每次训练结果相同,方便调试和复现!
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA 和 MPS 都设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_str: str) -> torch.device:
    """
    获取计算设备

    自动选择:
    ----------
    - Apple Silicon (M1/M2/M3): 使用 MPS (Metal Performance Shaders)
    - NVIDIA GPU: 使用 CUDA
    - 其他: 使用 CPU
    """
    if device_str == 'auto':
        # 优先顺序: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def detect_and_select_device() -> str:
    """
    检测可用设备并让用户选择

    返回:
    -----
    str: 选择的设备字符串 ('cpu', 'cuda', 'mps')
    """
    print("\n" + "=" * 60)
    print("检测可用计算设备...")
    print("=" * 60)

    # 检测各设备可用性
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    devices = []
    device_info = {}

    # CPU 始终可用
    devices.append('cpu')
    device_info['cpu'] = 'CPU (中央处理器)'

    if cuda_available:
        devices.append('cuda')
        device_info['cuda'] = f"CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})"

    if mps_available:
        devices.append('mps')
        device_info['mps'] = "MPS (Apple Silicon GPU: M1/M2/M3)"

    # 显示可用设备
    print("\n可用的设备:")
    print("-" * 40)
    for i, dev in enumerate(devices):
        print(f"  {i + 1}. {device_info[dev]}")
    print("-" * 40)

    # 如果只有一个设备,直接使用
    if len(devices) == 1:
        print(f"\n仅检测到 {device_info[devices[0]]}, 将自动使用。")
        return devices[0]

    # 让用户选择
    print("\n自动选择设备:")
    print("  0. auto (自动选择最快设备)")

    while True:
        try:
            choice = input("\n请选择设备 [0-{}]: ".format(len(devices))).strip()

            if choice == '':
                # 默认选择最快的设备
                if mps_available:
                    selected = 'mps'
                elif cuda_available:
                    selected = 'cuda'
                else:
                    selected = 'cpu'
                print(f"已选择: {device_info[selected]}")
                return selected

            choice = int(choice)
            if choice < 0 or choice > len(devices):
                print("无效选择,请重试!")
                continue

            if choice == 0:
                # auto: 选择最快设备
                if mps_available:
                    selected = 'mps'
                elif cuda_available:
                    selected = 'cuda'
                else:
                    selected = 'cpu'
                print(f"自动选择最快设备: {device_info[selected]}")
                return selected
            else:
                selected = devices[choice - 1]
                print(f"已选择: {device_info[selected]}")
                return selected

        except ValueError:
            print("请输入有效的数字!")


def compute_metrics(predictions, labels, task_type: str):
    """
    计算评估指标

    回归任务指标 (ESOL):
    --------------------
    - RMSE (均方根误差): 预测值与真实值的偏差
    - MAE (平均绝对误差): 预测误差的平均值
    - R² (决定系数): 模型解释数据的能力,1 为完美

    分类任务指标 (BBBP, ClinTox):
    -----------------------------
    - Accuracy (准确率): 预测正确的比例
    - ROC-AUC: 分类器区分能力的指标
    """
    if task_type == 'regression':
        # ====== 回归任务 ======
        # RMSE: 预测误差的平方的均值,再开根号
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)

        # MAE: 预测误差绝对值的均值
        mae = np.mean(np.abs(predictions - labels))

        # R²: 模型解释的方差比例
        ss_res = np.sum((labels - predictions) ** 2)  # 残差平方和
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)  # 总平方和
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
    else:
        # ====== 分类任务 ======
        # 把概率转换为 0/1 标签
        pred_labels = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_labels == labels)

        # ROC-AUC
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
    gradient_accumulation_steps: int = 1,
):
    """
    训练一个 epoch

    训练流程 (一个 batch):
    ---------------------
    1. 把数据加载到 GPU
    2. 前向传播: 输入 -> 模型 -> 输出
    3. 计算损失: 比较输出和真实标签
    4. 反向传播: 计算梯度
    5. 梯度裁剪: 防止梯度爆炸
    6. 更新参数: 优化器调整参数

    梯度累积:
    ---------
    当显存不足时,可以使用梯度累积来模拟更大的批大小。
    例如: batch_size=16, gradient_accumulation_steps=4, 则有效批大小为 64。

    参数:
    -----
    model: 要训练的模型
    dataloader: 训练数据
    optimizer: 优化器
    criterion: 损失函数
    device: 计算设备
    task_type: 任务类型
    gradient_accumulation_steps: 梯度累积步数

    返回:
    -----
    包含训练指标 (loss, metrics) 的字典
    """
    model.train()  # 开启训练模式 (启用 dropout 等)
    total_loss = 0
    all_preds = []
    all_labels = []

    # 梯度累积计数器
    accumulation_counter = 0

    # tqdm 进度条
    pbar = tqdm(dataloader, desc='Training')

    for batch in pbar:
        # ====== 1. 获取数据 ======
        input_ids, labels = batch
        input_ids = input_ids.to(device)  # 移到 GPU
        labels = labels.to(device)

        # ====== 2. 前向传播 ======
        # 清零梯度 (重要!)
        # 只有在累积计数为0时才清零梯度
        if accumulation_counter == 0:
            optimizer.zero_grad()

        # 模型前向计算
        outputs = model(input_ids)

        # ====== 3. 计算损失 ======
        # squeeze(-1) 把最后多余的维度去掉
        # 例如: (32, 1) -> (32,)
        if task_type == 'classification':
            loss = criterion(outputs.squeeze(-1), labels)
        else:
            loss = criterion(outputs.squeeze(-1), labels)

        # 缩放损失以便梯度累积
        loss = loss / gradient_accumulation_steps

        # ====== 4. 反向传播 ======
        loss.backward()

        # ====== 5. 梯度累积 ======
        accumulation_counter += 1

        # 当累积到指定步数时,更新参数
        if accumulation_counter >= gradient_accumulation_steps:
            # 梯度裁剪: 防止梯度太大导致训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ====== 6. 更新参数 ======
            optimizer.step()

            # 重置累积计数器
            accumulation_counter = 0

        # ====== 7. 收集预测结果 ======
        # 使用原始损失值(未缩放)用于显示
        display_loss = loss.item() * gradient_accumulation_steps

        if task_type == 'classification':
            # 分类任务: 用 sigmoid 转换为概率
            preds = torch.sigmoid(outputs.squeeze(-1)).detach().cpu().numpy()
        else:
            # 回归任务: 直接用原始输出
            preds = outputs.squeeze(-1).detach().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        total_loss += display_loss

        # 更新进度条显示
        pbar.set_postfix({'loss': display_loss})

    # ====== 8. 计算指标 ======
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
    评估模型

    与训练的区别:
    ------------
    - 不计算梯度 (节省显存和计算)
    - 不更新参数
    - 使用 eval() 模式 (禁用 dropout)

    参数: 同 train_epoch

    返回:
    -----
    包含验证指标 (loss, metrics) 的字典
    """
    model.eval()  # 开启评估模式 (禁用 dropout)
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        # 获取数据
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(input_ids)

        # 计算损失
        if task_type == 'classification':
            loss = criterion(outputs.squeeze(-1), labels)
            preds = torch.sigmoid(outputs.squeeze(-1))
        else:
            loss = criterion(outputs.squeeze(-1), labels)
            preds = outputs.squeeze(-1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels, task_type)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def main():
    """主训练函数"""
    # ====== 1. 解析参数 ======
    args = parse_args()

    # ====== 2. 设置随机种子 ======
    set_seed(args.seed)

    # ====== 3. 获取设备 ======
    # 如果是 auto 模式,让用户选择设备
    if args.device == 'auto':
        device_str = detect_and_select_device()
    else:
        device_str = args.device

    device = get_device(device_str)
    print(f"使用设备: {device}")

    # ====== 4. 创建保存目录 ======
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ====== 5. 加载数据集 ======
    print(f"\n加载 {args.dataset} 数据集...")
    train_dataset, val_dataset, test_dataset, tokenizer = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        max_length=args.max_length,
    )

    # 获取任务类型
    task_type = get_task_type(args.dataset)
    print(f"任务类型: {task_type}")

    # ====== 6. 创建 DataLoader ======
    # DataLoader 负责把数据集分成批次
    # 使用 num_workers 并行加载数据,pin_memory 加速 GPU 传输
    # 注意: MPS 不支持 pin_memory,需要排除
    use_pin_memory = device.type in ('cuda', 'cpu')  # MPS 不支持 pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # 打印数据加载信息
    print(f"数据加载: num_workers={args.num_workers}, pin_memory={use_pin_memory}")
    print(f"批大小: {args.batch_size}, 梯度累积: {args.gradient_accumulation_steps}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"有效批大小: {effective_batch_size}")

    # ====== 7. 创建模型 ======
    print(f"\n创建 Bi-Mamba 模型...")
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
    model = model.to(device)  # 移到 GPU

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # ====== 8. 损失函数 ======
    if task_type == 'classification':
        # 分类: 二元交叉熵损失
        criterion = nn.BCEWithLogitsLoss()
    else:
        # 回归: 均方误差损失
        criterion = nn.MSELoss()

    # ====== 9. 优化器 ======
    # AdamW: 带权重衰减的 Adam 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ====== 10. 学习率调度器 ======
    # 余弦退火: 让学习率慢慢下降
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # ====== 11. 训练循环 ======
    # 记录最佳验证指标
    best_val_metric = float('inf') if task_type == 'regression' else 0.0
    best_epoch = 0

    print(f"\n开始训练 ({args.epochs} 个 epoch)...")
    print("-" * 80)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # 训练一个 epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, task_type,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # 验证
        val_metrics = evaluate(
            model, val_loader, criterion, device, task_type
        )

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 打印结果
        if task_type == 'regression':
            print(
                f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
                f"训练 Loss: {train_metrics['loss']:.4f} | "
                f"验证 Loss: {val_metrics['loss']:.4f} | "
                f"验证 RMSE: {val_metrics['rmse']:.4f} | "
                f"验证 R²: {val_metrics['r2']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
                f"训练 Loss: {train_metrics['loss']:.4f} | "
                f"验证 Loss: {val_metrics['loss']:.4f} | "
                f"验证准确率: {val_metrics['accuracy']:.4f} | "
                f"验证 AUC: {val_metrics['auc']:.4f}"
            )

        # ====== 12. 保存最佳模型 ======
        # 判断是否是最佳模型
        if task_type == 'regression':
            # 回归: RMSE 越低越好
            is_best = val_metrics['rmse'] < best_val_metric
        else:
            # 分类: AUC 越高越好
            is_best = val_metrics['auc'] > best_val_metric

        if is_best:
            # 更新最佳指标
            best_val_metric = (val_metrics['rmse'] if task_type == 'regression'
                               else val_metrics['auc'])
            best_epoch = epoch + 1

            # 保存模型
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
            print(f"  -> 最佳模型已保存! (验证 {task_type}: {best_val_metric:.4f})")

    print("-" * 80)

    # ====== 13. 最终测试 ======
    print("\n加载最佳模型进行最终评估...")
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.dataset}_bi_mamba_best.pt')

    # 如果 checkpoint 不存在,跳过最终测试
    if not os.path.exists(checkpoint_path):
        print("未找到最佳模型checkpoint,跳过最终测试")
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            test_metrics = evaluate(model, test_loader, criterion, device, task_type)

            print(f"\n最终测试结果 (epoch {best_epoch}):")
            if task_type == 'regression':
                print(f"  RMSE: {test_metrics['rmse']:.4f}")
                print(f"  MAE: {test_metrics['mae']:.4f}")
                print(f"  R²: {test_metrics['r2']:.4f}")
            else:
                print(f"  准确率: {test_metrics['accuracy']:.4f}")
                print(f"  ROC-AUC: {test_metrics['auc']:.4f}")
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
            print("跳过最终测试")

    print(f"\n模型已保存至: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
