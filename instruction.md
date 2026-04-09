# Bi-Mamba-Chem 训练与评估完整指南

本文档说明如何从零开始训练和评估分子性质预测模型。任何 AI 只需按照以下步骤执行即可完成训练。

---

## 目录

- [第一步：连接远程服务器](#第一步连接远程服务器)
- [第二步：同步代码](#第二步同步代码)
- [第三步：数据分割（重要！）](#第三步数据分割重要)
- [第四步：开始训练](#第四步开始训练)
- [第五步：评估模型](#第五步评估模型)
- [完整示例命令](#完整示例命令)

---

## 第一步：连接远程服务器

### SSH 连接

```bash
ssh qfh@6.tcp.cpolar.cn -p 13234
```

进入后切换到项目目录：

```bash
cd ~/graduation-project
```

---

## 第二步：同步代码

每次开始新实验前，先拉取最新代码：

```bash
git pull origin main
```

### 验证环境

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace localhost/bimamba bash -c "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.version.cuda}\")'"
```

预期输出：
```
PyTorch: 2.x.x
CUDA: 12.x
```

---

## 第三步：数据分割（重要！）

### 重要说明

**数据集只有一个 CSV 文件（如 `delaney.csv`），需要先分割成 train/val/test 三部分。**

每次实验使用不同的 random seed，可以获得不同的数据分割，增强实验多样性。

### 自动分割（推荐）

使用以下脚本自动完成数据分割和训练：

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace localhost/bimamba bash -c "
python -c \"
from src.data.molecule_dataset import random_split_dataset, get_next_split_seed
import os

# 自动获取 seed（每次实验自动递增）
seed = get_next_split_seed()
print(f'使用 seed: {seed}')

# 分割数据（ESOL 示例）
train, val, test = random_split_dataset(
    'dataset/ESOL/delaney.csv',
    output_dir='dataset/ESOL/',
    seed=seed
)
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
\"
"
```

### 分割参数说明

```python
random_split_dataset(
    input_csv="dataset/ESOL/delaney.csv",  # 原始数据文件
    output_dir="dataset/ESOL/",            # 输出目录
    train_ratio=0.8,                       # 训练集比例
    val_ratio=0.1,                        # 验证集比例
    test_ratio=0.1,                       # 测试集比例
    seed=42,                              # 随机种子（相同 seed 产生相同分割）
    n_jobs=None                           # None=使用全部 CPU 核心加速
)
```

### 手动分割示例

如果需要更细粒度控制：

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace localhost/bimamba bash -c "
python -c \"
from src.data.molecule_dataset import random_split_dataset, get_next_split_seed

# 获取并递增 seed
seed = get_next_split_seed()
print(f'当前 seed: {seed}')

# 分割 ZINC250K 数据集
train, val, test = random_split_dataset(
    'dataset/ZINC250K/250k_rndm_zinc_drugs_clean_3.csv',
    output_dir='dataset/ZINC250K/',
    seed=seed,
    n_jobs=None  # 使用全部 CPU 核心
)
print(f'ZINC250K: Train={len(train)}, Val={len(val)}, Test={len(test)}')
\"
"
```

### 验证分割结果

```bash
ls -la dataset/ESOL/*.csv
```

预期：
```
train.csv  val.csv  test.csv
```

---

## 第四步：开始训练

### 基础训练命令

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python train.py \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv \
    --model_type mamba_ssm \
    --task_type regression \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --device cuda \
    --no_db"
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | ✅ | - | 数据集名称（用于日志） |
| `--data_dir` | ✅ | - | 数据目录 |
| `--train_file` | ✅ | - | 训练数据文件名 |
| `--val_file` | ✅ | - | 验证数据文件名 |
| `--test_file` | ✅ | - | 测试数据文件名 |
| `--model_type` | ✅ | - | **必须设为 `mamba_ssm`** |
| `--task_type` | | `regression` | `regression` 或 `classification` |
| `--epochs` | | `10` | 训练轮数 |
| `--batch_size` | | `32` | 批大小 |
| `--learning_rate` | | `1e-4` | 学习率 |
| `--device` | | `cuda` | 设备：`cuda` / `mps` / `cpu` |
| `--no_db` | | - | 禁用数据库记录（推荐） |

### 不同数据集配置

#### ESOL（回归任务）

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python train.py \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv \
    --model_type mamba_ssm \
    --task_type regression \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --device cuda \
    --no_db"
```

#### BBBP（分类任务）

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python train.py \
    --dataset BBBP \
    --data_dir ./dataset/BBBP \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv \
    --model_type mamba_ssm \
    --task_type classification \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --device cuda \
    --no_db"
```

#### ZINC250K（大规模回归）

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python train.py \
    --dataset ZINC250K \
    --data_dir ./dataset/ZINC250K \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv \
    --model_type mamba_ssm \
    --task_type regression \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --device cuda \
    --no_db"
```

### 显存不足？

减小 batch_size：

```bash
--batch_size 8 \
--d_model 128 \
--n_layers 2 \
```

---

## 第五步：评估模型

训练完成后会生成 `./checkpoints/ESOL_bi_mamba_best.pt`

### 评估命令

```bash
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python eval.py \
    --checkpoint ./checkpoints/ESOL_bi_mamba_best.pt \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --test_file test.csv \
    --model_type mamba_ssm \
    --device cuda"
```

---

## 完整示例命令

### 一次性执行（推荐）

复制以下命令，按顺序执行即可完成整个训练流程：

```bash
# 1. 连接远程服务器
ssh qfh@6.tcp.cpolar.cn -p 13234

# 2. 进入项目目录
cd ~/graduation-project

# 3. 同步最新代码
git pull origin main

# 4. 数据分割
podman run --rm -v "$(pwd):/workspace" --workdir /workspace localhost/bimamba \
  bash -c "python -c \"
from src.data.molecule_dataset import random_split_dataset, get_next_split_seed
seed = get_next_split_seed()
print(f'Seed: {seed}')
train, val, test = random_split_dataset(
    'dataset/ESOL/delaney.csv',
    output_dir='dataset/ESOL/',
    seed=seed
)
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
\""

# 5. 开始训练
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python train.py \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv \
    --model_type mamba_ssm \
    --task_type regression \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --device cuda \
    --no_db"

# 6. 评估（训练完成后）
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python eval.py \
    --checkpoint ./checkpoints/ESOL_bi_mamba_best.pt \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --test_file test.csv \
    --model_type mamba_ssm \
    --device cuda"
```

### 显存优化配置（RTX 5060 Ti 16GB）

```bash
# 大 batch 训练
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  bash -c "python train.py \
    --dataset Lipophilicity \
    --data_dir ./dataset/Lipophilicity \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv \
    --model_type mamba_ssm \
    --task_type regression \
    --epochs 100 \
    --batch_size 256 \
    --num_workers 16 \
    --learning_rate 1e-3 \
    --device cuda \
    --no_db"
```

---

## 常见问题

### 1. 训练很慢？
- 小数据集（ESOL 1K）GPU 利用率呈脉冲式是正常的
- 大数据集（ZINC250K 250K）GPU 会持续满载

### 2. 显存不足？
```bash
--batch_size 8
--d_model 128
--n_layers 2
```

### 3. 需要重新分割数据？
每次实验前重新执行数据分割步骤，seed 会自动递增。

### 4. 如何复现之前的实验？
查看 `.split_seed` 文件中的 seed 值，使用相同 seed 重新分割数据。

---

## 文件结构

```
graduation-project/
├── dataset/
│   ├── ESOL/
│   │   ├── delaney.csv      # 原始数据（不要修改）
│   │   ├── train.csv        # 分割后训练集
│   │   ├── val.csv          # 分割后验证集
│   │   └── test.csv         # 分割后测试集
│   └── ...
├── checkpoints/
│   └── *_best.pt            # 训练好的模型
├── src/
│   └── data/
│       └── molecule_dataset.py  # 数据处理函数
├── train.py
├── eval.py
└── .split_seed              # 自动管理的 seed 文件
```
