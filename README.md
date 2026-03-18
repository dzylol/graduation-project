# Bi-Mamba-Chem: 基于双向状态空间模型的分子性质预测

基于双向 Mamba 架构的分子性质预测模型，支持回归和分类任务。

## 环境配置

### 1. 激活 Python 环境

```bash
# 方法一：使用 conda（推荐）
conda activate my_chem

# 方法二：使用完整路径
/opt/homebrew/Caskroom/miniforge/base/envs/my_chem/bin/python
```

### 2. 安装依赖

```bash
# 使用 conda
conda install pytorch torchvision torchaudio -c pytorch
conda install rdkit pandas scikit-learn -c conda-forge

# 或使用 pip
pip install -r requirements.txt
```

## 项目结构

```
.
├── src/
│   ├── models/
│   │   └── bimamba.py      # Bi-Mamba 模型实现
│   └── data/
│       └── molecule_dataset.py  # 数据处理工具
├── tests/
│   ├── test_model.py       # 模型测试
│   └── test_data.py        # 数据测试
├── data/
│   ├── ESOL/              # ESOL 数据集（回归任务）
│   ├── BBBP/              # BBBP 数据集（分类任务）
│   └── ClinTox/           # ClinTox 数据集（分类任务）
├── checkpoints/           # 模型保存目录
├── train.py               # 训练脚本
├── eval.py                # 评估脚本
└── download_datasets.py   # 数据集下载脚本
```

## 数据准备

### 下载示例数据集

```bash
python download_datasets.py
```

这会创建三个示例数据集：`ESOL`、`BBBP`、`ClinTox`。

### 数据格式

CSV 格式，第一列为 SMILES，第二列为标签：

```csv
smiles,label
CCO,1.5
CC(=O)OC,1.8
c1ccccc1,3.2
```

## 训练指南

### 快速开始

```bash
# 1. 激活环境
conda activate my_chem

# 2. 运行测试确保代码正常
python tests/test_model.py
python tests/test_data.py

# 3. 训练回归模型（ESOL 水溶解度预测）
python train.py \
    --dataset ESOL \
    --data_dir ./data/ESOL \
    --epochs 100 \
    --batch_size 16 \
    --device mps \
    --learning_rate 1e-3

# 4. 训练分类模型（BBBP 血脑屏障渗透）
python train.py \
    --dataset BBBP \
    --data_dir ./data/BBBP \
    --task_type classification \
    --epochs 100 \
    --batch_size 16 \
    --device mps
```

### 训练参数说明

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--dataset` | 必需 | 数据集名称 | ESOL, BBBP, ClinTox |
| `--data_dir` | `./data` | 数据目录 | |
| `--train_file` | `train.csv` | 训练数据文件 | |
| `--val_file` | `val.csv` | 验证数据文件 | |
| `--test_file` | `test.csv` | 测试数据文件 | |
| `--task_type` | `regression` | 任务类型 | `regression` 或 `classification` |
| `--d_model` | 256 | 模型隐藏层维度 | 128-512 |
| `--n_layers` | 4 | Bi-Mamba 层数 | 2-8 |
| `--pooling` | `mean` | 池化方法 | `mean`, `max`, `cls` |
| `--epochs` | 10 | 训练轮数 | 50-200 |
| `--batch_size` | 32 | 批大小 | 8-64 |
| `--learning_rate` | 1e-4 | 学习率 | 1e-3 ~ 1e-5 |
| `--dropout` | 0.1 | Dropout 率 | 0.0-0.3 |
| `--device` | `auto` | 设备 | `cuda`, `mps`, `cpu` |
| `--max_length` | 512 | 最大序列长度 | 128-1024 |

### 设备选择

| 设备 | 说明 | 命令 |
|------|------|------|
| **MPS** (Mac GPU) | Apple Silicon GPU 加速 | `--device mps` |
| **CUDA** (NVIDIA GPU) | NVIDIA GPU 加速 | `--device cuda` |
| **CPU** | CPU 运行 | `--device cpu` |

```bash
# Mac M1/M2/M3 使用 MPS
python train.py --dataset ESOL --device mps --batch_size 16

# 有 NVIDIA GPU 使用 CUDA
python train.py --dataset ESOL --device cuda --batch_size 32

# 调试时使用 CPU
python train.py --dataset ESOL --device cpu --batch_size 4
```

### 完整训练示例

```bash
python train.py \
    --dataset ESOL \
    --data_dir ./data/ESOL \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv \
    --d_model 256 \
    --n_layers 4 \
    --task_type regression \
    --pooling mean \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --dropout 0.1 \
    --device mps \
    --seed 42 \
    --max_length 256
```

## 模型评估

### 评估已训练的模型

```bash
python eval.py \
    --checkpoint checkpoints/ESOL_bi_mamba_best.pt \
    --dataset ESOL \
    --data_dir ./data/ESOL \
    --test_file test.csv \
    --device mps
```

### 评估指标

| 任务类型 | 指标 |
|----------|------|
| 回归任务 | RMSE, MAE, MSE |
| 分类任务 | ROC-AUC, Accuracy |

## 超参数调优建议

### 回归任务 (ESOL)

```bash
python train.py \
    --dataset ESOL \
    --d_model 256 \
    --n_layers 4 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --epochs 100
```

### 分类任务 (BBBP)

```bash
python train.py \
    --dataset BBBP \
    --task_type classification \
    --d_model 256 \
    --n_layers 4 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --epochs 100
```

### 长序列分子

```bash
python train.py \
    --dataset ESOL \
    --d_model 512 \
    --n_layers 6 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --max_length 1024
```

## 常见问题

### Q: 训练出现 NaN 损失怎么办？

**解决方案：**
```bash
# 1. 降低学习率
python train.py --learning_rate 1e-4

# 2. 使用梯度裁剪
python train.py --max_grad_norm 0.5

# 3. 减小 batch_size
python train.py --batch_size 8
```

### Q: GPU 显存不足怎么办？

```bash
# 1. 减小 batch_size
python train.py --batch_size 4

# 2. 减小模型维度
python train.py --d_model 128 --n_layers 2

# 3. 使用梯度累积
python train.py --batch_size 4 --gradient_accumulation_steps 4
```

### Q: 如何加快训练速度？

```bash
# 1. 使用 GPU
python train.py --device mps

# 2. 增加 batch_size
python train.py --batch_size 32

# 3. 减少序列长度
python train.py --max_length 128
```

## 代码测试

```bash
# 测试模型
python tests/test_model.py

# 测试数据处理
python tests/test_data.py
```

## 许可证

MIT License
