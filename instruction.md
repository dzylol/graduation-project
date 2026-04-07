# Bi-Mamba-Chem 训练与评估指南

本文档说明如何使用 `podman image localhost/bimambaa` 环境进行模型训练与评估。**使用 `bimamba_with_mamba_ssm.py` 实现（非人工实现的 `bimamba.py`）**。

---

## 目录

- [环境准备](#环境准备)
- [下载数据集](#下载数据集)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [常用命令参考](#常用命令参考)

---

## 环境准备

### 1. 连接远程机器

使用 cpolar-tunnel SSH 隧道连接远程机器：

```bash
ssh cpolar-tunnel
```

### 2. 检查 Podman 镜像

在远程机器上执行：

```bash
podman images | grep bimamb
```

确认 `localhost/bimamba` 镜像存在。如果没有，请先构建：

```bash
# 在项目根目录执行（需要 Dockerfile）
podman build -t localhost/bimamba .
```

### 3. 挂载项目目录

训练产生的模型权重、日志等文件需要持久化到主机目录。运行容器时使用 `-v` 挂载：

**远程机器上执行**：

```bash
# 项目目录路径
REMOTE_PROJECT_DIR="/home/qfh/graduation-project"

podman run --rm -it \
  -v "${REMOTE_PROJECT_DIR}:/workspace" \
  --workdir /workspace \
  --device nvidia.com/gpu=all \
  localhost/bimamba \
  bash
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--rm` | 容器退出后自动删除 |
| `-it` | 交互式终端 |
| `-v` | 挂载主机目录到容器内 `/workspace` |
| `--workdir` | 容器内工作目录 |
| `--device nvidia.com/gpu=all` | GPU 直通 (NVIDIA GeForce RTX 5060 Ti) |

### 4. 验证环境

容器内执行：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
python -c "import mamba_ssm; print('mamba_ssm OK')"
python -c "from src.models.bimamba_with_mamba_ssm import BiMambaForPropertyPrediction; print('BiMambaWithMambaSSM OK')"
```

---

## 下载数据集

### 使用 DeepChem 下载完整数据集（推荐）

```bash
python download_datasets.py
```

这会下载以下 MoleculeNet 数据集：

| 数据集 | 任务类型 | 分子数 | 评价指标 |
|--------|----------|--------|----------|
| ESOL | 回归 | 1,128 | RMSE |
| BBBP | 分类 | 2,039 | ROC-AUC |
| ClinTox | 分类 | 1,478 | ROC-AUC |
| FreeSolv | 回归 | 642 | RMSE |
| Lipophilicity | 回归 | 4,200 | RMSE |
| SIDER | 分类 | 1,427 | ROC-AUC |

### 下载特定数据集

```bash
python download_datasets.py --dataset ESOL
```

### 强制使用示例数据（无需 DeepChem）

```bash
python download_datasets.py --example
```

数据集下载后保存在 `./data/` 目录，结构如下：

```
data/
├── ESOL/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── meta.json
├── BBBP/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── meta.json
└── ...
```

---

## 模型训练

### 基础训练命令

**关键参数 `--model_type mamba_ssm`**：使用 `bimamba_with_mamba_ssm.py` 实现（基于 mamba_ssm 库）。

#### ESOL 回归任务

```bash
python train.py \
  --dataset ESOL \
  --data_dir ./data/ESOL \
  --model_type mamba_ssm \
  --task_type regression \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --device cuda \
  --d_model 256 \
  --n_layers 4 \
  --pooling mean \
  --no_db
```

#### BBBP 分类任务

```bash
python train.py \
  --dataset BBBP \
  --data_dir ./data/BBBP \
  --model_type mamba_ssm \
  --task_type classification \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --device cuda \
  --no_db
```

### 完整参数列表

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | ✅ | - | 数据集名称（如 ESOL, BBBP） |
| `--data_dir` | | `./data` | 数据文件目录 |
| `--model_type` | | `manual` | **必须设置为 `mamba_ssm`** |
| `--task_type` | | `regression` | `regression` 或 `classification` |
| `--train_file` | | `train.csv` | 训练数据文件名 |
| `--val_file` | | `val.csv` | 验证数据文件名 |
| `--test_file` | | `test.csv` | 测试数据文件名 |
| `--d_model` | | `256` | 模型隐藏层维度 |
| `--n_layers` | | `4` | BiMamba 层数 |
| `--pooling` | | `mean` | 池化方式：`mean` / `max` / `cls` |
| `--dropout` | | `0.1` | Dropout 概率 |
| `--epochs` | | `10` | 训练轮数 |
| `--batch_size` | | `32` | 批大小 |
| `--learning_rate` | | `1e-4` | 学习率 |
| `--weight_decay` | | `1e-5` | 权重衰减 |
| `--max_grad_norm` | | `1.0` | 梯度裁剪范数 |
| `--gradient_accumulation_steps` | | `1` | 梯度累积步数 |
| `--warmup_epochs` | | `5` | 学习率预热轮数 |
| `--device` | | `auto` | `cuda` / `mps` / `cpu` / `auto` |
| `--max_length` | | `512` | 最大序列长度 |
| `--seed` | | `42` | 随机种子 |
| `--output_dir` | | `./checkpoints` | 模型保存目录 |
| `--db_path` | | `interactive` | SQLite 数据库路径 |
| `--no_db` | | `False` | 禁用数据库记录 |

### 显存不足时的配置

如果遇到 OOM（显存不足），尝试减小批次大小或模型维度：

```bash
python train.py \
  --dataset ESOL \
  --data_dir ./data/ESOL \
  --model_type mamba_ssm \
  --batch_size 8 \
  --d_model 128 \
  --n_layers 2 \
  --device cuda \
  --no_db
```

### GPU 优化配置（RTX 5060 Ti 16GB 最佳实践）

**推荐配置**：充分利用 16GB 显存，提高 GPU 利用率

```bash
python train.py \
  --dataset Lipophilicity \
  --data_dir ./data/Lipophilicity \
  --model_type mamba_ssm \
  --batch_size 256 \
  --num_workers 16 \
  --learning_rate 1e-3 \
  --device cuda \
  --no_db
```

**参数说明**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--batch_size` | 256-512 | 16GB 显存推荐 256，大数据集可用 512 |
| `--num_workers` | 8-16 | DataLoader 并行加载，充分利用 CPU |
| 混合精度 (AMP) | 自动启用 | CUDA 自动混合精度，减少显存占用 |

**显存占用参考**（BiMamba d_model=256, n_layers=4）：

| batch_size | 显存占用 | GPU 利用率 |
|------------|----------|-----------|
| 128 | ~4GB | 脉冲式 100% |
| 256 | ~7GB | 脉冲式 100% |
| 512 | ~12GB | 脉冲式 100% |

**注意**：小数据集（如 Lipophilicity 3360 样本）GPU 利用率呈脉冲式是正常现象，因为模型计算太快（O(N) 复杂度），数据加载跟不上。大数据集（ZINC250K 25万分子）GPU 会持续满载。

### 训练输出

训练完成后：

1. **模型权重**保存在 `./checkpoints/` 目录：
   ```
   checkpoints/
   └── ESOL_bi_mamba_best.pt   # 验证集最优模型
   ```

2. **训练参数**保存在 `checkpoints/args.json`

3. **实验记录**保存在 SQLite 数据库（默认启用）

---

## 模型评估

### 评估已训练模型

**关键参数 `--model_type`**：必须与训练时使用的模型一致。

```bash
# 评估 mamba_ssm 训练的模型
python eval.py \
  --checkpoint ./checkpoints/ESOL_bi_mamba_best.pt \
  --dataset ESOL \
  --data_dir ./data/ESOL \
  --test_file test.csv \
  --model_type mamba_ssm \
  --device cuda
```

### 评估参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | ✅ | - | 模型检查点路径 |
| `--dataset` | ✅ | - | 数据集名称 |
| `--model_type` | | `manual` | **必须设置为 `mamba_ssm`（如果用 mamba_ssm 训练）** |
| `--data_dir` | | `./data` | 数据文件目录 |
| `--test_file` | | `test.csv` | 测试数据文件名 |
| `--task_type` | | `regression` | `regression` 或 `classification` |
| `--d_model` | | `256` | 模型维度（需与训练时一致） |
| `--n_layers` | | `4` | 层数（需与训练时一致） |
| `--pooling` | | `mean` | 池化方式（需与训练时一致） |
| `--batch_size` | | `32` | 批大小 |
| `--device` | | `auto` | `cuda` / `mps` / `cpu` |
| `--max_samples` | | `-1` | 最大评估样本数（-1 表示全部） |

### 评估指标

| 任务类型 | 主要指标 | 其他指标 |
|----------|----------|----------|
| 回归 | RMSE | MAE, MSE |
| 分类 | ROC-AUC | Accuracy |

---

## 常用命令参考

### 1. 完整训练 + 评估流程（ESOL）

```bash
# === 步骤 1: 下载数据 ===
python download_datasets.py --dataset ESOL

# === 步骤 2: 训练模型 ===
python train.py \
  --dataset ESOL \
  --data_dir ./data/ESOL \
  --model_type mamba_ssm \
  --task_type regression \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --device cuda \
  --no_db

# === 步骤 3: 评估模型 ===
python eval.py \
  --checkpoint ./checkpoints/ESOL_bi_mamba_best.pt \
  --dataset ESOL \
  --data_dir ./data/ESOL \
  --model_type mamba_ssm \
  --task_type regression \
  --device cuda
```

### 2. 完整训练 + 评估流程（BBBP 分类）

```bash
# === 步骤 1: 下载数据 ===
python download_datasets.py --dataset BBBP

# === 步骤 2: 训练模型 ===
python train.py \
  --dataset BBBP \
  --data_dir ./data/BBBP \
  --model_type mamba_ssm \
  --task_type classification \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --device cuda \
  --no_db

# === 步骤 3: 评估模型 ===
python eval.py \
  --checkpoint ./checkpoints/BBBP_bi_mamba_best.pt \
  --dataset BBBP \
  --data_dir ./data/BBBP \
  --model_type mamba_ssm \
  --task_type classification \
  --device cuda
```

### 3. 查看训练日志

```bash
# 实时查看训练日志
tail -f training.log

# 查看最后 50 行
tail -50 training.log
```

### 4. 查看实验记录

```bash
python scripts/manage_experiments.py --list
```

### 5. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_model.py -v

# 直接运行测试脚本（无需 pytest）
python tests/test_model.py
```

### 6. Podman 交互式训练（远程机器）

**在远程机器上执行**：

```bash
# 进入容器
podman run --rm -it \
  -v "/home/qfh/graduation-project:/workspace" \
  --workdir /workspace \
  --device nvidia.com/gpu=all \
  localhost/bimamba \
  bash

# 容器内执行训练
python train.py \
  --dataset ESOL \
  --data_dir ./data/ESOL \
  --model_type mamba_ssm \
  --epochs 100 \
  --batch_size 32 \
  --device cuda \
  --no_db
```

---

## 注意事项

1. **必须设置 `--model_type mamba_ssm`**：默认是 `manual`（使用 `bimamba.py`），要使用 mamba_ssm 库的实现必须显式指定。

2. **评估时也要设置 `--model_type`**：训练和评估的 `--model_type` 必须一致。

3. **设备选择**：
   - NVIDIA GPU：使用 `--device cuda`
   - Apple Silicon Mac：使用 `--device mps`
   - CPU 调试：使用 `--device cpu`

4. **数据库**：训练默认启用 SQLite 实验追踪。如需禁用，添加 `--no_db` 参数。

5. **随机种子**：默认 `--seed 42`，确保实验可复现。

6. **模型参数一致性**：训练和评估时的 `d_model`、`n_layers`、`pooling` 等参数必须保持一致，否则加载权重会出错。
