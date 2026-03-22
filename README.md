# Bi-Mamba-Chem: 基于双向状态空间模型的分子性质预测

基于双向 Mamba 架构的分子性质预测模型，支持回归、分类和多任务学习。

## 快速导航

| 场景 | 命令 |
|------|------|
| [快速开始训练](#快速开始) | `python train.py --dataset ESOL --device mps` |
| [运行测试](#代码测试) | `python tests/test_model.py` |
| [NVIDIA GPU + Podman](#方式二使用-podman-容器测试推荐用于-nvidia-gpu) | `podman run --gpus all ...` |
| [Apple Silicon / Conda](#方式三使用-conda-本地测试apple-silicon-或无-gpu) | `conda install pytorch -c pytorch` |
| [评估模型](#模型评估) | `python eval.py --checkpoint checkpoints/...` |

## 新增功能 

- **数据库管理** - SQLite 存储实验记录和分子数据
- **实验追踪** - 自动记录训练过程、对比实验结果
- **多任务学习** - 同时预测多个分子属性
- **可视化模块** - 训练曲线、预测散点图、分子结构可视化

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
│   │   ├── bimamba.py          # Bi-Mamba 模型实现
│   ├── data/
│   │   ├── molecule_dataset.py    # 数据处理工具
│   ├── db/                    # 数据库管理模块
│   │   ├── database.py         # SQLite 连接
│   │   ├── molecule_repo.py    # 分子数据 CRUD
│   │   └── experiment_repo.py # 实验记录 CRUD
│   └── visualization/          # 可视化模块
│       ├── training_plots.py   # 训练曲线
│       ├── prediction_plots.py # 预测散点图
│       ├── molecule_plots.py   # 分子结构可视化
│       └── dashboard.py        # 实验仪表盘
├── tests/
│   ├── test_model.py       # 模型测试
│   └── test_data.py        # 数据测试
├── scripts/
│   └── manage_experiments.py # 实验管理工具
├── data/
│   ├── ESOL/              # ESOL 数据集（回归任务）
│   ├── BBBP/              # BBBP 数据集（分类任务）
│   └── ClinTox/           # ClinTox 数据集（分类任务）
├── checkpoints/           # 模型保存目录
├── train.py               # 单任务训练脚本
├── train_multitask.py     # 多任务训练脚本
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

---

## 数据库与实验追踪

项目使用 SQLite 数据库自动记录训练过程和实验结果。

### 数据库文件位置

数据库文件位于：`src/data/database/` 目录

```
src/data/database/
├── experiment1.db
├── experiment2.db
└── ...
```

### 交互式选择数据库

训练时可以选择数据库：

```bash
# 交互式选择（默认）- 会列出 src/data/database/ 下所有 .db 文件
python train.py --dataset ESOL

# 输出示例：
# 可用数据库：
#   [1] experiment1.db
#   [2] experiment2.db
# 请选择数据库编号: 1
```

### 相关参数

| 参数 | 说明 |
|------|------|
| `--db_path interactive` | 交互式选择数据库（默认） |
| `--db_path my_db.db` | 指定数据库文件 |
| `--no_db` | 禁用数据库记录 |

### 实验管理命令

```bash
# 列出所有实验
python scripts/manage_experiments.py --list

# 列出特定状态的实验
python scripts/manage_experiments.py --list --status completed

# 查看实验详情
python scripts/manage_experiments.py -d 1

# 对比多个实验
python scripts/manage_experiments.py -c 1 2 3

# 删除实验
python scripts/manage_experiments.py --delete 5
```

### 训练时自动记录

训练时会自动记录：
- 模型配置（维度、层数、池化方法）
- 超参数（学习率、批大小等）
- 每个 epoch 的训练/验证指标
- 最终测试结果

---

## 多任务学习

支持同时预测多个分子属性（回归或分类）。

### 任务配置格式

```
task_name:type:weight
```

- `task_name`: 任务名称
- `type`: `regression` 或 `classification`
- `weight`: 损失权重（默认 1.0）

### 多任务数据格式

CSV 格式，第一列为 SMILES，后续列为各任务标签：

```csv
smiles,solubility,toxicity,logp
CCO,-2.5,0,1.3
CC(=O)OC,-1.8,1,0.5
```

### 多任务训练示例

```bash
python train_multitask.py \
    --dataset multitask \
    --data_dir ./data/multitask \
    --tasks "solubility:regression:1.0,toxicity:classification:0.5,logp:regression:0.8" \
    --epochs 100 \
    --batch_size 16 \
    --device mps
```

### 多任务参数

| 参数 | 说明 |
|------|------|
| `--tasks` | 任务配置字符串（必需） |
| `--task_strategy` | `shared`（共享头部）或 `separate`（独立头部） |
| `--d_model` | 模型维度 |
| `--n_layers` | 层数 |

### 任务策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `shared` | 共享多任务头部，学习任务关联 | 任务相关性强 |
| `separate` | 每个任务独立预测头 | 任务相对独立 |

---

## 可视化模块

提供丰富的训练过程和预测结果可视化功能。

### 训练曲线

```python
from src.visualization import plot_training_curves, plot_metric_comparison

# 从数据库加载并绘制训练曲线
from src.visualization.training_plots import plot_experiment_training
fig = plot_experiment_training(exp_id=1, save_path="training.png")

# 或手动绘制
logs = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4}, ...]
plot_training_curves(logs, save_path="curves.png")

# 对比多个实验的指标
results = {"exp1": {"mae": 0.1}, "exp2": {"mae": 0.15}}
plot_metric_comparison(results, metric="mae")
```

### 预测结果可视化

```python
from src.visualization import plot_prediction_scatter, plot_residuals

# 预测散点图
plot_prediction_scatter(
    y_true=y_true,
    y_pred=y_pred,
    task_name="ESOL",
    save_path="scatter.png"
)

# 残差分析
plot_residuals(y_true, y_pred, save_path="residuals.png")
```

### 分子结构可视化

```python
from src.visualization import draw_molecule, plot_molecule_grid

# 绘制单个分子
draw_molecule("CCO", legend="Ethanol", save_path="ethanol.png")

# 绘制分子网格
smiles_list = ["CCO", "CC(=O)OC", "c1ccccc1"]
plot_molecule_grid(smiles_list, mols_per_row=3, save_path="molecules.png")
```

### 实验仪表盘

```python
from src.visualization import create_experiment_dashboard
from src.visualization.dashboard import create_dashboard_from_db

# 从数据库创建仪表盘
create_dashboard_from_db(exp_ids=[1, 2, 3], save_path="dashboard.png")

# 或手动创建
experiments = [
    {"name": "exp1", "metrics": {...}, "training_logs": [...]},
    {"name": "exp2", "metrics": {...}, "training_logs": [...]},
]
create_experiment_dashboard(experiments, save_path="dashboard.png")
```

### 可视化功能汇总

| 类型 | 功能 |
|------|------|
| 训练曲线 | 损失、MAE、RMSE 随 epoch 变化 |
| 指标对比 | 多个实验的指标柱状图 |
| 预测散点图 | 真实值 vs 预测值 |
| 残差分析 | 残差分布和 QQ 图 |
| 分子结构 | RDKit 渲染分子图 |
| 实验仪表盘 | 综合对比面板 |

---

## 训练参数说明

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

### Q: Podman 无法访问 GPU？

```bash
# 1. 确认安装了 nvidia-container-toolkit
# Linux: sudo apt-get install nvidia-container-toolkit
# 然后: sudo systemctl restart podman

# 2. 确认 nvidia-smi 可以正常显示
nvidia-smi

# 3. 如果仍然不行，使用 conda 本地测试
conda install pytorch -c pytorch
```

### Q: Apple Silicon (M1/M2/M3) 运行出错？

```bash
# 1. 确保使用 conda 安装支持 MPS 的 PyTorch
conda install pytorch torchvision torchaudio -c pytorch -y

# 2. 验证 MPS 是否可用
python -c "import torch; print(torch.backends.mps.is_available())"

# 3. 如果 MPS 不可用，强制使用 CPU
python train.py --device cpu --batch_size 4
```

### Q: RDKit 安装失败？

```bash
# 使用 conda 安装（推荐）
conda install -c conda-forge rdkit -y

# 或使用 pip
pip install rdkit-pypi
```

## 代码测试

### 方式一：直接运行测试脚本

```bash
# 测试模型
python tests/test_model.py

# 测试数据处理
python tests/test_data.py

# 运行所有测试
python -m pytest tests/ -v
```

### 方式二：使用 Podman 容器测试（推荐用于 NVIDIA GPU）

**1. 检查是否有 NVIDIA 显卡**

```bash
# Linux/macOS
nvidia-smi
```

如果显示类似以下内容，说明有 NVIDIA 显卡：
```
+------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03  CUDA Version: 12.2 |
+------------------------------------------------------------------+
```

**NVIDIA 显卡要求：**
| 要求 | 最低版本 |
|------|----------|
| 驱动版本 | 525.60.13+ |
| CUDA 工具包 | 11.6+ (推荐 12.x) |
| 显存 | 2GB (小模型) / 8GB (大模型) |

**2. 安装 Podman**

```bash
# macOS
brew install podman

# Ubuntu/Debian
sudo apt-get install podman

# Fedora
sudo dnf install podman
```

**3. 运行容器测试**

有 NVIDIA 显卡时：
```bash
podman run --rm \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
    bash -c "pip install -r requirements.txt && python -m pytest tests/ -v"
```

### 方式三：使用 Conda 本地测试（Apple Silicon 或无 GPU）

如果你的电脑是 Apple Silicon (M1/M2/M3) 或没有 NVIDIA 显卡，使用 conda：

**1. 创建 conda 环境**

```bash
# 创建环境
conda create -n bimamba python=3.10 -y
conda activate bimamba

# 安装 PyTorch (支持 MPS GPU 加速)
conda install pytorch torchvision torchaudio -c pytorch -y

# 安装其他依赖
conda install -c conda-forge numpy pandas scikit-learn tqdm matplotlib -y
conda install -c conda-forge rdkit -y
```

**2. 验证 MPS 可用性（Apple Silicon）**

```bash
python -c "import torch; print(f'MPS 可用: {torch.backends.mps.is_available()}')"
```

**3. 运行测试**

```bash
# 测试模型
python tests/test_model.py

# 测试数据处理
python tests/test_data.py
```

### 运行单个测试函数

```bash
# 使用 pytest（推荐）
python -m pytest tests/test_data.py::test_tokenization -v

# 或直接导入运行
python -c "from tests.test_data import test_tokenization; test_tokenization()"
```

### 测试覆盖报告

```bash
pip install pytest pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
# 报告生成在 htmlcov/index.html
```

---

## 许可证

MIT License
