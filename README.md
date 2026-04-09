# Bi-Mamba-Chem

基于**双向 Mamba SSM** 的分子性质预测模型，支持回归、分类和多任务学习。O(N) 线性复杂度（vs Transformer 的 O(N²)），适合长序列分子（如蛋白质、聚合物）。

> 核心模型见 [`mamba.tutorial.md`](./mamba.tutorial.md) —— 从零理解 Mamba SSM 的完整教程。

---

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据集](#数据集)
- [训练](#训练)
- [评估](#评估)
- [实验追踪](#实验追踪)
- [可视化](#可视化)
- [测试](#测试)
- [常见问题](#常见问题)

---

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/yourrepo/bi-mamba-chem.git
cd bi-mamba-chem

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载示例数据
python download_datasets.py

# 4. 开始训练（Mac GPU）
python train.py --dataset ESOL --device mps --epochs 100 --batch_size 16

# 5. 运行测试
python -m pytest tests/ -v
```

**推荐硬件配置**：

| 场景 | 推荐配置 |
|------|---------|
| 快速实验 | Mac M1+/M2+（MPS），batch=16 |
| 正式训练 | NVIDIA GPU（CUDA 11.8+），batch=32 |
| 长序列 | d_model=512, n_layers=6, batch=8 |

---

## 项目结构

```
bi-mamba-chem/
├── src/
│   ├── models/
│   │   └── bimamba.py           # Bi-Mamba 模型（核心实现）
│   ├── data/
│   │   └── molecule_dataset.py  # SMILES 数据集 + MoleculeTokenizer
│   ├── db/                      # SQLite 实验追踪
│   │   ├── database.py
│   │   ├── experiment_repo.py
│   │   └── molecule_repo.py
│   └── visualization/           # 训练曲线、预测散点图、分子图
│       ├── training_plots.py
│       ├── prediction_plots.py
│       ├── molecule_plots.py
│       └── dashboard.py
├── tests/
│   ├── test_model.py            # 模型前向/反向传播测试
│   └── test_data.py             # 数据处理 + tokenization 测试
├── scripts/
│   └── manage_experiments.py    # 实验 CRUD 命令行工具
├── data/                        # MoleculeNet 数据集（下载后生成）
│   ├── ESOL/                    # 水溶解度（回归）
│   ├── BBBP/                    # 血脑屏障渗透（分类）
│   └── ClinTox/                 # 药物毒性（分类）
├── checkpoints/                 # 模型权重保存目录
├── train.py                     # 单任务训练入口
├── eval.py                      # 模型评估入口
├── download_datasets.py         # 下载示例数据集
├── mamba.tutorial.md            # Mamba SSM 完全入门指南
└── README.md
```

---

## 环境配置

### 依赖安装

```bash
# 推荐：用 conda
conda create -n bimamba python=3.10
conda activate bimamba
conda install pytorch torchvision torchaudio -c pytorch
conda install rdkit pandas scikit-learn tqdm matplotlib -c conda-forge

# 或用 pip
pip install -r requirements.txt
```

### 设备选择

| 设备 | 说明 | 命令 |
|------|------|------|
| **MPS** | Apple Silicon GPU（Mac M1+/M2+/M3+） | `--device mps` |
| **CUDA** | NVIDIA GPU | `--device cuda` |
| **CPU** | CPU（调试用） | `--device cpu` |

```bash
# 验证 MPS 可用性（Mac）
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 验证 CUDA 可用性（NVIDIA）
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

> **Mac 用户**：如果 MPS 不可用，确保用 conda 安装 PyTorch（pip 版本可能缺少 MPS 支持）。
>
> ```bash
> conda install pytorch torchvision torchaudio -c pytorch
> ```

---

## 数据集

### 下载示例数据

```bash
python download_datasets.py
```

会生成三个 MoleculeNet 子集：

| 数据集 | 任务 | 分子数 | 评价指标 |
|--------|------|--------|---------|
| ESOL | 回归（水溶解度） | ~1,100 | RMSE, MAE |
| BBBP | 分类（血脑屏障） | ~2,000 | ROC-AUC |
| ClinTox | 分类（药物毒性） | ~1,500 | ROC-AUC |

### 数据格式

CSV 格式，`smiles` 列 + `label` 列：

```csv
smiles,label
CCO,-2.5
CC(=O)OC,-1.8
c1ccccc1,3.2
O=C(C)Oc1ccccc1C=O,0.5
```

### 多任务数据格式

CSV 格式，`smiles` 列 + 多个任务列：

```csv
smiles,solubility,toxicity,logp
CCO,-2.5,0,1.3
CC(=O)OC,-1.8,1,0.5
```

### 数据分割（随机分割）

支持每次实验使用不同的随机 seed 分割数据集，确保实验可复现且数据混合不同：

```python
from src.data.molecule_dataset import random_split_dataset, get_next_split_seed, get_current_split_seed

# 方式1: 自动获取 seed（每次递增）
seed = get_next_split_seed()  # 返回当前 seed 并自动递增
train, val, test = random_split_dataset(
    "dataset/ESOL/delaney.csv",
    output_dir="dataset/ESOL/",
    seed=seed
)

# 方式2: 指定 seed（可复现）
train, val, test = random_split_dataset(
    "dataset/ESOL/delaney.csv",
    seed=42  # 相同 seed 产生相同分割
)

# 查看当前 seed（不递增）
current_seed = get_current_split_seed()
print(f"当前 seed: {current_seed}")
```

**分割参数**：
```python
random_split_dataset(
    input_csv="dataset/ESOL/delaney.csv",
    output_dir="dataset/ESOL/",  # 可选：保存到文件
    train_ratio=0.8,              # 默认 0.8
    val_ratio=0.1,              # 默认 0.1
    test_ratio=0.1,             # 默认 0.1
    seed=42,                    # 随机种子
    n_jobs=None                 # None=使用全部 CPU 核心
)
```

**seed 管理机制**：
- seed 默认从 42 开始
- `get_next_split_seed()` 获取当前 seed 并自动递增
- seed 持久化在 `.split_seed` 文件中
- 不同 seed 产生不同分割结果，方便做数据增强实验

**多线程加速**：
```python
# 自动使用全部 CPU 核心（推荐）
train, val, test = random_split_dataset("dataset/ZINC250K/...", n_jobs=None)

# 指定线程数
train, val, test = random_split_dataset("dataset/ZINC250K/...", n_jobs=8)
```

---

## 训练

### 单任务训练

```bash
# ESOL 回归
python train.py \
    --dataset ESOL \
    --data_dir ./data/ESOL \
    --epochs 100 \
    --batch_size 16 \
    --device mps \
    --learning_rate 1e-3

# BBBP 分类
python train.py \
    --dataset BBBP \
    --data_dir ./data/BBBP \
    --task_type classification \
    --epochs 100 \
    --batch_size 16 \
    --device mps
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必需 | 数据集名称 |
| `--data_dir` | `./data` | 数据目录 |
| `--task_type` | `regression` | `regression` 或 `classification` |
| `--d_model` | 256 | 模型维度 |
| `--n_layers` | 4 | Bi-Mamba 层数 |
| `--pooling` | `mean` | `mean` / `max` / `cls` |
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--dropout` | 0.1 | Dropout 率 |
| `--device` | `auto` | `cuda` / `mps` / `cpu` |
| `--max_length` | 512 | 最大 SMILES 长度 |
| `--db_path` | `interactive` | SQLite 数据库路径（`interactive` / `none`） |

### 训练输出

训练过程中会自动：

- 打印每个 epoch 的 loss、验证指标
- 保存验证集上最优的模型权重到 `checkpoints/`
- 记录实验到 SQLite 数据库（默认启用）

```
checkpoints/
└── ESOL_bi_mamba_epoch10_valLoss0.4521.pt
```

### 超参数参考

| 场景 | d_model | n_layers | batch_size | lr | epochs |
|------|---------|---------|-----------|-----|--------|
| 默认（Mac） | 256 | 4 | 16 | 1e-3 | 100 |
| 长序列分子 | 512 | 6 | 8 | 5e-5 | 100 |
| 小数据集 | 128 | 2 | 8 | 1e-4 | 200 |

### RTX 4060 8GB 推荐配置

**推荐数据集: ESOL**（数据量小 ~1,128 样本，训练快，显存占用低）

```bash
python train.py --dataset ESOL --device cuda --batch_size 32 --epochs 100 --learning_rate 1e-3
```

| 设置 | 推荐值 | 说明 |
|------|--------|------|
| batch_size | 32-64 | 8GB 显存足够 |
| 学习率 | 1e-3 ~ 5e-4 | 初始学习率 |
| 梯度裁剪 | `clip_grad_norm_(model.parameters(), 1.0)` | 防止梯度爆炸 |
| 混合精度 | 可选 (`torch.cuda.amp`) | 进一步节省显存 |

**如遇 OOM（显存不足）**：
- 降低 `batch_size` 到 16 或 8
- 使用 `torch.inference_mode()` 验证
- 减小模型 `d_model` 到 128 或 `n_layers` 到 2

### 分布式训练（多 GPU）

```bash
# 单机多卡（PyTorch DDP）
torchrun --nproc_per_node=4 train.py \
    --dataset ESOL \
    --epochs 100 \
    --batch_size 32 \
    --device cuda
```

---

## 评估

### 评估已训练模型

```bash
python eval.py \
    --checkpoint checkpoints/ESOL_bi_mamba_best.pt \
    --dataset ESOL \
    --data_dir ./data/ESOL \
    --test_file test.csv \
    --device mps
```

### 评估指标

| 任务 | 主要指标 | 其他指标 |
|------|---------|---------|
| 回归 | RMSE | MAE, MSE |
| 分类 | ROC-AUC | Accuracy, F1 |

---

## 实验追踪

项目使用 SQLite 自动记录每次训练实验，方便对比和复现。

### 数据库位置

```
src/db/
└── experiments.db
```

### 交互式选择数据库

```bash
# 交互式选择（默认）— 列出所有 .db 文件
python train.py --dataset ESOL

# 指定数据库
python train.py --dataset ESOL --db_path ./src/db/my_exp.db

# 禁用数据库
python train.py --dataset ESOL --db_path none
```

### 管理实验

```bash
# 列出所有实验
python scripts/manage_experiments.py --list

# 按状态筛选
python scripts/manage_experiments.py --list --status completed

# 查看实验详情
python scripts/manage_experiments.py -d 1

# 对比多个实验
python scripts/manage_experiments.py -c 1 2 3

# 删除实验
python scripts/manage_experiments.py --delete 5
```

### 训练时自动记录的内容

- 模型配置（d_model, n_layers, pooling）
- 超参数（lr, batch_size, dropout）
- 每个 epoch 的训练/验证 loss 和指标
- 最终测试集结果
- 训练时长、硬件信息

---

## 可视化

### 训练曲线

```python
from src.visualization.training_plots import plot_experiment_training

# 从数据库加载并绘图
fig = plot_experiment_training(exp_id=1, save_path="training.png")
```

### 预测散点图

```python
from src.visualization.prediction_plots import plot_prediction_scatter

plot_prediction_scatter(
    y_true=y_true,
    y_pred=y_pred,
    task_name="ESOL",
    save_path="scatter.png"
)
```

### 分子结构图

```python
from src.visualization.molecule_plots import draw_molecule, plot_molecule_grid

# 单个分子
draw_molecule("CCO", legend="Ethanol", save_path="ethanol.png")

# 分子网格
smiles_list = ["CCO", "CC(=O)OC", "c1ccccc1"]
plot_molecule_grid(smiles_list, mols_per_row=3, save_path="molecules.png")
```

### 实验仪表盘

```python
from src.visualization.dashboard import create_dashboard_from_db

# 综合对比面板
create_dashboard_from_db(exp_ids=[1, 2, 3], save_path="dashboard.png")
```

---

## 测试

### 运行测试套件

```bash
# 所有测试
python -m pytest tests/ -v

# 单个测试文件
python -m pytest tests/test_model.py -v

# 单个测试函数
python -m pytest tests/test_data.py::test_tokenization -v
```

### 测试覆盖报告

```bash
pip install pytest pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### 快速验证

```bash
# 模型前向传播
python tests/test_model.py

# 数据处理
python tests/test_data.py
```

---

## 常见问题

### NaN 损失

```bash
# 降低学习率
python train.py --learning_rate 1e-4

# 启用梯度裁剪
python train.py --max_grad_norm 1.0

# 减小 batch_size
python train.py --batch_size 8
```

### GPU 显存不足

```bash
# 减小 batch_size
python train.py --batch_size 4

# 减小模型维度
python train.py --d_model 128 --n_layers 2

# 梯度累积
python train.py --batch_size 4 --gradient_accumulation_steps 4
```

### MPS 不可用（Mac）

```bash
# 确认 PyTorch 版本支持 MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# 如果不行，用 conda 重装
conda install pytorch torchvision torchaudio -c pytorch

# 或者用 CPU 调试
python train.py --device cpu --batch_size 4
```

### RDKit 安装失败

```bash
# 用 conda（推荐）
conda install -c conda-forge rdkit -y

# 或 pip
pip install rdkit-pypi
```

---

## 参考

- **Mamba 论文**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- **mamba_ssm 库**: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- **HiPPO 初始化**: [HiPPO (NeurIPS 2020)](https://papers.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)
- **Mamba 教程**: [`mamba.tutorial.md`](./mamba.tutorial.md) — 从零理解 Mamba SSM

---

## 许可证

MIT License
