# Bi-Mamba-Chem

基于**双向 Mamba SSM** 的分子性质预测模型，支持回归、分类任务。O(N) 线性复杂度（vs Transformer 的 O(N²)），适合长序列分子。

> 核心模型见 [`mamba.tutorial.md`](./mamba.tutorial.md) —— Mamba SSM 完全入门指南。

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
git clone <your-repo-url>
cd bi-mamba-chem

# 2. 安装依赖
pip install -r requirements.txt
# 还需要安装 PyTorch（支持 MPS/CUDA）
pip install torch torchvision

# 3. 开始训练（Mac GPU）
python train.py --dataset ESOL --data_dir ./dataset/ESOL --epochs 100 --batch_size 16 --device mps

# 4. 运行测试
python -m pytest tests/ -v
```

**推荐硬件配置**：

| 场景 | 推荐配置 |
|------|---------|
| 快速实验 | Mac M1+/M2+（MPS），batch=16 |
| 正式训练 | NVIDIA GPU（CUDA），batch=32 |
| 长序列 | d_model=512, n_layers=6, batch=8 |

---

## 项目结构

```
bi-mamba-chem/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bimamba.py                    # 手动实现 SSM（无外部依赖）
│   │   ├── bimamba_with_mamba_ssm.py     # mamba_ssm 包封装版本
│   │   └── bimamba_with_mamba_ssm_architecture.md
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py                  # SMILES 分词器
│   │   ├── dataset.py                    # Dataset 基类
│   │   ├── molecule_dataset.py           # MoleculeDataset + MoleculeTokenizer
│   │   ├── dataloader.py                # create_data_loaders + LabelNormalizer
│   │   ├── split.py                      # scaffold_split / random_split
│   │   ├── column_mapping.py             # CSV 列名检测
│   │   └── molecule_dataset_architecture.md
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py                  # SQLite 单例 + Dataclass 定义
│   │   ├── experiment_repo.py           # ExperimentRepository CRUD
│   │   └── molecule_repo.py              # MoleculeRepository CRUD
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── dashboard.py                  # 多实验对比仪表盘
│   │   ├── training_plots.py            # 训练曲线
│   │   ├── prediction_plots.py           # 预测散点图
│   │   └── molecule_plots.py             # RDKit 分子图
│   └── shared/
│       ├── __init__.py
│       └── utils.py                      # 训练/评估参数解析、通用工具
├── tests/
│   ├── __init__.py
│   ├── test_model.py                    # 模型前向/反向传播测试
│   ├── test_data.py                     # 数据处理 + tokenization 测试
│   └── test_column_detection.py         # CSV 列名检测测试
├── scripts/
│   ├── manage_experiments.py            # 实验 CRUD 命令行工具
│   ├── batch_train_phase1.py            # 批量训练脚本（RTX 5060 Ti 优化）
│   └── benchmarks/
│       ├── benchmark_efficiency.py      # O(N) 效率基准测试
│       ├── benchmark_transformer.py      # Transformer 对比基准测试
│       ├── train_esol_pooling.py        # 不同 pooling 对比
│       ├── split_esol.py                # ESOL 数据集划分
│       └── split_all.py                 # 全数据集划分
├── dataset/                             # MoleculeNet 数据集
│   ├── ESOL/                           # 水溶解度（回归）
│   ├── BBBP/                           # 血脑屏障渗透（分类）
│   ├── ClinTox/                        # 药物毒性（分类）
│   ├── FreeSolv/                      # 水合自由能（回归）
│   ├── Lipophilicity/                 # 脂溶性（回归）
│   ├── bace/                          # BACE 抑制剂（分类）
│   ├── HIV/                           # HIV 感染性（分类）
│   ├── SIDER/                         # 药物副作用（分类）
│   ├── MUV/                           # MUV 数据集
│   ├── ZINC250K/                      # ZINC250K（回归）
│   ├── EGFR/                          # EGFR 抑制剂
│   ├── mpro/                          # Mpro 抑制剂
│   └── BDB2020+/                      # BindingDB 2020+
├── checkpoints/                        # 模型权重保存目录
├── logs/                               # 训练日志目录
├── train.py                            # 单任务训练入口
├── eval.py                             # 模型评估入口
├── mamba.tutorial.md                   # Mamba SSM 完全入门指南
├── requirements.txt                    # Python 依赖
└── README.md
```

---

## 环境配置

### 依赖安装

```bash
# 基础依赖
pip install -r requirements.txt

# PyTorch（根据你的硬件选择）
# Apple Silicon Mac
pip install torch torchvision

# NVIDIA GPU (CUDA 11.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# RDKit（强烈推荐用 conda 安装）
conda install -c conda-forge rdkit -y
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

---

## 数据集

### 已有数据集

项目在 `dataset/` 目录下包含以下 MoleculeNet 数据集：

| 数据集 | 任务 | 分子数 | 类型 |
|--------|------|--------|------|
| ESOL | 回归（水溶解度） | ~1,100 | 回归 |
| BBBP | 分类（血脑屏障） | ~2,000 | 分类 |
| ClinTox | 分类（药物毒性） | ~1,500 | 分类 |
| FreeSolv | 回归（水合自由能） | ~640 | 回归 |
| Lipophilicity | 回归（脂溶性） | ~4,200 | 回归 |
| bace | 分类（BACE 抑制剂） | ~1,500 | 分类 |
| HIV | 分类（HIV 感染性） | ~40,000 | 分类 |
| SIDER | 分类（药物副作用） | ~1,400 | 分类 |
| MUV | 分类（MAV 验证） | ~35,000 | 分类 |
| ZINC250K | 回归（溶解度） | ~250,000 | 回归 |
| EGFR | 回归（EGFR 抑制剂） | ~5,000 | 回归 |
| mpro | 回归（Mpro 抑制剂） | ~67,000 | 回归 |

### 数据格式

CSV 格式，`smiles` 列 + `label` 列：

```csv
smiles,label
CCO,-2.5
CC(=O)OC,-1.8
c1ccccc1,3.2
O=C(C)Oc1ccccc1C=O,0.5
```

### 数据划分

支持两种划分策略：**随机划分** 和 **Scaffold 划分**：

```bash
# 随机划分（默认）
python train.py --dataset ESOL --split random --split_seed 42

# Scaffold 划分（基于分子骨架）
python train.py --dataset ESOL --split scaffold --split_seed 42
```

Scaffold 划分能更好地评估模型对未知分子骨架的泛化能力。

---

## 训练

### 单任务训练

```bash
# ESOL 回归（manual SSM，无外部依赖）
python train.py \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --epochs 100 \
    --batch_size 16 \
    --device mps \
    --model_type manual

# BBBP 分类（使用 mamba_ssm 包）
python train.py \
    --dataset BBBP \
    --data_dir ./dataset/BBBP \
    --task_type classification \
    --epochs 100 \
    --batch_size 16 \
    --device mps \
    --model_type mamba_ssm
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必需 | 数据集名称 |
| `--data_dir` | `./dataset` | 数据目录 |
| `--model_type` | `manual` | `manual`（无依赖）或 `mamba_ssm`（需安装 mamba-ssm） |
| `--task_type` | `regression` | `regression` 或 `classification` |
| `--split` | `random` | `random` 或 `scaffold` |
| `--d_model` | 256 | 模型维度 |
| `--d_mamba` | 256 | Mamba 内部维度 |
| `--n_layers` | 4 | Bi-Mamba 层数 |
| `--pooling` | `mean` | `mean` / `max` / `cls` |
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--dropout` | 0.1 | Dropout 率 |
| `--device` | `auto` | `cuda` / `mps` / `cpu` |
| `--max_length` | 512 | 最大 SMILES 长度 |
| `--db_path` | `interactive` | SQLite 数据库路径 |
| `--single_file` | None | 单数据文件路径（自动划分 train/val/test） |

### 模型类型

| 类型 | 说明 | 依赖 |
|------|------|------|
| `manual` | 纯 PyTorch 手动实现 SSM，无外部依赖 | 无 |
| `mamba_ssm` | 使用 `mamba-ssm` 包，高效 GPU 计算 | `pip install mamba-ssm` |

### 训练输出

- 模型权重保存到 `checkpoints/{dataset}_bi_mamba_best.pt`
- 训练日志保存到 `logs/{dataset}_{model_type}_{timestamp}.json`

---

## 评估

### 评估已训练模型

```bash
python eval.py \
    --checkpoint checkpoints/ESOL_bi_mamba_best.pt \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --test_file test.csv \
    --device mps
```

### 评估指标

| 任务 | 主要指标 | 其他指标 |
|------|---------|---------|
| 回归 | RMSE | MAE, MSE, RMSE_orig |
| 分类 | ROC-AUC | Accuracy |

---

## 实验追踪

项目使用 SQLite 自动记录每次训练实验。

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
from src.visualization import plot_training_curves

# 从日志文件加载并绘图
plot_training_curves(
    logs="logs/ESOL_manual_2026-04-14_21-44-19.json",
    save_path="training.png"
)
```

### 预测散点图

```python
from src.visualization import plot_prediction_scatter

plot_prediction_scatter(
    y_true=y_true,
    y_pred=y_pred,
    task_name="ESOL",
    save_path="scatter.png"
)
```

### 分子结构图

```python
from src.visualization import draw_molecule, plot_molecule_grid

# 单个分子
draw_molecule("CCO", legend="Ethanol", save_path="ethanol.png")

# 分子网格
smiles_list = ["CCO", "CC(=O)OC", "c1ccccc1"]
plot_molecule_grid(smiles_list, mols_per_row=3, save_path="molecules.png")
```

### 实验仪表盘

```python
from src.visualization import create_experiment_dashboard

# 综合对比面板
create_experiment_dashboard(
    experiments=[exp1, exp2, exp3],
    save_path="dashboard.png"
)
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

# 直接运行（无需 pytest）
python tests/test_model.py
```

### 测试覆盖报告

```bash
pip install pytest pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
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
```

### MPS 不可用（Mac）

```bash
# 用 conda 安装 PyTorch（推荐）
conda install pytorch torchvision torchaudio -c pytorch

# 或用 CPU 调试
python train.py --device cpu --batch_size 4
```

### RDKit 安装失败

```bash
# 用 conda（推荐）
conda install -c conda-forge rdkit -y
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
