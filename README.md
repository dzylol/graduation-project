# Bi-Mamba-Chem

Bidirectional Mamba SSM for molecular property prediction (regression + classification).

Implemented a Bi-Mamba model from scratch in PyTorch — no `mamba-ssm` dependency required. Evaluated on MoleculeNet benchmarks (ESOL, FreeSolv, Lipophilicity, HIV, BBBP, etc.).

Mamba's O(N) complexity is designed for long sequences (thousands of tokens). SMILES strings are short (50–150 tokens), so the efficiency gain over Transformer is limited. The model performs fine on small molecules but really benefits from longer inputs like protein sequences.

Model walkthrough: [`mamba.tutorial.md`](./mamba.tutorial.md)

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
# 1. Clone
git clone <your-repo-url>
cd bi-mamba-chem

# 2. Dependencies
pip install -r requirements.txt
pip install torch torchvision

# 3. Train (Mac GPU)
python train.py --dataset ESOL --data_dir ./dataset/ESOL --epochs 100 --batch_size 16 --device mps

# 4. Run tests
python -m pytest tests/ -v
```

**硬件建议**：

| 场景 | 配置 |
|------|------|
| 快速实验 | Mac M1+/M2+（MPS），batch=16 |
| 正式训练 | NVIDIA GPU（CUDA），batch=32 |
| 长序列 | d_model=512, n_layers=6, batch=8 |

> SMILES 分子序列一般在 50–150 tokens。Mamba 的 O(N) 优势要到 1000+ tokens 才明显。长序列场景（蛋白质、聚合物）更合适。

---

## 项目结构

```
bi-mamba-chem/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bimamba.py                    # 纯 PyTorch SSM 实现
│   │   ├── bimamba_with_mamba_ssm.py     # mamba_ssm 封装版本
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
│   │   ├── database.py                  # SQLite 单例
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
│       └── utils.py                      # 训练/评估参数解析
├── tests/
│   ├── __init__.py
│   ├── test_model.py                    # 前向/反向传播测试
│   ├── test_data.py                     # 数据处理测试
│   └── test_column_detection.py         # CSV 列名检测测试
├── scripts/
│   ├── manage_experiments.py            # 实验 CRUD 命令行工具
│   ├── batch_train_phase1.py            # 批量训练（RTX 5060 Ti）
│   └── benchmarks/
│       ├── benchmark_efficiency.py
│       ├── benchmark_transformer.py
│       ├── train_esol_pooling.py
│       ├── split_esol.py
│       └── split_all.py
├── dataset/                             # MoleculeNet
│   ├── ESOL/                           # 水溶解度（回归）
│   ├── BBBP/                           # 血脑屏障渗透（分类）
│   ├── ClinTox/                        # 药物毒性（分类）
│   ├── FreeSolv/                      # 水合自由能（回归）
│   ├── Lipophilicity/                 # 脂溶性（回归）
│   ├── bace/                          # BACE 抑制剂（分类）
│   ├── HIV/                           # HIV 感染性（分类）
│   ├── SIDER/                         # 药物副作用（分类）
│   ├── MUV/                           # MUV（分类）
│   ├── ZINC250K/                      # ZINC250K（回归）
│   ├── EGFR/                          # EGFR 抑制剂
│   ├── mpro/                          # Mpro 抑制剂
│   └── BDB2020+/                      # BindingDB 2020+
├── checkpoints/
├── logs/
├── train.py
├── eval.py
├── mamba.tutorial.md
├── requirements.txt
└── README.md
```

---

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt

# PyTorch（Apple Silicon）
pip install torch torchvision

# PyTorch（NVIDIA GPU, CUDA 11.8+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# RDKit
conda install -c conda-forge rdkit -y
```

### 设备选择

| 设备 | 说明 | 命令 |
|------|------|------|
| MPS | Apple Silicon GPU | `--device mps` |
| CUDA | NVIDIA GPU | `--device cuda` |
| CPU | 调试用 | `--device cpu` |

```bash
# 验证 MPS（Mac）
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 验证 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 数据集

| 数据集 | 任务 | 分子数 | 类型 |
|--------|------|--------|------|
| ESOL | 水溶解度 | ~1,100 | 回归 |
| BBBP | 血脑屏障 | ~2,000 | 分类 |
| ClinTox | 药物毒性 | ~1,500 | 分类 |
| FreeSolv | 水合自由能 | ~640 | 回归 |
| Lipophilicity | 脂溶性 | ~4,200 | 回归 |
| bace | BACE 抑制剂 | ~1,500 | 分类 |
| HIV | HIV 感染性 | ~40,000 | 分类 |
| SIDER | 药物副作用 | ~1,400 | 分类 |
| MUV | MUV | ~35,000 | 分类 |
| ZINC250K | 溶解度 | ~250,000 | 回归 |
| EGFR | EGFR 抑制剂 | ~5,000 | 回归 |
| mpro | Mpro 抑制剂 | ~67,000 | 回归 |

### 数据格式

```csv
smiles,label
CCO,-2.5
CC(=O)OC,-1.8
c1ccccc1,3.2
```

### 数据划分

```bash
# 随机划分
python train.py --dataset ESOL --split random --split_seed 42

# Scaffold 划分
python train.py --dataset ESOL --split scaffold --split_seed 42
```

---

## 训练

```bash
# ESOL 回归（纯 PyTorch SSM）
python train.py \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --epochs 100 \
    --batch_size 16 \
    --device mps \
    --model_type manual

# BBBP 分类（mamba_ssm）
python train.py \
    --dataset BBBP \
    --data_dir ./dataset/BBBP \
    --task_type classification \
    --epochs 100 \
    --batch_size 16 \
    --device mps \
    --model_type mamba_ssm
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset` | 必需 | 数据集名称 |
| `--data_dir` | ./dataset | 数据目录 |
| `--model_type` | manual | manual / mamba_ssm |
| `--task_type` | regression | regression / classification |
| `--split` | random | random / scaffold |
| `--d_model` | 256 | 模型维度 |
| `--n_layers` | 4 | Bi-Mamba 层数 |
| `--pooling` | mean | mean / max / cls |
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--dropout` | 0.1 | Dropout 率 |
| `--device` | auto | cuda / mps / cpu |
| `--max_length` | 512 | 最大 SMILES 长度 |

### 模型类型

| 类型 | 说明 | 依赖 |
|------|------|------|
| manual | 纯 PyTorch SSM | 无 |
| mamba_ssm | mamba-ssm 包 | pip install mamba-ssm |

---

## 评估

```bash
python eval.py \
    --checkpoint checkpoints/ESOL_bi_mamba_best.pt \
    --dataset ESOL \
    --data_dir ./dataset/ESOL \
    --test_file test.csv \
    --device mps
```

| 任务 | 指标 |
|------|------|
| 回归 | RMSE, MAE, MSE |
| 分类 | ROC-AUC, Accuracy |

---

## 实验追踪

SQLite 自动记录训练实验。

```bash
python scripts/manage_experiments.py --list
python scripts/manage_experiments.py --list --status completed
python scripts/manage_experiments.py -d 1
python scripts/manage_experiments.py -c 1 2 3
python scripts/manage_experiments.py --delete 5
```

每次训练记录：模型配置、超参数、每 epoch loss/指标、测试结果、训练时长。

---

## 可视化

```python
from src.visualization import plot_training_curves, plot_prediction_scatter
from src.visualization import draw_molecule, plot_molecule_grid, create_experiment_dashboard

plot_training_curves(logs="logs/ESOL_manual_2026-04-14_21-44-19.json", save_path="training.png")
plot_prediction_scatter(y_true=y_true, y_pred=y_pred, task_name="ESOL", save_path="scatter.png")
draw_molecule("CCO", legend="Ethanol", save_path="ethanol.png")
create_experiment_dashboard(experiments=[exp1, exp2, exp3], save_path="dashboard.png")
```

---

## 测试

```bash
python -m pytest tests/ -v
python -m pytest tests/test_model.py -v
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 常见问题

**NaN 损失** — 降低 lr (`--learning_rate 1e-4`)，加梯度裁剪 (`--max_grad_norm 1.0`)，减 batch (`--batch_size 8`)

**GPU 显存不足** — 减 batch (`--batch_size 4`)，降模型维度 (`--d_model 128 --n_layers 2`)

**MPS 不可用（Mac）** — `conda install pytorch torchvision torchaudio -c pytorch`

**RDKit 安装失败** — `conda install -c conda-forge rdkit -y`

**SMILES 上效率提升不明显？** — 正常的。Mamba 设计目标是长序列（语言模型上下文数千 tokens，基因组数万 bp）。SMILES 只有 50–150 tokens，这个长度下 Transformer 已经足够快。

---

## 参考

- Mamba: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- mamba_ssm: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- HiPPO: [NeurIPS 2020](https://papers.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)

---

MIT License
