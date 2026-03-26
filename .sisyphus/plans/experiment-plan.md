# Bi-Mamba-Chem 系统性实验计划

**项目**: Bi-Mamba-Chem (双向 Mamba SSM 分子性质预测)
**制定日期**: 2026-03-26
**用户需求**: 5次重复 × 全部超参数 × 中等规模 (3-5天)
**GPU约束**: 16GB VRAM (服务器 GPU)

---

## 一、环境准备

### 1.1 环境状态

| 组件 | 状态 | 详情 |
|------|------|------|
| **镜像** | ✅ | `localhost/bimamba-train:latest` (已构建) |
| **GPU** | ⏳ | 服务器 16GB GPU (待验证) |
| **PyTorch** | ✅ | 2.4.0 + CUDA 12.4 |
| **RDKit** | ✅ | 2025.9.6 |
| **scikit-learn** | ✅ | 1.8.0 |
| **数据集** | ⏳ | 需在服务器上运行 `download_datasets.py` |
| **训练测试** | ⏳ | 需在服务器上验证 |

### 1.2 服务器部署步骤
```bash
# 1. 上传项目到服务器
scp -r /home/ziyu/graduation-project user@server:/path/to/

# 2. 在服务器上构建镜像（如需要）
podman build -t localhost/bimamba-train:latest .

# 3. 启动容器并下载数据
podman run --gpus all --rm -it \
  -v /path/to/graduation-project:/workspace \
  localhost/bimamba-train:latest bash
  
  # 容器内
  python /workspace/download_datasets.py

# 4. 验证环境
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import rdkit; print('RDKit:', rdkit.__version__)"

# 5. 运行训练测试
python /workspace/train.py --dataset ESOL --d_model 128 --n_layers 2 --epochs 1
```

---

## 二、实验设计

### 2.1 数据集概览
| 数据集 | 任务类型 | 分子数 | 评估指标 | 预计训练时间 |
|--------|---------|--------|---------|-------------|
| ESOL | 回归 | ~1,100 | RMSE, MAE | ~3min/epoch |
| BBBP | 分类 | ~2,000 | ROC-AUC | ~5min/epoch |
| ClinTox | 分类 | ~1,500 | ROC-AUC | ~4min/epoch |

### 2.2 GPU 显存约束 (16GB GPU)

根据 16GB 显存设置动态 batch_size：

| d_model | n_layers | 最大 batch_size |
|----------|----------|-----------------|
| 128 | 2, 4, 6 | 32 |
| 256 | 2, 4 | 32 |
| 256 | 6 | 16 |
| 512 | 2 | 32 |
| 512 | 4 | 16 |
| 512 | 6 | 8 |

### 2.3 超参数搜索空间

#### 模型结构参数
| 参数 | 搜索值 | 说明 |
|------|--------|------|
| `d_model` | 128, 256, 512 | 模型维度 |
| `n_layers` | 2, 4, 6 | Bi-Mamba 层数 |

#### 训练策略参数
| 参数 | 搜索值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-4, 5e-4, 1e-3 | 学习率 |
| `batch_size` | **动态** (见上表) | 批大小 |
| `dropout` | 0.0, 0.1, 0.2 | Dropout 率 |

#### Pooling 策略
| 参数 | 搜索值 |
|------|--------|
| `pooling` | mean, max, cls |

#### 融合模式 (已在代码中固定为 `gate`)
> 注：BiMambaEncoder 使用 `gate` 融合前向/后向隐藏状态

### 2.4 实验矩阵

#### 组合数量计算
- `d_model`: 3 种
- `n_layers`: 3 种
- `learning_rate`: 3 种
- `batch_size`: 3 种
- `dropout`: 3 种
- `pooling`: 3 种

**每数据集超参数组合数**: 3×3×3×3×3×3 = **729 种**

#### 阶段化实验设计（避免组合爆炸）

| 阶段 | 目标 | 策略 |
|------|------|------|
| **Phase 1** | 确定最佳 pooling 和模型结构 | 固定 lr=1e-3, bs=16, dropout=0.1；对比 pooling × d_model × n_layers |
| **Phase 2** | 优化训练策略 | 固定最佳 pooling 和结构；对比 lr × bs × dropout |
| **Phase 3** | 最终调参与验证 | 在最佳配置上进行 5 次重复实验 |

### 2.4 各阶段详细配置

#### Phase 1: 模型结构探索
```
固定参数:
  learning_rate = 1e-3
  dropout = 0.1
  epochs = 100

变量组合:
  pooling ∈ {mean, max, cls}
  d_model ∈ {128, 256, 512}
  n_layers ∈ {2, 4, 6}

batch_size (动态，根据GPU显存约束):
  d_model=128:        batch_size = 32
  d_model=256, n≤4:   batch_size = 32
  d_model=256, n=6:   batch_size = 16
  d_model=512, n=2:    batch_size = 32
  d_model=512, n=4:    batch_size = 16
  d_model=512, n=6:    batch_size = 8

每数据集实验数: 3 × 3 × 3 = 27
3个数据集 × 27 = 81 experiments
每次实验: 100 epochs
```

#### Phase 2: 训练策略优化
```
固定参数 (Phase 1 最佳):
  最佳 pooling
  最佳 d_model
  最佳 n_layers
  epochs = 100

batch_size (根据固定的最佳结构):
  d_model=128:        batch_size ∈ {16, 32}
  d_model=256, n≤4:   batch_size ∈ {16, 32}
  d_model=256, n=6:   batch_size ∈ {8, 16}
  d_model=512, n=2:    batch_size ∈ {16, 32}
  d_model=512, n=4:    batch_size ∈ {8, 16}
  d_model=512, n=6:    batch_size ∈ {4, 8}

变量组合:
  learning_rate ∈ {1e-4, 5e-4, 1e-3}
  batch_size ∈ {动态范围，见上}
  dropout ∈ {0.0, 0.1, 0.2}

每数据集实验数: 3 × 2 × 3 = 18
3个数据集 × 18 = 54 experiments
```

#### Phase 3: 最终验证 (5次重复)
```
固定最佳配置:
  最佳 pooling
  最佳 d_model
  最佳 n_layers
  最佳 learning_rate
  最佳 batch_size (根据约束确定)
  最佳 dropout
  epochs = 150  (增加 epochs 以获得更稳定结果)

重复: 5 次 × 3 个数据集 = 15 experiments
```

---

## 三、时间估算

### 3.1 单次实验时间
| 数据集 | Epochs | 小 batch_size | 大 batch_size |
|--------|--------|-------------|---------------|
| ESOL | 100 | ~8 分钟 | ~4 分钟 |
| BBBP | 100 | ~12 分钟 | ~6 分钟 |
| ClinTox | 100 | ~10 分钟 | ~5 分钟 |

> 注：batch_size 较小会导致每 epoch 时间稍长

### 3.2 总时间估算

| 阶段 | 实验数 | 平均时间 | 总时间 |
|------|--------|---------|--------|
| Phase 1 | 81 | ~7 分钟 | ~9.5 小时 |
| Phase 2 | 54 | ~7 分钟 | ~6.3 小时 |
| Phase 3 | 15 | ~8 分钟 | ~2 小时 |
| **总计** | **150** | - | **~18 小时** |

> **注意**: Phase 2 减少到 54 个实验（batch_size 搜索空间缩小为 2 个值）
> Phase 3 epochs 增加到 150 以获得更稳定的验证结果

---

## 四、实验执行脚本

### 4.1 自动批量训练脚本 (16GB GPU 优化版)
```python
#!/usr/bin/env python3
"""
batch_train.py - 批量自动训练脚本 (16GB GPU 优化)
输出目录: experiment-data/
"""
import itertools
import subprocess
import json
import os
from datetime import datetime

# GPU 显存约束映射 (16GB GPU)
GPU_BATCH_SIZE_MAP = {
    (128, 2): 32, (128, 4): 32, (128, 6): 32,
    (256, 2): 32, (256, 4): 32, (256, 6): 16,
    (512, 2): 32, (512, 4): 16, (512, 6): 8,
}

def get_batch_size(d_model, n_layers):
    """根据模型配置获取安全的 batch_size"""
    return GPU_BATCH_SIZE_MAP.get((d_model, n_layers), 16)

# 阶段1配置
PHASE1_CONFIG = {
    "pooling": ["mean", "max", "cls"],
    "d_model": [128, 256, 512],
    "n_layers": [2, 4, 6],
    "learning_rate": [1e-3],
    # batch_size 由 get_batch_size() 动态确定
    "dropout": [0.1],
    "epochs": 100,
}

def generate_experiments(config, dataset):
    """生成所有实验组合"""
    keys = [k for k in config if isinstance(config[k], list)]
    values = [config[k] for k in keys]
    
    for combo in itertools.product(*values):
        exp = {k: config[k] for k in config if k not in keys}
        exp.update(dict(zip(keys, combo)))
        exp["dataset"] = dataset
        # 动态设置 batch_size
        exp["batch_size"] = get_batch_size(exp["d_model"], exp["n_layers"])
        yield exp

def run_experiment(exp_config, seed=42):
    """运行单个实验"""
    # 实验输出目录
    output_dir = f"experiment-data/checkpoints/{exp_config['dataset']}"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python", "train.py",
        "--dataset", exp_config["dataset"],
        "--d_model", str(exp_config["d_model"]),
        "--n_layers", str(exp_config["n_layers"]),
        "--pooling", exp_config["pooling"],
        "--learning_rate", str(exp_config["learning_rate"]),
        "--batch_size", str(exp_config["batch_size"]),
        "--dropout", str(exp_config["dropout"]),
        "--epochs", str(exp_config["epochs"]),
        "--seed", str(seed),
        "--device", "cuda",
        "--output_dir", output_dir,
        "--db_path", "experiment-data/experiments.db",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

# 示例用法
for dataset in ["ESOL", "BBBP", "ClinTox"]:
    for exp in generate_experiments(PHASE1_CONFIG, dataset):
        print(f"Running: {dataset} - d_model={exp['d_model']}, "
              f"n_layers={exp['n_layers']}, pooling={exp['pooling']}, "
              f"batch_size={exp['batch_size']}")
        # run_experiment(exp)
```

### 4.2 快速启动命令（手动单次训练）

```bash
# ESOL 回归示例 (d_model=256, n_layers=4 -> batch_size=32)
python train.py \
    --dataset ESOL \
    --d_model 256 \
    --n_layers 4 \
    --pooling mean \
    --learning_rate 1e-3 \
    --batch_size 32 \
    --dropout 0.1 \
    --epochs 100 \
    --device cuda \
    --seed 42

# BBBP 分类示例 (d_model=512, n_layers=6 -> batch_size=8, 16GB GPU 安全)
python train.py \
    --dataset BBBP \
    --task_type classification \
    --d_model 512 \
    --n_layers 6 \
    --pooling mean \
    --learning_rate 1e-3 \
    --batch_size 8 \
    --dropout 0.1 \
    --epochs 100 \
    --device cuda
```

### 4.3 Phase 2 配置生成（根据 Phase 1 最佳结果填充）

```python
# Phase 2 需要根据 Phase 1 结果确定最佳配置后填充
# 这里展示模板

# 假设 Phase 1 最佳配置为: d_model=256, n_layers=4, pooling=mean
PHASE2_BEST = {
    "d_model": 256,
    "n_layers": 4,
    "pooling": "mean",
}

PHASE2_CONFIG = {
    "pooling": [PHASE2_BEST["pooling"]],
    "d_model": [PHASE2_BEST["d_model"]],
    "n_layers": [PHASE2_BEST["n_layers"]],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "dropout": [0.0, 0.1, 0.2],
    "epochs": 100,
}

def get_phase2_batch_sizes(d_model, n_layers):
    """Phase 2 的 batch_size 搜索空间（比 Phase 1 更精细）"""
    base = get_batch_size(d_model, n_layers)
    if base >= 16:
        return [base // 2, base]
    else:
        return [base, base * 2]  # 小 batch_size 配置搜索更小值

# 示例: d_model=256, n_layers=4 -> batch_size ∈ {16, 32}
```

---

## 五、实验记录与分析

### 5.1 实验记录文件
每次实验自动保存：
- `checkpoints/{dataset}_bi_mamba_best.pt` - 最佳模型
- `args.json` - 实验参数
- `training.log` - 训练日志

### 5.2 结果分析脚本
```python
"""
analyze_results.py - 实验结果分析
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path

def load_experiments(db_path="experiment-data/experiments.db"):
    """从数据库加载实验结果"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM experiments", conn)
    conn.close()
    return df

def find_best_config(df, metric="val_loss", minimize=True):
    """找出最佳配置"""
    if minimize:
        return df.loc[df[metric].idxmin()]
    return df.loc[df[metric].idxmax()]

# 使用示例
# df = load_experiments()
# best = find_best_config(df, "val_rmse")
# print(best)
```

---

## 六、GPU 显存安全检查表 (16GB GPU)

在运行任何实验前，确认 batch_size 与 GPU 显存匹配：

```
✅ d_model=128, n_layers=2-6:     batch_size ≤ 32
✅ d_model=256, n_layers=2-4:      batch_size ≤ 32
✅ d_model=256, n_layers=6:        batch_size ≤ 16
✅ d_model=512, n_layers=2:        batch_size ≤ 32
✅ d_model=512, n_layers=4:        batch_size ≤ 16
⚠️ d_model=512, n_layers=6:        batch_size ≤ 8  (显存紧张，需监控)
```

> 如果遇到 OOM (Out of Memory)，立即减小 batch_size 或使用 `gradient_accumulation_steps` 增大有效 batch

---

## 七、建议的实验顺序

```
Week 1 (Day 1-2): 环境搭建与验证
  ├── 修复 mamba-env 缺少的包 (rdkit, sklearn)
  ├── 下载并验证数据集
  └── 运行 1-2 个 quick test 确保流程通顺

Week 1 (Day 3-5): Phase 1 - 模型结构探索
  ├── 81 experiments
  └── 每天约 27 个实验

Week 2 (Day 1-2): Phase 2 - 训练策略优化
  ├── 54 experiments
  └── 每天约 27 个实验

Week 2 (Day 3): Phase 3 - 最终验证
  ├── 15 experiments (5 runs × 3 datasets)
  └── 结果分析与报告
```

---

## 附录: 关键文件位置

| 文件 | 位置 | 说明 |
|------|------|------|
| **实验数据** | `experiment-data/` | **所有实验结果存放目录** |
| 数据库 | `experiment-data/experiments.db` | 实验记录 (SQLite) |
| 检查点 | `experiment-data/checkpoints/` | 模型权重 |
| 日志 | `experiment-data/logs/` | 训练日志 |
| 结果 | `experiment-data/results/` | 分析结果 |
| 训练脚本 | `train.py` | 主训练入口 |
| 评估脚本 | `eval.py` | 模型评估 |
| 数据下载 | `download_datasets.py` | 获取数据集 |
| 源码 | `src/` | 模型实现 |
| Dockerfile | `Dockerfile` | 容器镜像构建 |

### experiment-data/ 目录结构
```
experiment-data/
├── experiments.db          # SQLite 数据库
├── checkpoints/           # 模型权重
│   ├── ESOL_d_model=256_nlayers=4/
│   └── BBBP_d_model=512_nlayers=6/
├── logs/                 # 训练日志
│   ├── phase1_ESOL_001.log
│   └── phase2_BBBP_001.log
└── results/              # 分析结果
    ├── phase1_summary.csv
    └── best_config.json
```

---

## 附录: Dockerfile 构建

构建支持 16GB GPU 的训练镜像：

```bash
# 构建镜像
podman build -t localhost/bimamba-train:latest .

# 验证镜像
podman run --rm localhost/bimamba-train:latest python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 运行训练 (示例) - 数据输出到 experiment-data/
podman run --gpus all --rm \
  -v /home/ziyu/graduation-project:/workspace \
  localhost/bimamba-train:latest \
  python /workspace/train.py \
    --dataset ESOL \
    --d_model 256 \
    --n_layers 4 \
    --batch_size 32 \
    --epochs 100 \
    --output_dir /workspace/experiment-data/checkpoints \
    --db_path /workspace/experiment-data/experiments.db
```

> **注意**: `--gpus all` 需要 NVIDIA Container Toolkit 支持
