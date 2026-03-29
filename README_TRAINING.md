# Bi-Mamba-Chem Training Guide

## 项目结构

本项目包含两个 BiMamba 模型实现：

| 模型 | 文件 | 依赖 | 位置 |
|------|------|------|------|
| **Manual SSM** | `src/models/bimamba.py` | 纯 PyTorch，无外部依赖 | 通用 |
| **mamba_ssm** | `src/models/bimamba_with_mamba_ssm.py` | 需要 `mamba-ssm` 包 | 需要 CUDA |

## 快速开始

### 方式一：本地训练（CPU，仅 manual 模型）

```bash
# 激活 conda 环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate chem

# 训练 manual 模型
python train.py --dataset ESOL --epochs 10 --batch_size 16 --device cpu --model_type manual --no_db
```

### 方式二：Docker/Podman 容器（支持两个模型）

#### 1. 拉取基础镜像（约 15GB，网络慢时可能需要 30 分钟以上）

```bash
podman pull docker.io/pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

#### 2. 构建包含 mamba-ssm 的训练镜像

```bash
cd /path/to/graduation-project
podman build -t bimamba-train:latest .
```

#### 3. 运行训练

```bash
# 训练 manual 模型（无需 mamba-ssm）
podman run --gpus all \
  -v $(pwd):/app:rw \
  bimamba-train:latest \
  python train.py --dataset ESOL --epochs 100 --batch_size 32 --device cuda --model_type manual

# 训练 mamba_ssm 模型（需要 mamba-ssm）
podman run --gpus all \
  -v $(pwd):/app:rw \
  bimamba-train:latest \
  python train.py --dataset ESOL --epochs 100 --batch_size 32 --device cuda --model_type mamba_ssm
```

#### GPU 配置说明

Podman 5.0+ 原生支持 `--gpus all`：

```bash
podman run --gpus all --device nvidia.com/gpu=all bimamba-train:latest nvidia-smi
```

**前提条件**：
- 安装 `nvidia-container-toolkit`（注意是 toolkit，不是 nvidia-container-runtime）
- RHEL/Fedora 需要在 `containers.conf` 中设置 `no-cgroups = true`

## 代码修改记录

### `--model_type` 参数

`train.py` 已添加 `--model_type` 参数选择模型类型：

```bash
python train.py --model_type manual   # 使用 bimamba.py（无外部依赖）
python train.py --model_type mamba_ssm # 使用 bimamba_with_mamba_ssm.py（需要 mamba-ssm）
```

### 延迟导入修复

`src/models/bimamba_with_mamba_ssm.py` 已修改为延迟导入 `mamba_ssm`：

```python
# 修改前（模块级别导入）
from mamba_ssm import Mamba2

# 修改后（__init__ 中延迟导入）
class BiMambaBlock(nn.Module):
    def __init__(self, ...):
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(...)
```

这样即使 `mamba_ssm` 未安装，`manual` 模型也能正常导入。

## 依赖说明

### Manual 模型依赖

```
torch>=2.0.0
rdkit-pypi>=2022.9.5
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.3.0
```

### mamba-ssm 额外依赖

```
causal-conv1d>=1.4.0  # 必须先安装
mamba-ssm>=1.2.0       # 需要 CUDA 11.6+, PyTorch 1.12+
```

**编译要求**：mamba-ssm 需要 nvcc 编译器（CUDA Toolkit），CPU-only 环境无法安装。

## 数据集

数据放在 `data/` 目录下：

```
data/
├── ESOL/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── BBBP/
└── ClinTox/
```

每个数据集需要 `smiles,label` 格式的 CSV 文件。

## 检查点保存

训练后检查点保存在 `./checkpoints/` 目录：

```
checkpoints/
├── ESOL_bi_mamba_best.pt          # 最佳模型
├── ESOL_bi_mamba_epoch_{N}.pt     # 定期保存
└── args.json                       # 训练参数
```

## 镜像构建缓存优化

Dockerfile 使用多阶段构建减小镜像大小：

- **Builder stage**：包含编译工具链，安装 mamba-ssm
- **Runtime stage**：仅包含运行时依赖（无编译器）

如需跳过 mamba-ssm 安装，可注释掉 Dockerfile 中相关行。
