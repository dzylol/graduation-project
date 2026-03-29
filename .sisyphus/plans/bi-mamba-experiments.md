# Plan: Bi-Mamba-Chem 论文实验计划

## TL;DR

> 基于论文大纲《基于双向状态空间模型（Bi-Mamba）的分子性质预测方法研究》，规划所有实验。

**论文题目**: Efficient Molecular Property Prediction via Bidirectional Mamba Architecture

**实验数据目录**: `./experiment-data/`

**Estimated Effort**: XL (多个阶段的大规模实验)
**Parallel Execution**: YES (不同数据集可并行)

---

## 第一阶段：数据集准备

### 1.1 MoleculeNet 标准数据集

| 数据集 | 任务 | 分子数 | 指标 | 用途 |
|--------|------|--------|------|------|
| ESOL | 回归 | 1,128 | RMSE | 主实验 |
| BBBP | 二分类 | 2,039 | ROC-AUC | 主实验 |
| ClinTox | 二分类 | 1,478 | ROC-AUC | 主实验 |
| FreeSolv | 回归 | 642 | RMSE | 消融 |
| Lipophilicity | 回归 | 4,200 | RMSE | 消融 |

**下载命令**:
```bash
python download_datasets.py --dataset ESOL
python download_datasets.py --dataset BBBP
python download_datasets.py --dataset ClinTox
python download_datasets.py --dataset FreeSolv
python download_datasets.py --dataset Lipophilicity
```

### 1.2 长序列数据集（关键亮点）

**目的**: 突显 Mamba 的长程记忆优势，区别于普通 Transformer 论文

| 数据集 | 描述 | 来源 | 任务 | 分子数 | 序列长度 |
|--------|------|------|------|--------|----------|
| Peptide | 多肽序列 | PubChem/文献 | 回归 | ~2,000 | 50-200 |
| Long-Mol | 大分子化合物 | PDB | 回归 | ~1,000 | >100 atoms |

**要求**: 筛选 SMILES 长度 > 100 的分子

### 1.3 基线模型数据

| 基线模型 | 类型 | 数据来源 | 准备方式 |
|----------|------|----------|----------|
| ECFP + XGBoost | 传统ML | MoleculeNet | 特征提取脚本 |
| Bi-LSTM | RNN类 | MoleculeNet | 复用相同数据 |
| GCN/GAT | 图神经网络 | MoleculeNet | 复用相同数据 |
| ChemBERTa | Transformer | MoleculeNet | HuggingFace 加载 |

---

## 第二阶段：主实验（4.2 主实验结果）

### 2.1 实验配置

```bash
# 训练脚本
python train.py \
    --dataset ESOL \
    --epochs 100 \
    --batch_size 32 \
    --device mps \
    --d_model 256 \
    --n_layers 4 \
    --pooling mean \
    --learning_rate 1e-3
```

### 2.2 主实验结果表

**输出文件**: `experiment-data/main-results.csv`

| 模型 | 数据集 | 指标 | 值 | 标准差 |
|------|--------|------|-----|--------|
| Bi-Mamba | ESOL | RMSE | TBD | ±TBD |
| Bi-Mamba | BBBP | ROC-AUC | TBD | ±TBD |
| Bi-Mamba | ClinTox | ROC-AUC | TBD | ±TBD |
| ... | ... | ... | ... | ... |

### 2.3 效率对比（4.3）

**实验**: 推理速度 vs 序列长度

```
序列长度: [32, 64, 128, 256, 512, 1024]
测量: 推理时间 (ms), GPU 显存 (MB)
模型: Bi-Mamba, Transformer (ChemBERTa)
```

**输出文件**: 
- `experiment-data/inference-time-vs-length.png`
- `experiment-data/gpu-memory-vs-length.png`

---

## 第三阶段：消融实验（5.1）

### 3.1 双向机制必要性

| 实验组 | 配置 | 说明 |
|--------|------|------|
| Uni-Mamba | n_layers=4, bidirectional=False | 单向基线 |
| Bi-Mamba | n_layers=4, bidirectional=True | 双向（本文方法） |

**数据集**: ESOL, BBBP, ClinTox, Peptide

### 3.2 池化策略对比

| 实验组 | 配置 |
|--------|------|
| Mean Pooling | pooling=mean |
| Max Pooling | pooling=max |
| [CLS] Pooling | pooling=cls |

### 3.3 融合机制对比（3.3）

| 实验组 | 配置 |
|--------|------|
| Concat | fusion=concat |
| Add | fusion=add |
| Gated | fusion=gated |

---

## 第四阶段：长程依赖分析（5.2）

### 4.1 序列长度分组测试

**分组**:
- Short: length < 50
- Medium: 50 ≤ length < 100
- Long: length ≥ 100

**指标**: 按组计算 RMSE/ROC-AUC

**输出文件**: `experiment-data/length-group-analysis.csv`

### 4.2 效率随长度变化

测试不同序列长度下，各模型的:
- 推理时间
- GPU 显存
- 精度保持率

---

## 第五阶段：可视化与分析（5.3）

### 5.1 预测案例分析

**选取分子**:
- 预测正确的典型案例
- 预测错误的典型案例
- 长序列分子案例

### 5.2 注意力可视化

使用梯度回传方法（Saliency Map）：
- `experiment-data/saliency-maps/`

---

## 实验清单

### TODO-1: 数据集下载与预处理

- [ ] 1.1.1 下载 ESOL, BBBP, ClinTox, FreeSolv, Lipophilicity
- [ ] 1.1.2 筛选长序列分子 (length > 100)
- [ ] 1.1.3 生成 dataset metadata JSON

### TODO-2: 基线模型复现

- [ ] 2.1.1 ECFP + XGBoost 基线
- [ ] 2.1.2 Bi-LSTM 基线
- [ ] 2.1.3 GCN/GAT 基线（如有开源代码）

### TODO-3: Bi-Mamba 主实验

- [ ] 3.1.1 ESOL 回归实验 (5 runs)
- [ ] 3.1.2 BBBP 分类实验 (5 runs)
- [ ] 3.1.3 ClinTox 分类实验 (5 runs)
- [ ] 3.1.4 Peptide 长序列实验 (5 runs)

### TODO-4: 效率对比实验

- [ ] 4.1.1 推理时间 vs 序列长度
- [ ] 4.1.2 GPU 显存 vs 序列长度

### TODO-5: 消融实验

- [ ] 5.1.1 Uni-Mamba vs Bi-Mamba (所有数据集)
- [ ] 5.1.2 池化策略对比 (Mean/Max/[CLS])
- [ ] 5.1.3 融合机制对比 (Concat/Add/Gated)

### TODO-6: 可视化

- [ ] 6.1.1 预测散点图
- [ ] 6.1.2 训练曲线
- [ ] 6.1.3 Saliency Map

---

## 输出文件结构

```
experiment-data/
├── datasets/
│   ├── ESOL/
│   ├── BBBP/
│   ├── ClinTox/
│   ├── FreeSolv/
│   ├── Lipophilicity/
│   └── Peptide/
├── baselines/
│   ├── ecfp_xgboost/
│   ├── bi_lstm/
│   └── gcn/
├── main-results/
│   ├── esol_results.csv
│   ├── bbbp_results.csv
│   └── ...
├── ablation/
│   ├── uni_vs_bi/
│   ├── pooling/
│   └── fusion/
├── efficiency/
│   ├── inference-time.png
│   └── gpu-memory.png
└── visualization/
    ├── scatter-plots/
    ├── training-curves/
    └── saliency-maps/
```

---

## 成功标准

### 主实验
- Bi-Mamba 在 ESOL 上 RMSE < 1.0
- Bi-Mamba 在 BBBP 上 ROC-AUC > 0.85
- Bi-Mamba 在长序列上显著优于 LSTM

### 效率实验
- Mamba 推理时间呈线性增长 (O(N))
- Transformer 推理时间呈二次增长 (O(N²))

### 消融实验
- Bi-Mamba 在所有数据集上优于 Uni-Mamba
- 最佳池化策略确定
