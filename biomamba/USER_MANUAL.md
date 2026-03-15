# Bi-Mamba-Chem 用户手册

## 项目简介

Bi-Mamba-Chem 是一个基于**双向Mamba架构**的分子属性预测模型。该项目探索了使用**状态空间模型(SSM)** 进行分子性质预测的可能性,并展示了相对于Transformer模型的**O(N)线性复杂度**优势。

### 核心创新点

1. **双向扫描**: 同时从分子序列的两个方向提取化学环境信息
2. **状态空间模型**: 使用选择性扫描算法,高效处理长序列分子
3. **门控融合**: 智能融合前向和后向表示

---

## 快速开始

### 环境准备

```bash
# 激活你的conda/pip环境
conda activate biomamba  # 或 source activate biomamba

# Mac用户需要设置此环境变量(避免OpenMP冲突)
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 训练模型

```bash
cd biomamba

# 使用交互式设备选择(推荐)
python train.py --dataset ESOL --epochs 100

# 指定设备
python train.py --dataset ESOL --device mps --epochs 100   # Apple Silicon
python train.py --dataset ESOL --device cuda --epochs 100  # NVIDIA GPU
python train.py --dataset ESOL --device cpu --epochs 100   # CPU

# 常用训练参数
python train.py --dataset ESOL --batch_size 64 --num_workers 4
```

### 评估模型

```bash
# 使用训练好的模型进行评估
python eval.py --checkpoint checkpoints/ESOL_bi_mamba_best.pt --dataset ESOL
```

---

## 支持的数据集

| 数据集 | 任务类型 | 描述 | 评估指标 |
|--------|----------|------|----------|
| ESOL | 回归 | 水溶解度预测 | RMSE |
| BBBP | 分类 | 血脑屏障穿透性 | ROC-AUC |
| ClinTox | 分类 | 临床毒性预测 | ROC-AUC |

---

## 命令行参数详解

### 训练参数 (train.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集选择 | ESOL |
| `--device` | 计算设备 | auto |
| `--epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 64 |
| `--lr` | 学习率 | 0.001 |
| `--num_workers` | 数据加载线程数 | 4 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--d_model` | 模型隐藏层维度 | 256 |
| `--n_layers` | 模型层数 | 4 |
| `--d_state` | SSM状态维度 | 128 |
| `--d_conv` | 卷积核大小 | 4 |
| `--fusion` | 双向融合方式 | gate |
| `--pool_type` | 池化方式 | mean |

### 融合方式说明

- **gate (推荐)**: 门控融合,使用可学习的门控机制权衡两个方向的信息
- **concat**: 拼接前后向表示,维度翻倍
- **add**: 简单相加,维度不变

### 池化方式说明

- **mean**: 平均池化,对所有token取平均
- **max**: 最大池化,对所有token取最大值
- **cls**: 使用[CLS]标记的表示

---

## 项目架构

```
biomamba/
├── config.py          # 配置文件,所有超参数的默认值
├── train.py           # 训练脚本
├── eval.py            # 评估脚本
├── data/              # 数据处理模块
│   ├── tokenizer.py   # SMILES分词器
│   └── dataset.py    # 数据集加载
├── models/           # 模型定义
│   ├── ssm_core.py   # SSM核心实现
│   ├── mamba_block.py # Mamba块
│   ├── bi_mamba.py   # 双向Mamba
│   └── predictor.py  # 预测头
└── utils/            # 工具函数
    ├── metrics.py    # 评估指标
    └── logger.py     # 日志记录
```

### 数据流

```
SMILES字符串 → AtomTokenizer → 整数ID序列
                                      ↓
                              BiMambaEncoder (双向处理)
                                      ↓
                              PredictionHead (池化+MLP)
                                      ↓
                              预测结果 (回归值或分类概率)
```

---

## 训练过程解读

### 1. 数据加载

分词器将SMILES字符串转换为整数序列:
- `CCO` → `[12, 12, 15]` (示例)
- 支持元素符号: C, H, O, N, Cl, Br等
- 支持特殊符号: # (三键), % (环闭合), [ ] (括号原子)

### 2. 模型前向

1. **Embedding**: 将整数ID转换为向量表示
2. **位置编码**: 添加位置信息(虽然SSM不严格需要,但有助于学习)
3. **Bi-Mamba Encoder**:
   - 分为前向和后向两个分支
   - 每个分支使用Mamba块处理
   - 最后使用门控机制融合两个方向
4. **Pooling**: 将序列表示聚合成单个向量
5. **Prediction Head**: 输出最终预测

### 3. 损失函数

- **回归任务**: MSE (Mean Squared Error)
- **分类任务**: BCE (Binary Cross Entropy)

### 4. 优化

- **优化器**: AdamW
- **学习率调度**: Warmup + Cosine Decay
- **正则化**: Dropout + Weight Decay + Gradient Clipping

---

## 评估指标说明

### 回归任务

- **RMSE** (均方根误差): 预测值与真实值差异的平方根,越小越好
- **MAE** (平均绝对误差): 预测值与真实值差异的绝对值的平均,越小越好
- **R²** (决定系数): 模型解释目标变量变异的程度,越接近1越好

### 分类任务

- **Accuracy** (准确率): 正确预测的比例
- **Precision** (精确率): 预测为正类中实际为正类的比例
- **Recall** (召回率): 实际为正类中被正确预测的比例
- **F1**: Precision和Recall的调和平均
- **AUC**: ROC曲线下面积,越接近1越好

---

## 常见问题

### Q: 为什么选择Mamba而不是Transformer?

A: Mamba等状态空间模型具有O(N)的线性复杂度,而Transformer是O(N²)。对于长序列分子(如聚合物、蛋白质),Mamba可以更高效地处理。

### Q: 如何选择融合方式?

A: `gate`是默认推荐,它通过可学习的门控机制自适应地权衡两个方向的信息。`concat`维度翻倍,`add`最简单但可能表达能力有限。

### Q: 训练时间太长怎么办?

A: 尝试以下优化:
- 减小`--d_model`和`--n_layers`
- 增加`--batch_size`
- 使用GPU加速(推荐Apple Silicon或NVIDIA GPU)
- 减少`--epochs`(使用早停)

### Q: 模型过拟合怎么办?

A:
- 增加`--dropout`
- 减小模型容量(d_model, n_layers)
- 增加`weight_decay`
- 使用验证集早停

---

## 输出文件

训练后会生成以下文件:

```
checkpoints/
├── ESOL_bi_mamba_last.pt    # 最后一个epoch的模型
├── ESOL_bi_mamba_best.pt    # 验证集上最好的模型
└── ...

logs/
├── training_20240101_120000.log  # 训练日志
└── training_metrics.json         # 训练指标历史
```

---

## 设计思路

### 为什么使用双向扫描?

分子中的化学基团可以出现在中心基团的左边或右边。双向扫描让模型能够同时从两个方向学习化学环境信息,类似于BERT在自然语言处理中的双向表示学习。

### 门控融合的优势

简单的拼接(add)或拼接(concat)会平等地对待两个方向的表示。但实际上,对于不同的分子和不同的预测任务,两个方向的信息重要性可能不同。门控机制允许模型学习这种重要性权重。

### SSM的选择性扫描

传统的SSM使用卷积进行计算,而Mamba引入了选择性机制,允许模型动态决定关注哪些输入。这对于分子预测特别有用,因为不同的原子位置可能对最终属性有不同程度的贡献。
