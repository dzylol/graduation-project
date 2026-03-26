# Bi-Mamba (mamba_ssm 版本) 架构解析

> 本文档解析 `src/models/bimamba_with_mamba_ssm.py` 的代码架构，对比 `bimamba.py`（手动实现）的差异。
>
> **学习路径**：先读 [§1 总览](#1-总览) 建立整体印象 → [§2 核心原理](#2-核心原理) 理解数学基础 → [§3 代码实现](#3-代码实现) 追踪数据流 → [§4 对比 bimambapy](#4-与-bimapmypy-的差异) 理解设计权衡。
>
> **本文档特色**：每个变换都附有**具体数值示例**，展示 (B=2, L=4, d_model=8, d_state=4) 下的完整计算过程。

---

## 1. 总览

### 1.1 三个核心类

```
BiMambaBlock                   ← 最底层：单层选择性 SSM（Mamba2 封装）
    ↓
BiMambaEncoder                 ← 中间层：双向堆叠 + 融合门
    ↓
BiMambaForPropertyPrediction   ← 最上层：Embedding → Encoder → Pooling → Classifier
```

### 1.2 数据流总图

```
输入: SMILES token ids  (B, L)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ BiMambaForPropertyPrediction                             │
│                                                         │
│   ① Token Embedding + Position Embedding                 │
│      (vocab_size → d_model)                             │
│   ② BiMambaEncoder                                      │
│      ├─ Forward: L× BiMambaBlock                         │
│      └─ Backward: L× BiMambaBlock + flip               │
│   ③ Pooling (mean / max / cls)                         │
│   ④ Dropout → Classifier (d_model → num_labels)         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
输出: 预测值 (B,) 或 (B, num_labels)
```

### 1.3 参数一览

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `d_model` | 256 | 每个 token 的向量维度（hidden dimension） |
| `d_state` | 64 | SSM 状态维度（state space dimension） |
| `d_conv` | 4 | 因果卷积核宽度（局部上下文） |
| `expand` | 2 | 内部维度扩展因子，`d_inner = expand * d_model = 512` |
| `n_layers` | 4 | 每方向的 Mamba 层数（总层数 = 2×n_layers） |
| `max_seq_length` | 512 | 最大序列长度（用于 Position Embedding） |

### 1.4 三个关键维度详解

> **常见混淆**：`d_model` 不是层数！它是每个 token 的向量长度。层数是 `n_layers`。

#### 统一对比表

| 维度 | 默认值 | 本质 | 类比 |
|------|--------|------|------|
| **d_model** | 256 | 每个 token 的**向量长度** | 图像的通道数（RGB=3） |
| **d_state** | 64 | SSM 的**记忆容量** | 记忆细胞的数量 |
| **n_layers** | 4 | Mamba Block 的**堆叠层数**（每方向） | 网络深度 |

#### d_model（向量维度）

```
d_model = 256:  每个 token 的向量长度 = 256

"CCO" (乙醇)
  C → [0.12, -0.34, 0.56, ..., 0.67]  ← 256 维向量
  C → [0.91, 0.23, -0.45, ..., 0.89]  ← 256 维向量
  O → [0.19, -0.28, 0.37, ..., -0.82] ← 256 维向量

维度变化：
  输入: (B, L)                    # B=batch, L=序列长度
    ↓ token_embedding (vocab_size → d_model)
  (B, L, d_model=256)             # 每个位置变成 256 维向量
```

#### d_state（SSM 状态维度）

```
d_state = 64:  SSM 内部状态维度

每个 token 位置有一个 64 维的"记忆状态"：
  h_t: [h_0, h_1, h_2, ..., h_63]  ← 64 维状态向量
        ↓      ↓      ↓
       最慢    中等    最快
       衰减    衰减    衰减
       记忆    记忆    记忆
```

#### d_state 与 A 的关系（常见误解辨析）

**d_state** 和 **A** 是两个相关但本质不同的东西：

| 符号 | 名称 | 本质 | 形状 |
|------|------|------|------|
| **d_state** (N) | 状态维度 | 一个**整数**——表示用多少个正交基函数 | 标量，如 64 |
| **A** | HiPPO 矩阵 | 一个**矩阵**——定义了每个状态如何随时间衰减 | (d_state, d_state)，如 (64, 64) |

**类比**：

```
d_state = 4  →  有 4 个"记忆槽"
A = diag([-1, -2, -3, -4])  →  每个槽的衰减速度不同

记忆槽 0（衰减率 -1）：e^(-t) → 衰减很慢，能记住很久以前的事
记忆槽 1（衰减率 -2）：e^(-2t) → 衰减稍快
记忆槽 2（衰减率 -3）：e^(-3t) → 衰减更快
记忆槽 3（衰减率 -4）：e^(-4t) → 几乎只关注当前时刻
```

**物理直觉**：把 A 想象成一组"记忆槽"的衰减控制器：

```
A[i,i] = -(i+1)  →  第 i 个槽的衰减率

d_state 越大 → 记忆槽越多 → 能同时捕捉从"极慢变化"到"极快变化"的模式
d_state = 1：只能记住一种时间尺度的信息
d_state = 64：同时记住 64 种不同时间尺度的信息（从毫秒到分钟）
```

**一句话总结**：`d_state` 是"要多少个记忆槽"，`A` 是"每个槽衰减多快"。d_state 是 A 的维度规格，A 是具体衰减系数矩阵。

#### n_layers（层数）

```
n_layers = 4 (每方向):
  
  输入 → Layer0 → Layer1 → Layer2 → Layer3 → 输出
          ↓       ↓       ↓       ↓
        Mamba   Mamba   Mamba   Mamba
        Block   Block   Block   Block
        (256→256) (256→256) (256→256) (256→256)
        
  总层数 = 4 (forward) + 4 (backward) = 8 层
```

#### 三维对比（含义 + 类比 + 默认值）

| 维度 | d_model | d_state | n_layers |
|------|---------|---------|----------|
| **本质** | 每个 token 的**向量长度** | SSM 的**记忆容量** | Mamba Block **层数** |
| **类比** | 图像通道数（RGB=3） | 记忆细胞数量 | 网络深度 |
| **NLP 类比** | BERT d_model=768 | — | BERT 12 层 |
| **默认值** | 256 | 64 | 4（每方向）|
| **形状** | 标量 | 标量 | 标量 |
| **影响** | 模型"看多细" | 模型"记多少" | 模型"抽象多深" |
| **是否可调** | 是 | 是 | 是 |
| **增加此维度** | 更丰富的特征表示 | 更长的记忆跨度 | 更强的特征抽象 |

#### 维度关系图

```
输入 token: "C"  (标量 ID)
     ↓
token_embedding: vocab_size → d_model
     ↓
每个 token 变成 d_model=256 维向量
     ↓
BiMambaEncoder: n_layers=4 组 Mamba Block 处理
     ↓
每层内部: d_model=256 → d_inner=512 → SSM 状态 d_state=64
     ↓
Pooling: (B, L, d_model) → (B, d_model)
     ↓
输出预测头
```

#### 一句话速记

```
d_model = "多宽"（每个token的向量有多长）
d_state = "多深"（SSM状态里有多少个记忆槽）
n_layers = "多深"（Mamba Block堆多少层）
```

#### expand（内部维度扩展因子）

`expand` 是 Mamba 的关键设计：它将输入维度扩展到更高的内部维度，以便进行更丰富的特征变换。

```python
# bimamba.py 第 48 行
self.d_inner = int(self.expand * self.d_model)

# d_model=256, expand=2 时：
d_inner = 2 * 256 = 512
```

**为什么需要 expand？**

```
d_model=256:  输入/输出维度
d_inner=512:  内部计算维度（更大）

好处：
  - 中间层有更多参数，表达能力更强
  - 类似 Transformer 中的 FFN 扩展（d_model → d_ffn = 4*d_model）
  - 内部计算不会压缩信息
```

**数据流中的位置**：

```
输入 hidden_states: (B, L, d_model=256)
    │
    ├───→ in_proj: Linear(256, 512*2)  ← d_inner*2，分成 x 和 z 两部分
    │         ↓
    │     x, z 各 (B, L, 512)
    │         │
    │         ├───→ conv1d: Conv1d(512, 512, kernel=4)  ← 在 d_inner 上操作
    │         ├───→ x_proj: 512 → dt_rank + d_state*2  ← 选择性参数
    │         ├───→ dt_proj: 512 → dt_rank
    │         └───→ out_proj: 512 → 256  ← 输出投影回 d_model
    │
    └───→ 跳过连接（shortcut）：直接加到输出
```

**in_proj 的双重投影**：

```python
# bimamba.py 第 55-56 行
self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
# 输出: (B, L, d_inner * 2) — 分成两份

# 前向时：
x, z = hidden_states.chunk(2, dim=-1)
# x: (B, L, 512) — 用于 SSM 计算
# z: (B, L, 512) — 用于门控（gate）
```

**完整的内部维度链条**：

```
d_model=256 → in_proj → d_inner=512
                        ├─── conv1d(512) → x_conv
                        ├─── x_proj(512 → dt_rank + 2*d_state) → [dt, B, C]
                        ├─── dt_proj(512 → dt_rank) → dt
                        └─── out_proj(512 → 256) → 输出
```

**expand 的默认值为什么是 2？**

| expand | d_inner | 参数量 | 效果 |
|--------|---------|--------|------|
| 1 | 256 | 少 | 信息可能压缩过度 |
| 2 | 512 | 中等 | 平衡表达力和效率 |
| 4 | 1024 | 多 | 表达力强但慢（类似 BERT FFN） |

Mamba 选择 expand=2 是因为：
1. 足够大的中间维度避免信息瓶颈
2. 相比 Transformer 的 FFN(4×d_model)，Mamba 保持轻量
3. 与 d_conv=4, d_state=64 配合，整体计算量可控

#### expand 如何扩大维度（矩阵乘法）

维度扩大通过 **`nn.Linear`**（线性投影层）实现，本质是矩阵乘法：

```python
# bimamba.py 第 55-56 行
self.in_proj = nn.Linear(
    self.d_model,          # in_features = 256
    self.d_inner * 2,      # out_features = 512 * 2 = 1024
    bias=bias,
)
```

**矩阵运算**：

```
输入: (B, L, d_model=256)
        ↓
  output = input @ W.T + bias
  input:  (..., 256)
  W:     (1024, 256)  ← 可学习参数
  output: (..., 1024)
```

**形象理解**：

```
输入向量:  [x₀, x₁, ..., x₂₅₅]  (256维)
              ↓ 矩阵乘法 @ W.T(1024×256)
输出向量:  [y₀, y₁, ..., y₁₀₂₃]  (1024维)

W 的每一行是一个"特征提取器"：
  W[0]:  提取输入的第0个特征组合
  W[1]:  提取输入的第1个特征组合
  ...
  W[1023]: 提取输入的第1023个特征组合
```

**参数量**：

```python
# in_proj 参数量
in_proj = nn.Linear(d_model, d_inner * 2)

参数量 = d_model × (d_inner × 2) + bias
       = 256 × 1024 + 1024
       = 262,144 + 1,024
       ≈ 263K 参数
```

#### 所有可学习参数一览

> Mamba 中**所有权重都是通过反向传播训练**的。`nn.Linear` 的权重 W 初始化为随机值，前向传播时计算 `output = input @ W.T + bias`，然后通过 `loss.backward()` 计算梯度，`optimizer.step()` 更新 W。

**BiMambaForPropertyPrediction 中的可训练参数**：

| 参数 | 形状 | 作用 | 是否通过反向传播更新 |
|------|------|------|------------------|
| `token_embedding.weight` | (68, 256) | token → 向量 | ✅ 是 |
| `position_embedding.weight` | (512, 256) | 位置 → 向量 | ✅ 是 |
| **每层 BiMambaBlock**（×8层） | | | |
| `in_proj.weight` | (1024, 256) | 维度扩展 | ✅ 是 |
| `in_proj.bias` | (1024,) | 偏置 | ✅ 是 |
| `conv1d.weight` | (512, 512, 4) | 因果卷积核 | ✅ 是 |
| `conv1d.bias` | (512,) | 卷积偏置 | ✅ 是 |
| `x_proj.weight` | (96, 512) | 选择性参数投影 | ✅ 是 |
| `dt_proj.weight` | (16, 512) | 时间步长投影 | ✅ 是 |
| `dt_proj.bias` | (16,) | dt 偏置 | ✅ 是 |
| `A_log` | (512, 64) | SSM 状态矩阵 | ✅ 是（HiPPO初始化+可学习）|
| `D` | (512,) | 跳连接权重 | ✅ 是 |
| `out_proj.weight` | (256, 512) | 输出投影 | ✅ 是 |
| `out_proj.bias` | (256,) | 输出偏置 | ✅ 是 |
| **BiMambaEncoder** | | | |
| `fusion_gate.weight` | (512, 512) | 双向融合门 | ✅ 是 |
| `fusion_gate.bias` | (512,) | 融合门偏置 | ✅ 是 |
| `norm.weight` | (256,) | LayerNorm | ✅ 是 |
| `norm.bias` | (256,) | LayerNorm | ✅ 是 |
| **BiMambaForPropertyPrediction** | | | |
| `classifier.weight` | (1, 256) | 分类头 | ✅ 是 |
| `classifier.bias` | (1,) | 分类偏置 | ✅ 是 |

**参数量估算**（d_model=256, n_layers=4, d_state=64, expand=2）：

```
Token Embedding:        68 × 256 ≈ 17K
Position Embedding:    512 × 256 ≈ 131K

每层 BiMambaBlock × 8层（4 forward + 4 backward）：
  in_proj:     256 × 512 × 2 ≈ 262K
  conv1d:      512 × 512 × 4 ≈ 1M
  x_proj:      512 × 96 ≈ 49K
  dt_proj:     512 × 16 ≈ 8K
  A_log:       512 × 64 ≈ 33K
  D:           512 ≈ 0.5K
  out_proj:    512 × 256 ≈ 131K

  每层小计: ≈ 1.5M
  8层合计: ≈ 12M

Fusion Gate + LayerNorm: ≈ 263K + 0.5K
Classifier: ≈ 0.3K

=========================================
总计: 约 13M 可训练参数 ← 全部通过反向传播更新！
```

#### 维度变化总览

```
输入: (B, L)                    # B=batch, L=序列长度
  ↓ token_embedding
(B, L, d_model=256)            # 每个位置变成 256 维向量
  ↓ n_layers=4 个 Mamba Block
(B, L, d_model=256)            # 维度不变，抽象程度增加
  ↓ pooling
(B, d_model=256)               # 序列维度压缩
  ↓ classifier
(B, num_labels)                # 预测结果
```

---

### 1.5 d_conv：因果卷积详解

> `d_conv=4` 是 Mamba 中因果卷积的核宽度。本节详细解释它是什么、为什么需要、以及在整体架构中的位置。

#### 1.5.1 什么是因果卷积

因果卷积（causal convolution）是一个宽度为 `d_conv` 的一维卷积，具有**因果掩码**特性：位置 `t` 的输出只依赖于 `x[0] ... x[t]`，绝不偷看未来信息。

```python
# Mamba 中的因果卷积实现（简化）
self.conv1d = nn.Conv1d(
    d_inner,      # 输入通道
    d_inner,      # 输出通道
    kernel_size=d_conv,  # = 4
    padding=d_conv - 1,   # = 3，保证输出长度不变
    groups=d_inner,       # 逐通道独立卷积（可分离）
)
```

**因果性的保证**：`padding=d_conv-1` 使得每个输出位置 `t` 的卷积窗口为 `[t, t-1, t-2, t-3]`，恰好不包含 `t+1` 及之后的 token。

```
时间步:     t=0    t=1    t=2    t=3    t=4    t=5    ...
输入 x:    x[0]   x[1]   x[2]   x[3]   x[4]   x[5]   ...

因果卷积输出 x_conv[t]（kernel_size=4）：

x_conv[0] = w0·x[0]                        ← 只能看到 x[0]
x_conv[1] = w0·x[1] + w1·x[0]              ← 只能看到 x[0], x[1]
x_conv[2] = w0·x[2] + w1·x[1] + w2·x[0]   ← 只能看到 x[0], x[1], x[2]
x_conv[3] = w0·x[3] + w1·x[2] + w2·x[1] + w3·x[0]  ← 完整窗口
x_conv[4] = w0·x[4] + w1·x[3] + w2·x[2] + w3·x[1]  ← 滑动窗口
        ↑ 不包含 x[5], x[6]...（未来）✅
```

#### 1.5.2 为什么需要因果卷积

SSM（状态空间模型）本身通过**线性递推**建模序列：

```
h_t = dA · h_{t-1} + dB · x_t
```

这种递推结构的特性：

| 依赖类型 | SSM 递推效率 | 说明 |
|---------|------------|------|
| 极短距离（1-4 个 token） | ❌ 低效 | 需要多次递推才能建立相邻 token 之间的关系 |
| 中距离（5-100 个 token） | ✅ 高效 | 线性递推直接建模 |
| 极长距离（100+ token） | ✅ 高效（HiPPO 初始化） | 状态矩阵设计保证长期记忆 |

**因果卷积填补了这个空白**——用并行的卷积操作直接捕获局部模式，弥补 SSM 在短距离依赖上的低效。

```
┌─────────────────────────────────────────────────────┐
│  Mamba 的局部 + 全局建模策略                          │
├─────────────────────────────────────────────────────┤
│  局部（1-4 token）: 因果卷积 → 直接捕获，高效 ✅       │
│  全局（5+ token）:  SSM 递推   → 线性建模，高效 ✅     │
└─────────────────────────────────────────────────────┘
```

#### 1.5.3 在数据流中的位置

```
原始输入:  x[t]  (d_model=256 维向量)
    │
    ▼
① in_proj: x → x  (d_model → d_inner=512)
    │
    ▼
② 因果卷积: x_conv = conv1d(x)  (d_inner=512, kernel_size=4)
    │         每个 token 变成自身 + 前3个 token 的加权组合
    ▼
③ x_proj: x_conv → (dt, B, C)  (选择性参数)
    │
    ▼
④ SSM 离散化 + 并行扫描
    │
    ▼
⑤ out_proj: y → y'  (d_inner → d_model)
```

#### 1.5.4 具体数值示例

以 `d_model=8, d_inner=16, d_conv=4` 为例：

```
输入（单个时间步，t=3）:
  x[3] ∈ R^8  →  x_expanded ∈ R^16（经 in_proj 后）

因果卷积计算（t=3）:
  卷积窗口: [x[3], x[2], x[1], x[0]]  ← 4 个时间步
  x_conv[3] = w0·x[3] + w1·x[2] + w2·x[1] + w3·x[0]
            ∈ R^16  ← 投影到 d_inner 维

  权重矩阵形状: (16, 16, 4)  ← output_channels × in_channels × kernel_size
  每个输出通道独立使用相同的 4 个权重

结果: x_conv[3] 包含了 t=3 及其前 3 个时间步的局部上下文
      这个向量接下来被用于生成 dt, B, C 参数
```

#### 1.5.5 d_conv=4 的设计考量

| 考量 | 说明 |
|------|------|
| **计算量小** | `d_conv=4` 意味着每个 token 只做 4 次乘法（固定成本），不会随序列长度增长 |
| **足够捕获局部模式** | 2-4 个 token 足够捕获大多数局部语法/语义模式（相邻原子关系、局部子结构） |
| **与 expand=2 配合** | d_inner=512，卷积核参数量 `512 × 512 × 4 ≈ 1M`，占整体参数量比例可接受 |
| **因果性天然满足** | 不需要额外的 mask，因为卷积核设计保证只看过去 |

#### 1.5.6 一句话总结

> **因果卷积（d_conv=4）** 是 Mamba 的"局部增强器"——它让每个 token 在进入 SSM 之前，先用卷积窗口提取前 3 个 token 的局部上下文。这弥补了 SSM 线性递推在捕获极短距离依赖时的低效，使 Mamba 能在 **O(L)** 复杂度下同时高效处理局部模式和全局依赖。

---

### 2.1 选择性状态空间模型（Selective SSM）

Mamba 的核心是 **选择性 SSM**，用以下方程描述：

```
h_t = A @ h_{t-1} + B @ x_t      ← 状态更新
y_t = C @ h_t                      ← 输出
```

**选择性体现在**：`dt`, `B`, `C` 都是输入 `x_t` 的函数，而非固定参数。

```
x_t ──[x_proj]──→ [dt_t, B_t, C_t]  ← 每个时间步有不同参数
```

这使得模型可以**根据内容选择记住或遗忘**——重要的 token（关键官能团）写入更多，垃圾 token（padding）被压制。

### 2.2 Mamba2：mamba_ssm 库的封装

`BiMambaBlock` 内部使用 `mamba_ssm.Mamba2`：

```python
self.mamba = Mamba2(
    d_model=d_model,
    d_state=d_state,
    d_conv=d_conv,
    expand=expand,
    **factory_kwargs,
)
```

**Mamba2 内部自动完成**：
- 选择性参数生成（`dt_proj`, `B_proj`, `C_proj`）
- 离散化（`dA = exp(dt*A)`, `dB = dt*B`）
- 并行扫描（硬件感知融合 kernel）
- 状态更新（`h_new = dA * h_old + dB * x`）

**用户只需调用**：

```python
output = self.mamba(hidden_states)  # (B, L, d_model) → (B, L, d_model)
```

### 2.2.1 HiPPO 矩阵初始化详解

A 矩阵的初始化是 Mamba 的关键设计。`bimamba.py` 中的实现逻辑如下：

#### 代码追踪

```python
# bimamba.py 第 88-93 行：初始化时
A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
# A = [1, 2, 3, ..., d_state]  ← 1D 向量，不是矩阵

A = A.repeat(self.d_inner, 1).contiguous()
# A: (d_inner, d_state)
# 每行 = [1, 2, 3, ..., d_state]

A_log = torch.log(A)
self.A_log = nn.Parameter(A_log)
# 存储 log(A)，训练时可学习调整
```

```python
# bimamba.py 第 201 行：前向传播时
A = -torch.exp(self.A_log)
# A = -[1, 2, 3, ..., d_state] = [-1, -2, -3, ..., -d_state]
```

#### 最终结构

A 是一个 **对角矩阵**：

```
A = diag([-1, -2, -3, ..., -d_state])
  = [[-1,  0,  0, ...,  0],
     [ 0, -2,  0, ...,  0],
     [ 0,  0, -3, ...,  0],
     ...
     [ 0,  0,  0, ..., -d_state]]
```

### 2.2.2 HiPPO vs Mamba 的对角简化

| | HiPPO 矩阵 | Mamba 的对角初始化 |
|--|-----------|-----------------|
| **结构** | 下三角矩阵 | 对角矩阵 |
| **元素** | A[i,j] = -(i+1) if i≥j | A[i,i] = -(i+1)，其余 0 |
| **复杂度** | O(N²) | O(N) |
| **记忆特性** | 完整的正交多项式投影 | 简化的独立衰减通道 |
| **并行性** | 有数据依赖，难以并行 | 完全无依赖，O(N) 并行 |

#### HiPPO 完整矩阵（N=4 示例）

```
A_HiPPO = [[-1,  0,  0,  0],    ← h_0 独立衰减
           [-2, -2,  0,  0],    ← h_1 受 h_0 影响
           [-3, -3, -3,  0],    ← h_2 受 h_0,h_1 影响
           [-4, -4, -4, -4]]    ← h_3 受 h_0,h_1,h_2 影响
```

#### Mamba 对角矩阵（N=4 示例）

```
A_Mamba = [[-1,  0,  0,  0],    ← h_0 独立衰减
           [ 0, -2,  0,  0],    ← h_1 独立衰减
           [ 0,  0, -3,  0],    ← h_2 独立衰减
           [ 0,  0,  0, -4]]    ← h_3 独立衰减
```

#### 简化理由

1. **并行计算**：HiPPO 下三角结构有数据依赖（h_2 需要 h_0,h_1 的结果），无法高效并行；对角矩阵每个状态独立计算
2. **硬件友好**：对角矩阵的离散化 `dA = exp(dt*A)` 可完全融合到硬件感知 kernel 中
3. **理论妥协**：丢失了 HiPPO"历史正交投影"的最优性，但获得了 O(N) 并行性

### 2.2.3 dA 的计算过程

```python
# A: (d_inner, d_state) 对角矩阵
A = diag([-1, -2, -3, ..., -d_state])

# dt: (batch, seqlen, d_inner) — 每个时间步、每个样本的选择性时间步长
# 离散化
dA = exp(dt * A)
  = exp(dt * diag([-1, -2, ..., -d_state]))
  = diag([exp(-dt), exp(-2dt), exp(-3dt), ..., exp(-d_state·dt)])
```

**物理含义**（d_state=4, dt=0.1）：

```
dA 的对角元素:
  exp(-0.1 × 1) = exp(-0.1) ≈ 0.905   ← h_0: 记忆最久，衰减慢
  exp(-0.1 × 2) = exp(-0.2) ≈ 0.819   ← h_1
  exp(-0.1 × 3) = exp(-0.3) ≈ 0.741   ← h_2
  exp(-0.1 × 4) = exp(-0.4) ≈ 0.670   ← h_3: 记忆最短，衰减快
```

**状态更新**（无交互的独立通道）：

```
h_new[0] = exp(-dt) × h_old[0] + dt × B[0] × x[0]    ← 通道 0 独立
h_new[1] = exp(-2dt) × h_old[1] + dt × B[1] × x[1]  ← 通道 1 独立
h_new[2] = exp(-3dt) × h_old[2] + dt × B[2] × x[2]  ← 通道 2 独立
h_new[3] = exp(-4dt) × h_old[3] + dt × B[3] × x[3]  ← 通道 3 独立
```

### 2.3 双向建模（Bi-Mamba）

单向 Mamba 只看前缀信息：

```
前向: x[0] → x[1] → x[2] → ... → x[N]
      只能利用 [x[0]..x[t-1]] 预测 x[t]
```

**Bi-Mamba** 同时看前缀和后缀：

```
前向:  x[0] → x[1] → ... → x[N]     ← 历史信息
后向:  x[N] ← x[N-1] ← ... ← x[0]   ← 未来信息

融合: output = gate_f * forward + gate_b * backward
```

### 2.4 融合门（Fusion Gate）

Bi-Mamba 的融合不是简单平均，而是**学习的动态加权**：

```python
combined = concat([forward_hidden, backward_hidden])  # (B, L, 2*d_model)
gate = sigmoid(linear(combined))                     # (B, L, 2*d_model)
gate_f, gate_b = gate.chunk(2, dim=-1)              # 各 (B, L, d_model)

fused = gate_f * forward_hidden + gate_b * backward_hidden
```

**直观**：每个位置、每个 token，模型自己决定更相信前向还是后向——重要官能团可能需要双向确认，padding 则被自动压低。

---

## 3. 代码实现

### 3.1 BiMambaBlock

```python
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, ...):
        self.mamba = Mamba2(d_model, d_state, d_conv, expand, ...)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, d_model)
        # output:       (B, L, d_model)
        return self.mamba(hidden_states)
```

**内部发生了什么**（由 Mamba2 封装）：

```python
# 以下全部由 Mamba2 自动完成（不在本文件中）

# ① 输入投影 + 因果卷积（捕获局部依赖）
x = linear(hidden_states)          # (B, L, d_inner)
x = causal_conv1d(x)              # (B, L, d_inner)

# ② 选择性参数生成
dt, B, C = x_proj(x)              # 各 (B, L, d_inner) 或 (B, L, d_state)

# ③ 离散化
dA = exp(dt * A)                  # A 由 HiPPO 对角初始化，详见 §2.2.1-2.2.3
dB = dt * B

# ④ 并行扫描计算所有时间步
# y[0], y[1], ..., y[L-1] = scan(dA, dB, C, x)
# O(L) 并行，深度 O(log L)

# ⑤ 门控 + 输出投影
y = y * silu(z)
output = out_proj(y)              # (B, L, d_model)
```

### 3.2 BiMambaEncoder

#### 初始化

```python
def __init__(self, vocab_size, d_model=256, n_layers=4, ...):
    # Embedding 层
    self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
    self.position_embedding = nn.Embedding(max_seq_length, d_model)
    
    # Forward / Backward 独立的两组 Mamba 层
    self.forward_layers  = self._make_layers(...)  # L× BiMambaBlock
    self.backward_layers = self._make_layers(...)  # L× BiMambaBlock
    
    # 融合
    self.fusion_gate = nn.Linear(d_model * 2, d_model * 2)
    self.norm = nn.LayerNorm(d_model)
```

**关键设计**：forward 和 backward **权重不共享**（vs `bimamba.py` 的权重共享）。

#### 前向传播

```python
def forward(self, input_ids, attention_mask=None):
    # ① Embedding
    token_embeds = self.token_embedding(input_ids)           # (B, L, d_model)
    position_ids  = arange(L).unsqueeze(0).expand(B, -1)
    pos_embeds    = self.position_embedding(position_ids)  # (B, L, d_model)
    hidden_states = token_embeds + pos_embeds               # (B, L, d_model)
    
    # Mask padding
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    
    # ② Forward pass
    forward_hidden = hidden_states
    for layer in self.forward_layers:
        forward_hidden = layer(forward_hidden)             # L 步串行
    
    # ③ Backward pass (flip → Mamba → flip)
    backward_hidden = torch.flip(hidden_states, dims=[1])   # (B, L, d_model)
    for layer in self.backward_layers:
        backward_hidden = layer(backward_hidden)
    backward_hidden = torch.flip(backward_hidden, dims=[1]) # (B, L, d_model)
    
    # ④ Gate fusion
    combined = concat([forward_hidden, backward_hidden], dim=-1)  # (B, L, 2*d_model)
    gate = torch.sigmoid(self.fusion_gate(combined))                # (B, L, 2*d_model)
    gate_f, gate_b = gate.chunk(2, dim=-1)                          # 各 (B, L, d_model)
    fused = gate_f * forward_hidden + gate_b * backward_hidden      # (B, L, d_model)
    
    return self.norm(fused)                                        # (B, L, d_model)
```

**维度追踪**（假设 B=2, L=16, d_model=256）：

```
input_ids:                (2, 16)
token_embeds:             (2, 16, 256)
position_embeds:           (2, 16, 256)
hidden_states:            (2, 16, 256)
    ↓ forward_layers (×4)
forward_hidden:           (2, 16, 256)
    ↓ backward flip
backward_hidden:          (2, 16, 256)  ← 先 flip(1) → Mamba → flip(1)
    ↓ concat
combined:                 (2, 16, 512)
    ↓ fusion_gate + sigmoid
gate_f, gate_b:           各 (2, 16, 256)
    ↓ 加权融合
fused:                    (2, 16, 256)
    ↓ LayerNorm
output:                   (2, 16, 256)
```

### 3.3 BiMambaForPropertyPrediction

#### 初始化

```python
def __init__(self, vocab_size, d_model=256, n_layers=4,
             d_state=64, d_conv=4, expand=2,
             num_labels=1, task_type="regression",
             pooling="mean", dropout=0.1, ...):
    
    self.encoder = BiMambaEncoder(vocab_size, d_model, n_layers, ...)
    
    if pooling == "cls":
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    self.classifier = nn.Linear(d_model, num_labels)
    
    if task_type == "regression":
        self.loss_fct = nn.MSELoss()
    else:
        self.loss_fct = nn.BCEWithLogitsLoss()
```

#### 前向传播

```python
def forward(self, input_ids, attention_mask=None, labels=None):
    # CLS pooling: 在序列开头插入 [CLS] token
    if self.pooling == "cls":
        cls_tokens = self.cls_token.expand(B, -1, -1)
        input_ids = torch.cat([pad_token, input_ids], dim=1)
        attention_mask = torch.cat([ones(B,1), attention_mask], dim=1)
    
    # ① Encoder
    encoder_outputs = self.encoder(input_ids, attention_mask)  # (B, L, d_model)
    
    # ② Pooling → (B, d_model)
    if self.pooling == "mean":
        pooled = mean(encoder_outputs * mask.unsqueeze(-1), dim=1)
    elif self.pooling == "max":
        pooled = max(encoder_outputs * mask.unsqueeze(-1), dim=1)
    elif self.pooling == "cls":
        pooled = encoder_outputs[:, 0]  # 第一个位置即 [CLS]
    
    # ③ 分类头
    pooled = self.dropout(pooled)
    logits = self.classifier(pooled)   # (B, num_labels)
    
    # ④ 损失计算（可选）
    if labels is not None:
        loss = self.loss_fct(logits, labels)
        return logits, loss
    return logits, None
```

### 3.4 三种 Pooling 详解

**Pooling 要解决什么问题？**

```
输入：一条分子 "CC(=O)OC" → 4 个 token 的向量序列
      每个向量是 d_model=256 维

Pooling 前：Shape (B=1, L=4, d_model=256)
      [vec(C), vec(C), vec(=O), vec(O), vec(C)]
      ← 4 个向量，每个 256 维

Pooling 后：Shape (B=1, d_model=256)
      [1个向量, 256维]

目标：把 4 个向量 → 1 个向量（用于预测分子属性）
```

---

#### ① Mean Pooling（平均池化）

**直觉**：把所有 token 的信息**加起来再除以数量**，得到一个"平均分子"。

```
分子 "CC(=O)OC" 的 4 个 token：
  vec(C)   = [1, 0, 2, ...]  ← 碳原子特征
  vec(C)   = [1, 0, 1, ...]  ← 碳原子特征
  vec(=O)  = [0, 3, 0, ...]  ← 羰基特征（氧的信号）
  vec(O)   = [0, 2, 0, ...]  ← 氧原子特征
  vec(C)   = [1, 0, 1, ...]  ← 碳原子特征

Mean Pooling = 平均：
  mean_vec = (vec(C) + vec(C) + vec(=O) + vec(O) + vec(C)) / 5
           = [0.6, 1.0, 0.8, ...]  ← 分子整体特征
```

**代码**：

```python
# encoder_outputs: (B=1, L=5, d_model=256)
# attention_mask:  (B=1, L=5) — 哪些位置是有效 token（1=有效，0=padding）

# Step 1：乘以 mask（排除 padding）
mask = attention_mask.unsqueeze(-1)              # (1, 5, 1)
weighted = encoder_outputs * mask                # padding 位置变成 0

# Step 2：求和
sum_vec = sum(weighted, dim=1)                   # (1, 256)

# Step 3：除以有效数量
count = sum(mask, dim=1).clamp(min=1e-9)        # (1, 1)，防除零
pooled = sum_vec / count                         # (1, 256)
```

**适用场景**：通用首选。适合需要"整体感知"的任务——分子属性是整体特征的平均。

---

#### ② Max Pooling（最大池化）

**直觉**：在每个维度上，只保留**最显著**的那个 token 的值。

```
假设 d_model=4（简化），5 个 token 的向量：
  vec(C)   = [1, 0, 2, 1]
  vec(C)   = [1, 0, 1, 1]
  vec(=O)  = [0, 3, 0, 0]
  vec(O)   = [0, 2, 0, 0]
  vec(C)   = [1, 0, 1, 1]

Max Pooling = 每个维度取最大值：
  max_vec = [max(1,1,0,0,1),   ← 维度0：碳原子贡献最大
             max(0,0,3,2,0),   ← 维度1：羰基氧贡献最大（3）
             max(2,1,0,0,1),   ← 维度2：碳原子贡献最大（2）
             max(1,1,0,0,1)]   ← 维度3：碳原子贡献最大（1）
           = [1, 3, 2, 1]
```

**代码**：

```python
# Step 1：padding 位置设成极小值（确保不会被选中）
masked = encoder_outputs.clone()
masked[attention_mask == 0] = -1e9              # padding → -∞

# Step 2：每个维度取最大值
pooled, _ = max(masked, dim=1)                  # (B, d_model)
```

**适用场景**：适合"提取最显著特征"——比如检测分子中是否有某个强官能团。不会平均掉弱信号。

---

#### ③ CLS Pooling（CLS token 池化）

**直觉**：模型专门用一个 `[CLS]` token 来"总结"整个序列，就像给模型一个"发言位"。

```
分子 "CC(=O)OC" → 插入 [CLS] token：
  [[CLS], C, C, (, =, O, ), O, C]
   ← 模型在 [CLS] 位置"总结"整个分子

训练时：[CLS] token 的向量被要求学习"分子整体属性"
推理时：直接取 [CLS] 位置的向量作为分子表示
```

**代码**：

```python
# encoder_outputs: (B, L, d_model)
# CLS token 索引 = 0（固定在序列开头）

pooled = encoder_outputs[:, 0, :]               # (B, d_model)
# 直接取第一个位置！
```

**前提条件**：需要用 `[CLS]` 预训练或在训练时明确监督它学习分子属性。

---

#### 一句话对比

| 方式 | 核心思想 | 类比 |
|------|---------|------|
| **Mean** | 所有 token 求平均 | "民主投票"——每个人的意见都等权重 |
| **Max** | 每个维度取最强者 | "精英治国"——只看最突出的那个 |
| **CLS** | 专用 token 负责总结 | "班长发言"——让一个人代表全班 |

**默认推荐**：先试 `mean`（最稳定）。如果任务侧重"有没有某个强特征"，试 `max`。如果有预训练的 `[CLS]` 向量，试 `cls`。

---

### 3.4.1 Mask 是什么？

**为什么需要 Mask？**

深度学习处理序列时，同一个 batch 里的序列**长度不一样**：

```
batch 内三条 SMILES：
样本0: "CC(=O)OC"     → 5个 token
样本1: "C"            → 1个 token  
样本2: "CCC"          → 3个 token
```

**GPU 需要矩形张量**：batch 里的每个样本必须是相同形状。

→ 需要把短序列**填充**（padding）到一样长：

```
统一 padding 到最大长度 5：

样本0: [C, C, (, =, O, ), O, C] → [5, 12, 3, 7, 9]     ← 长度5，无需 padding
样本1: "C"                       → [5, 0, 0, 0, 0]     ← padding了4个
样本2: "CCC"                     → [5, 5, 5, 0, 0]     ← padding了2个
                  ↑
             token 0 通常是 <pad>
```

**问题**：padding 是无意义的 0，向量也是零向量。如果不处理，模型会学习到"padding 位置的信息也是信息"，导致性能下降。

---

**Mask 的本质**

```
mask = 一个只包含 0 和 1 的张量

  1 = 真实 token（有意义，要计算）
  0 = padding token（无意义，要忽略）
```

```
attention_mask:  (B=2, L=5)
样本0: [1, 1, 1, 1, 1]  ← 全部是真实 token
样本1: [1, 0, 0, 0, 0]  ← 只有第0位是真实的
样本2: [1, 1, 1, 0, 0]  ← 前3位是真实的
```

**Mask 怎么生成？**

```python
# tokenizer 返回的 attention_mask 就是 mask
input_ids = [5, 12, 3, 7, 9]      # 样本的 token IDs
attention_mask = [1, 1, 1, 1, 1]   # 每个位置对应 1=有效

# tokenizer 自动处理：
encoding = tokenizer("CC(=O)OC")
# encoding['input_ids']:      [5, 12, 3, 7, 9]
# encoding['attention_mask']: [1, 1, 1, 1, 1]
```

**Mask 在不同地方的作用**

| 场景 | Mask 的作用 |
|------|-----------|
| **Attention** | padding 位置不参与注意力计算 |
| **Pooling** | padding 位置不参与求平均/最大值 |
| **Loss 计算** | padding 位置不计入 loss |

**一句话总结**

> **Mask = "真实位置是 1，padding 位置是 0" 的张量**。它的作用是告诉模型"哪些位置是真的数据，哪些是填充的无意义数据"。

---

### 3.4.2 Mask 处理详解：为什么乘以 mask 能排除 padding？

**Padding 是什么？**

```
SMILES 序列长度不一，需要 padding 到统一长度：

样本0: "CC(=O)OC" → token IDs: [5, 12, 3, 7, 9]  → 长度5，有效
样本1: "CO"       → token IDs: [5, 7, 0, 0, 0]  → 长度2，padding了3个

attention_mask:  (B=2, L=5)
样本0: [1, 1, 1, 1, 1]  ← 全部有效
样本1: [1, 1, 0, 0, 0]  ← 前2有效，后3是padding
```

**为什么需要 mask？**

```
不用 mask 直接平均（样本1）：
  mean = (vec[0] + vec[1] + vec[2] + vec[3] + vec[4]) / 5
       = (v0 + v1 + 0 + 0 + 0) / 5
       = (v0 + v1) / 5  ← ❌ 错误！除以了5，但实际只有2个有效token

用 mask：
  sum_mask = 1+1+0+0+0 = 2  ← 正确计数
  mean = (v0 + v1 + 0 + 0 + 0) / 2
       = (v0 + v1) / 2  ← ✅ 正确！
```

**为什么乘以 mask(0/1) 能排除 padding？**

```
mask = attention_mask.unsqueeze(-1):  (B=2, L=5, 1)
样本1: [[1], [1], [0], [0], [0]]

广播乘法：
  encoder_outputs * mask
  (B,L,d_model) × (B,L,1) → (B,L,d_model)

每个位置：
  有效位置(1) × 原值 = 原值（不变）
  padding位置(0) × 原值 = 0（清零）

具体例子，位置 t=3 是 padding：
  原始向量 = [1.5, 2.3, -0.7, ...]
  mask[t=3] = 0
  结果 = [1.5, 2.3, -0.7, ...] × 0 = [0, 0, 0, ...]
                                    ↑
                              变成了零向量
```

**为什么 Mean Pooling 用 0 而 Max Pooling 用 -1e9？**

```
Mean Pooling：用 0 没问题
  零向量参与求和 = 无贡献 ✅

Max Pooling：用 0 会有问题！
  假设有效向量都是负数：[−5, −2, 0, 0, 0]
  max = 0  ← ❌ 选中的是 padding（0 > -5）

  正确做法：padding 位置设成 -1e9
  vec = [-5, -2, -1e9, -1e9, -1e9]
  max = -2  ← ✅ 正确选中了有效位置

代码实现：
  masked = encoder_outputs.clone()
  masked[attention_mask == 0] = -1e9  # padding → -∞
  pooled, _ = max(masked, dim=1)
```

**三句话总结**

```
Mean Pooling：乘以 mask(0/1) → padding 变 0 → 求和无贡献，计数正确
Max Pooling：设成 -1e9 → padding 变极小 → max 不会被选中
CLS Pooling：不用 mask → 直接取第 0 位（CLS token 不受 padding 影响）
```

### 3.5 工厂函数

```python
def create_bimamba_model(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 4,
    task_type: str = "regression",
    num_labels: int = 1,
    **kwargs,          # d_state, d_conv, expand, pooling, dropout...
) -> BiMambaForPropertyPrediction:
    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        task_type=task_type,
        num_labels=num_labels,
        **kwargs,
    )

# 使用示例
model = create_bimamba_model(
    vocab_size=50,
    d_model=256,
    n_layers=4,
    d_state=64,
    task_type="regression",
    num_labels=1,
    pooling="mean",
)
```

---

## 4. 完整数据流逐step详解（数值示例）

> **设定**：B=2, L=4, d_model=8, vocab_size=20, n_layers=2
> 
> **目标**：追踪一个 batch 从 input_ids 到最终 logits 的**每一个 tensor 变换**

### 4.1 Step 0：输入

```python
# input_ids: shape (B, L) = (2, 4)
input_ids = torch.tensor([
    [5, 12, 3, 7],    # 样本0: 4个token
    [9, 0, 0, 0],     # 样本1: 1个token + 3个padding
])
# token 0 是 <pad>

# attention_mask: shape (B, L) = (2, 4)
attention_mask = torch.tensor([
    [1, 1, 1, 1],    # 样本0: 全部有效
    [1, 0, 0, 0],    # 样本1: 只有第0个有效
])
```

### 4.2 Step 1：Token Embedding

```python
# self.token_embedding: nn.Embedding(20, 8)
# vocab_size=20, d_model=8
# 权重形状: (20, 8)

token_embeds = self.token_embedding(input_ids)
# input_ids: (2, 4)
# B = batch size：同时处理多少条序列（2条）
# L = sequence length：每条序列有多少个 token（4个）
# token_embeds: (2, 4, 8) = (B, L, d_model)

# 具体数值示例（随机初始化后的典型值）：
token_embeds[0, 0] = [0.12, -0.34, 0.56, -0.78, 0.91, 0.23, -0.45, 0.67]  # token=5
token_embeds[0, 1] = [-0.11, 0.22, -0.33, 0.44, -0.55, 0.66, -0.77, 0.88] # token=12
token_embeds[0, 2] = [0.19, -0.28, 0.37, -0.46, 0.55, -0.64, 0.73, -0.82] # token=3
token_embeds[0, 3] = [-0.17, 0.26, -0.35, 0.44, -0.53, 0.62, -0.71, 0.80] # token=7
token_embeds[1, 0] = [0.13, -0.24, 0.35, -0.46, 0.57, -0.68, 0.79, -0.80] # token=9
token_embeds[1, 1] = [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]    # <pad>
token_embeds[1, 2] = [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]    # <pad>
token_embeds[1, 3] = [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]    # <pad>
```

### 4.3 Step 2：Position Embedding

> **核心问题**：Token Embedding 只编码"内容"（token 是什么），不编码"位置"（在序列第几位）。Position Embedding 填填补这个空白。

#### 为什么需要位置编码

```
Token Embedding 视角:
  "C-C-O" 和 "O-C-C" → token embeddings 完全相同 ❌

现实世界:
  "C-C-O" (乙醇，可饮用)  vs  "O-C-C" (甲醛，有毒！)
  原子相同，顺序不同 → 性质完全不同
```

#### 位置编码的实现

```python
self.position_embedding = nn.Embedding(max_seq_length=512, d_model=8)
```

这是一个**可学习的查找表**：

```
索引 → 向量
  0    →  v₀ ∈ R^8   （位置 0 的编码）
  1    →  v₁ ∈ R^8   （位置 1 的编码）
  2    →  v₂ ∈ R^8   （位置 2 的编码）
  ...
  511  →  v511 ∈ R^8 （位置 511 的编码）

总参数量: 512 × 8 = 4,096 个可学习参数
```

#### 数据流

```python
# 生成位置索引
position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
# L=4, B=2 → position_ids = [[0, 1, 2, 3], [0, 1, 2, 3]]

# 查表获取位置编码
pos_embeds = self.position_embedding(position_ids)
# 输出形状: (B, L, d_model) = (2, 4, 8)

# 具体数值示例（随机初始化后的典型值）：
pos_embeds[0, 0] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]  # pos=0
pos_embeds[0, 1] = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]  # pos=1
pos_embeds[0, 2] = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24]  # pos=2
pos_embeds[0, 3] = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32]  # pos=3
# 样本1的位置编码与样本0相同（共享，同一序列的不同样本位置相同）
```

#### 与 Token Embedding 的结合

```python
# Step 1: Token Embedding（内容）
token_embeds = self.token_embedding(input_ids)
# token_embeds: (B, L, d_model) = (2, 4, 8)

# Step 2: Position Embedding（位置）
position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
pos_embeds = self.position_embedding(position_ids)
# pos_embeds: (B, L, d_model) = (2, 4, 8)

# Step 3: 相加（内容 + 位置）
hidden_states = token_embeds + pos_embeds
# hidden_states: (B, L, d_model) = (2, 4, 8)
# 每个位置 = 该位置的 token 内容 + 该位置的编码
```

#### 具体数值示例

```
位置 0 的 token "C"：
  token_embedding: [1.2, 0.5, -0.3, 0.8, ...]  ← 碳原子的语义
  position_embedding: [0.01, 0.02, 0.03, 0.04, ...]  ← 位置 0
  相加结果: [1.21, 0.52, -0.27, 0.84, ...]  ← 位置感知的 token 表示

位置 3 的 token "C"：
  token_embedding: [1.2, 0.5, -0.3, 0.8, ...]  ← 碳原子（和位置 0 相同！）
  position_embedding: [0.04, 0.08, 0.12, 0.16, ...]  ← 位置 3
  相加结果: [1.24, 0.58, -0.18, 0.96, ...]  ← 不同的位置感知表示
```

#### Mamba 理论上需要位置编码吗？

| 架构 | 天然处理位置？ | 需要 Position Embedding？ |
|------|--------------|------------------------|
| **RNN** | ✅ 递推结构天然有序 | ❌ 不需要（隐式位置） |
| **Transformer** | ❌ Attention 是位置无关的 | ✅ 必须加（绝对或相对） |
| **Mamba SSM** | 理论上递推 = 隐式位置 | ⚠️ 可选，但实践中有帮助 |

**为什么 Mamba 实践中也用 Position Embedding？**

```
Mamba 的选择性机制: h_t = dA·h_{t-1} + dB·x_t

理论上：dA 和 dB 是位置相关的（dt 是输入依赖的），
       所以位置信息已经被编码到 dt 中

但实践发现：添加显式位置编码仍然有帮助
  → 让模型更容易区分"第 0 个 token"和"第 100 个 token"
  → 补充 SSM 选择性机制的不足
```

#### 一句话总结

> **Position Embedding** 是可学习的位置查找表（512 × d_model），将位置索引映射为 d_model 维向量，与 Token Embedding 相加后，让模型同时感知**内容**（token 是什么）和**位置**（在序列第几位）。

#### 附录：position_embedding.weight 详解

> **"是 tokens 向量根据位置查表获取权重矩阵，加到 token 上吗？"** — 这个问题值得单独澄清。

**不是"根据 token 查位置权重"，而是两个独立的查表。**

##### 两个独立的查找表

```
┌─────────────────────────────────────────────────────────┐
│  Token Embedding: token id → token 向量（语义）          │
│                                                         │
│  token_embedding.weight（vocab_size × d_model）        │
│  token "C" 的 id = 1                                    │
│    → token_embedding.weight[1]                         │
│    → [1.2, 0.5, -0.3, 0.8, ...]  ← 查表得到 "C" 的语义  │
└─────────────────────────────────────────────────────────┘
                        +
┌─────────────────────────────────────────────────────────┐
│  Position Embedding: position id → position 向量（位置）  │
│                                                         │
│  position_embedding.weight（max_seq_length × d_model） │
│  position = 3                                          │
│    → position_embedding.weight[3]                      │
│    → [0.04, 0.08, 0.12, 0.16, ...]  ← 查表得到位置 3   │
└─────────────────────────────────────────────────────────┘
                        ↓
            逐元素相加（element-wise add）
                        ↓
        result = [1.24, 0.58, -0.18, 0.96, ...]
```

##### 权重矩阵结构

```python
self.position_embedding = nn.Embedding(max_seq_length=512, d_model=8)

# position_embedding.weight 的形状: (512, 8)
# 即 max_seq_length 行，每行 d_model=8 维向量

# 第 i 行 = 位置 i 的位置编码向量
position_embedding.weight[0]  # 位置 0 的向量
position_embedding.weight[1]  # 位置 1 的向量
position_embedding.weight[511] # 位置 511 的向量
```

##### 形象比喻

```
Token Embedding = 字典
  "C" → 查 "C" 的解释 → [1.2, 0.5, ...]（语义）

Position Embedding = 字典
  位置 3 → 查 "第3位" 的解释 → [0.04, 0.08, ...]（位置）

相加 = 语义 + 位置 = "C在第3位" 的完整表示
```

##### 关键点

| 操作 | 说明 |
|------|------|
| **查表** | `weight[索引]` 是离散查表，不是矩阵乘法 |
| **相加** | 不是乘，是逐元素加（element-wise add） |
| **同一个 token** | "C" 在位置 0 和位置 3 的最终表示不同——因为加的位置向量不同 |

##### 为什么用加法而不是拼接

```
方案 A（拼接）: concat([token_vec, pos_vec]) → 2×d_model 维
方案 B（相加）: token_vec + pos_vec           → d_model 维

Mamba 用方案 B（相加）的原因：
  - 维度不变，保持 d_model
  - 训练时 token 和 position 的信息自然融合
  - 实践效果和拼接相当，但更节省维度
```

### 4.4 Step 3：相加 + Padding Mask

```python
hidden_states = token_embeds + pos_embeds
# hidden_states: (2, 4, 8)

# 应用 attention_mask（将 padding 位置的向量置零）
# attention_mask.unsqueeze(-1): (2, 4, 1)
# mask[:, :, None]: (2, 4, 1)
hidden_states = hidden_states * attention_mask.unsqueeze(-1)
# padding 位置的向量被乘以 0

# 结果：
# 样本0: 4个有效位置（非零向量）
# 样本1: 只有位置0有效，位置1-3变为 [0,0,0,0,0,0,0,0]
```

### 4.5 Step 4：Forward 层 — Layer 0

**输入**：`hidden_states` (2, 4, 8)

**Mamba2 内部计算**（展开）：

```python
# ① 输入投影 → d_inner = expand * d_model = 2 * 8 = 16
x = self.input_proj(hidden_states)
# x: (2, 4, 16)

# ② 因果卷积（d_conv=4）
# causal_conv1d 保证位置 t 只依赖 t, t-1, t-2, t-3
x_conv = causal_conv1d(x)
# x_conv: (2, 4, 16)

# ③ 选择性参数投影（x_proj）
# dt_proj: d_inner → dt_rank（通常 d_model/16 = 8/16 = 0.5 → 实际取整后）
# B_proj: d_inner → d_state
# C_proj: d_inner → d_state
dt, B, C = self.x_proj(x_conv)
# dt: (2, 4, dt_rank)   — 时间步长缩放因子
# B:  (2, 4, d_state)   — 输入到状态的映射
# C:  (2, 4, d_state)   — 状态到输出的映射

# ④ 离散化
# A: (d_state, d_state) = (4, 4)，HiPPO 初始化，形状如：
# A = [[-1,  0,  0,  0],
#      [-2, -2,  0,  0],
#      [-3, -3, -3,  0],
#      [-4, -4, -4, -4]]

# dA = exp(dt * A): (2, 4, 4, 4) — 每个时间步、每个样本有独立的 A
# dB = dt * B:      (2, 4, 4)   — 每个时间步有独立的 B

# ⑤ 并行扫描（硬件感知）
# y = SSM(dA, dB, C, x)
# y: (2, 4, 4) — (B, L, d_state)

# ⑥ 门控 + 输出投影
y = y * silu(z)  # z 来自 x_conv 的门控分支
output = self.out_proj(y)
# output: (2, 4, 8) = (B, L, d_model)
```

**Layer 0 输出**：`forward_hidden` (2, 4, 8)

### 4.6 Step 5：Forward 层 — Layer 1

**输入**：`forward_hidden` from Layer 0 (2, 4, 8)

**重复 Step 4 的计算过程**（Layer 1 有自己独立的权重）：

**Layer 1 输出**：`forward_hidden` (2, 4, 8)

### 4.7 Step 6：Backward 层 — Layer 0

**关键**：Backward 层在**Flip 后的序列**上运行 Mamba

```python
# ① Flip：翻转序列维度
backward_hidden = torch.flip(hidden_states, dims=[1])
# hidden_states: (2, 4, 8)
# Flip 后：位置 0↔3, 1↔2 交换
# backward_hidden[0]: [hidden[0,3], hidden[0,2], hidden[0,1], hidden[0,0]]
# backward_hidden[1]: [hidden[1,3], hidden[1,2], hidden[1,1], hidden[1,0]]

# ② 在 flip 后的序列上运行 Mamba（Layer 0）
# 这是"后向扫描"的核心：token t 看到的是 token t+1, t+2, ... 的信息
backward_hidden = backward_layers[0](backward_hidden)
# backward_hidden: (2, 4, 8)
```

### 4.8 Step 7：Backward 层 — Layer 1

```python
backward_hidden = backward_layers[1](backward_hidden)
# backward_hidden: (2, 4, 8)
```

### 4.9 Step 8：Backward 层 — Flip 还原

```python
backward_hidden = torch.flip(backward_hidden, dims=[1])
# 再次 flip，将序列恢复到原始位置顺序
# backward_hidden: (2, 4, 8)
# 此时 backward_hidden[t] 包含原始位置 t 的"未来信息"
```

### 4.10 Step 9：Gate Fusion

**输入**：
- `forward_hidden`: (2, 4, 8) — 包含"历史信息"
- `backward_hidden`: (2, 4, 8) — 包含"未来信息"

```python
# ① 拼接
combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
# combined: (2, 4, 16) = (B, L, 2*d_model)

# ② 门控投影
gate = self.fusion_gate(combined)
# fusion_gate: Linear(16, 16)
# gate: (2, 4, 16)

# ③ Sigmoid 激活
gate = torch.sigmoid(gate)
# gate: (2, 4, 16) — 每个元素在 (0, 1) 区间

# ④ 拆分为两个门
gate_f, gate_b = gate.chunk(2, dim=-1)
# gate_f: (2, 4, 8) — 前向门控
# gate_b: (2, 4, 8) — 后向门控

# 数值示例（某个位置）：
# gate_f[0, 2] = [0.7, 0.8, 0.6, 0.9, 0.5, 0.7, 0.8, 0.6]
# gate_b[0, 2] = [0.3, 0.2, 0.4, 0.1, 0.5, 0.3, 0.2, 0.4]
# 说明模型在该位置更信任前向信息（gate_f 更大）

# ⑤ 加权融合
fused = gate_f * forward_hidden + gate_b * backward_hidden
# fused: (2, 4, 8)

# ⑥ LayerNorm
fused = self.norm(fused)
# fused: (2, 4, 8) — encoder 最终输出
```

### 4.11 Step 10：Pooling

**输入**：`fused` (2, 4, 8)，attention_mask (2, 4)

#### Mean Pooling（假设 pooling="mean"）

```python
# ① Mask：排除 padding 的贡献
mask = attention_mask.unsqueeze(-1)  # (2, 4, 1)
masked_fused = fused * mask          # (2, 4, 8)
# padding 位置被置零

# ② 求和
sum_embeds = masked_fused.sum(dim=1)  # (2, 8)
# sum_embeds[0] = fused[0,0] + fused[0,1] + fused[0,2] + fused[0,3]
# sum_embeds[1] = fused[1,0] + 0 + 0 + 0  (只有位置0有效)

# ③ 求 mask 总和（防除零）
mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (2, 1)
# mask_sum[0] = 1+1+1+1 = 4
# mask_sum[1] = 1

# ④ 平均
pooled = sum_embeds / mask_sum  # (2, 8)
# pooled[0] = sum_embeds[0] / 4  ← 4个位置的平均
# pooled[1] = sum_embeds[1] / 1  ← 只有1个位置

# 最终 pooled: (2, 8) = (B, d_model)
```

#### Max Pooling（假设 pooling="max"）

```python
# ① 将 padding 位置设为极小值
masked = fused.clone()
masked[attention_mask == 0] = -1e9
# 这样 max 操作不会选到 padding 位置

# ② 逐维度取最大值
pooled = masked.max(dim=1).values  # (2, 8)
# pooled[0] = max over dim=1 (4 个有效位置的向量)
# pooled[1] = fused[1, 0]  (只有位置0有效)
```

#### CLS Pooling（假设 pooling="cls"）

```python
# 在序列开头插入 [CLS] token（训练时）
# 推理时直接取位置 0
pooled = fused[:, 0]  # (2, 8) — 直接取第一个位置的向量
```

### 4.12 Step 11：Dropout + Classifier

```python
# ① Dropout
pooled = self.dropout(pooled)
# pooled: (2, 8) — 训练时随机置零，推理时不生效

# ② 分类头
logits = self.classifier(pooled)
# classifier: Linear(8, 1) — d_model → num_labels
# logits: (2, 1) = (B, num_labels)
```

### 4.13 Step 12：损失计算（训练时）

```python
# 假设 task_type="regression"
# labels: (2, 1) = (B, num_labels) — 真实标签

loss = self.loss_fct(logits, labels)
# loss_fct = nn.MSELoss()
# loss = mean((logits - labels)^2)
```

---

## 5. 完整前向传播示例

**输入**：

```python
B, L, vocab_size = 2, 16, 50
input_ids = torch.randint(0, vocab_size, (B, L))   # (2, 16)
attention_mask = torch.ones(B, L)                  # 全有效
attention_mask[:, -2:] = 0                          # 最后2个是 padding
```

**前向**：

```python
# ① Embedding
token_embeds = token_embedding(input_ids)           # (2, 16, 256)
pos_embeds   = position_embedding(position_ids)      # (2, 16, 256)
hidden = token_embeds + pos_embeds                  # (2, 16, 256)
hidden = hidden * mask.unsqueeze(-1)                # (2, 16, 256) padding 清零

# ② Forward × 4 层
f0 = forward_layers[0](hidden)                      # (2, 16, 256)
f1 = forward_layers[1](f0)                          # (2, 16, 256)
f2 = forward_layers[2](f1)                          # (2, 16, 256)
f3 = forward_layers[3](f2)                          # (2, 16, 256)

# ③ Backward × 4 层（flip → Mamba → flip）
b = torch.flip(hidden, dims=[1])                   # (2, 16, 256)
b = backward_layers[0](b)                          # (2, 16, 256)
b = backward_layers[1](b)                          # (2, 16, 256)
b = backward_layers[2](b)                          # (2, 16, 256)
b = backward_layers[3](b)                          # (2, 16, 256)
b = torch.flip(b, dims=[1])                        # (2, 16, 256)

# ④ Gate fusion
combined = cat([f3, b], dim=-1)                    # (2, 16, 512)
gate = sigmoid(fusion_gate(combined))              # (2, 16, 512)
gf, gb = gate.chunk(2, dim=-1)                    # 各 (2, 16, 256)
fused = gf * f3 + gb * b                          # (2, 16, 256)
output = norm(fused)                              # (2, 16, 256)

# ⑤ Mean pooling（排除 padding）
mask_sum = mask.sum(dim=1, keepdim=True)          # (2, 1)
pooled = (fused * mask.unsqueeze(-1)).sum(dim=1) / mask_sum  # (2, 256)

# ⑥ 分类
logits = classifier(dropout(pooled))                # (2, 1) 或 (2, num_labels)
```

**输出**：

```python
logits:    (2, 1)       ← 回归任务
logits:    (2, num_labels) ← 分类任务
```

---

## 6. 模块依赖关系

```
BiMambaForPropertyPrediction
├── BiMambaEncoder
│   ├── nn.Embedding (token)
│   ├── nn.Embedding (position)
│   ├── nn.Dropout
│   ├── nn.ModuleList[BiMambaBlock × n_layers] (forward)
│   ├── nn.ModuleList[BiMambaBlock × n_layers] (backward)
│   ├── nn.Linear (fusion_gate)
│   └── nn.LayerNorm
├── nn.Dropout
├── nn.Linear (classifier)
└── nn.MSELoss / nn.BCEWithLogitsLoss (loss_fct)

BiMambaBlock
└── mamba_ssm.Mamba2
    ├── x_proj (选择性参数)
    ├── conv1d (因果卷积)
    ├── MambaCore (SSM 离散化 + 并行扫描)
    └── out_proj (输出投影)
```

---

## 7. 与 bimamba.py 的差异

| 特性 | `bimamba.py`（手动实现） | `bimamba_with_mamba_ssm.py`（本文件） |
|------|------------------------|--------------------------------------|
| SSM 实现 | 手动实现离散化 + 并行扫描 | 使用 `mamba_ssm.Mamba2` 库封装 |
| 前向/后向权重 | **共享**（节省参数） | **独立**（各自有完整参数） |
| 融合方式 | concat / add / **gate** 三种 | 仅 **gate** 一种 |
| 默认 d_state | 16 | 64（更大记忆容量） |
| 适用场景 | 研究/调试（透明但慢） | 生产/部署（高效但黑盒） |

### 7.1 权重共享 vs 独立

**bimamba.py（共享）**：

```
Forward Layer 0 ↔ Backward Layer 0  ← 同一组权重
Forward Layer 1 ↔ Backward Layer 1  ← 同一组权重
...
参数数量: L 层 × 1 套
```

**本文件（独立）**：

```
Forward Layer 0, Forward Layer 1, ... Forward Layer L-1
Backward Layer 0, Backward Layer 1, ... Backward Layer L-1
参数数量: 2L 层 × 各自独立
```

**权衡**：独立权重表达能力更强（参数多一倍），共享权重更省内存。

### 7.2 融合模式

**bimamba.py 支持三种**：

```python
if fusion == "concat":
    combined = concat([fwd, bwd])
elif fusion == "add":
    combined = fwd + bwd
elif fusion == "gate":
    gate = sigmoid(linear(fwd))
    combined = gate * fwd + (1 - gate) * bwd
```

**本文件仅 gate**：

```python
gate_f, gate_b = sigmoid(fusion_gate(concat([fwd, bwd]))).chunk(2, dim=-1)
combined = gate_f * fwd + gate_b * bwd  # 注意 gate_b ≠ (1 - gate_f)
```

**细微差异**：`bimamba.py` 的 gate 是 `gate_f = sigmoid(W_fwd)`，`gate_b = 1 - gate_f`（互斥）；本文件两个 gate 独立学习，更灵活。

---

## 8. 设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 使用 Mamba2 库 | mamba_ssm.Mamba2 | 硬件感知融合 kernel，比手动实现快 5-10 倍 |
| 双向独立权重 | forward/backward 分离 | 各自有完整表达能力 |
| 融合用 Gate | sigmoid 门控 | 动态学习双向加权，比 add/concat 更灵活 |
| d_state=64 | 默认 64 | 比 bimamba.py 的 16 大 4 倍，记忆容量更强 |
| 支持 CLS pooling | 额外插入 [CLS] token | 与 BERT 风格兼容，方便微调 |
| 损失可选 | `labels=None` 时不计算损失 | 支持推理（不需要 labels）和训练两种用法 |

---

## 9. 附录：参数数量估算

对于 **d_model=256, n_layers=4, d_state=64, d_conv=4, expand=2**：

```
Token Embedding: vocab_size × d_model = 68 × 256 ≈ 17K 参数
Position Embedding: max_seq_length × d_model = 512 × 256 ≈ 131K 参数

每层 BiMambaBlock (Mamba2):
  - x_proj (dt, B, C): d_inner × (dt_rank + 2 × d_state) ≈ 512 × 160 ≈ 82K
  - conv1d: d_inner × d_conv = 512 × 4 ≈ 2K
  - out_proj: d_inner × d_model = 512 × 256 ≈ 131K
  - A (HiPPO): d_state × d_state = 64 × 64 = 4K
  - D (跳接): d_model = 256

总每层: ≈ 220K 参数

Forward × 4: 4 × 220K ≈ 880K
Backward × 4: 4 × 220K ≈ 880K（独立权重）

Fusion Gate: Linear(512, 512) ≈ 262K
LayerNorm: 2 × 256 ≈ 512
Classifier: Linear(256, 1) ≈ 257

总计: ≈ 2.3M 参数
```

---

## 10. 参考

- **Mamba 论文**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- **mamba_ssm 库**: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- **Mamba 教程**: [`mamba.tutorial.md`](../../mamba.tutorial.md) — 从零理解 Mamba SSM 的完整指南
