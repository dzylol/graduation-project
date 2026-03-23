# Mamba SSM 结构与实现详解

> 本教程基于 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023) 及 `mamba_ssm` 库实现。
> 代码参考: `src/models/bimamba_with_mamba_ssm.py`

---

## 目录

1. [背景：为什么需要 Mamba？](#1-背景为什么需要-mamba)
2. [状态空间模型 (SSM) 基础](#2-状态空间模型-ssm-基础)
3. [连续到离散：离散化过程](#3-连续到离散离散化过程)
4. [Mamba 的核心创新：选择性机制](#4-mamba-的核心创新选择性机制)
5. [并行扫描算法 (Parallel Scan)](#5-并行扫描算法-parallel-scan)
6. [HiPPO 矩阵初始化](#6-hippo-矩阵初始化)
7. [完整前向传播流程](#7-完整前向传播流程)
8. [双向 Mamba (Bi-Mamba)](#8-双向-mamba-bi-mamba)
9. [代码实现逐行解析](#9-代码实现逐行解析)
10. [Mamba vs Transformer vs RNN](#10-mamba-vs-transformer-vs-rnn)

---

## 1. 背景：为什么需要 Mamba？

### Transformer 的困境

Transformer 的核心是 **Self-Attention**，其时间复杂度为 **O(N²)**（N 为序列长度）：

```
Attention: O(N²)  — 序列越长，计算量爆炸性增长
```

对于长序列（如基因组、音乐、长文本），Transformer 面临：
- **显存瓶颈**：注意力矩阵 `N × N` 巨大
- **延迟高**：无法实时流式输出

### 替代方案的失败

过去出现过多种**次二次方 (subquadratic)** 架构：

| 架构 | 问题 |
|------|------|
| Linear Attention | 无法做**内容-based 推理**（选择性遗忘/记住） |
| Gated Convolution | 感受野固定，无法动态调整 |
| RNN | 只能 O(N) 串行，无法并行 |

**核心弱点**：这些模型无法根据输入内容动态决定保留或丢弃哪些信息。

---

## 2. 状态空间模型 (SSM) 基础

### 2.1 什么是 SSM？

SSM 源自控制理论，描述**连续系统**如何响应输入信号：

```
连续微分方程:
    dh/dt = A · h + B · x      (状态更新)
    y     = C · h               (输出)
```

其中：
- `x(t)` — d 维输入信号（标量 t 的函数）
- `h(t)` — N 维隐藏状态（系统的"记忆"）
- `y(t)` — d 维输出信号
- `A` — N×N 状态转移矩阵
- `B` — N×d 输入矩阵
- `C` — d×N 输出矩阵

### 2.2 SSM 的直观理解

可以把 SSM 想象成一个**线性动态系统**：

```
x(t) 输入  →  [B]  →  叠加到状态 h
                     ↓
               h 变化由 A 控制（衰减/增长）
                     ↓
               h 通过 [C] 产生输出 y
```

类比 RLDDT 的残差网络：每个 token 的信息被压缩到状态向量 `h` 中。

### 2.3 维度标注（代码对照）

```
x_t  : (d_inner,)      输入向量（d_inner = expand * d_model）
h_t  : (d_state,)      隐藏状态向量（通常 64 或 128）
y_t  : (d_inner,)      输出向量
A    : (d_inner, d_state)  状态转移矩阵
B    : (d_inner, d_state)  输入矩阵
C    : (d_inner, d_state)  输出矩阵
```

---

## 3. 连续到离散：离散化过程

### 3.1 为什么需要离散化？

控制理论中的 SSM 是**连续时间**微分方程。但计算机处理的是**离散序列**：

```
连续世界: dh/dt = A·h + B·x      (微分方程，求导)
离散世界: h_t = A_d·h_{t-1} + B_d·x_t  (差分方程，迭代)
```

### 3.2 连续方程的精确解

从连续微分方程出发，两边积分可得**精确解**：

```
dx/dt = A·x + B·u

两边乘以 e^(-At) 并求积分：
x(t_b) = e^(A(t_b - t_a)) · x(t_a) + e^(A·t_b) · ∫_{t_a}^{t_b} e^(-Aτ) · B·u(τ) · dτ
```

设采样间隔 Δt = t_b - t_a，可得：

```
x_{k+1} = e^(A·Δt) · x_k + ∫_{0}^{Δt} e^(A·(Δt-τ)) · B·u_k · dτ
```

### 3.3 欧拉法（Forward Euler）—— 简单近似

用差分近似导数：`dh/dt ≈ (h_t - h_{t-1}) / Δt`

```
(h_t - h_{t-1}) / Δt = A·h_{t-1} + B·x_t
h_t = (I + Δt·A)·h_{t-1} + Δt·B·x_t
```

**问题**：Forward Euler **不稳定**，要求 `|I + Δt·A| < 1`，这限制了 Δt 的取值。

#### 3.3.1 稳定性分析

**离散系统**：h_{t+1} = (I + Δt·A) · h_t

**稳定性要求**：当 t→∞ 时，h_t 不发散 → 放大因子 |I + Δt·A| < 1

**标量直观例子**（A 为实数 a）：

```
连续系统: dh/dt = a·h
若 a = -2（稳定衰减）

Forward Euler: h_t = (1 + Δt·a)·h_{t-1}

  Δt = 0.5:  1 + 0.5·(-2) = 0       → |0| < 1  ✅ 稳定
  Δt = 1.0:  1 + 1.0·(-2) = -1      → |-1| = 1  ⚠️ 临界（振荡）
  Δt = 1.1:  1 + 1.1·(-2) = -1.2    → |-1.2| > 1 ❌ 发散！
```

**根轨迹视角**（s 平面到 z 平面的映射）：

连续时间系统用 **s 平面**分析（拉普拉斯变换），离散时间系统用 **z 平面**分析（Z变换）。

```
s 平面（连续）                    z 平面（离散）
  Im                              Im
   |                               |
  虚轴 |                        单位圆
 Re<0 | 稳定区                 | |z|=1
   |_________________________  | |z|<1 稳定
   |                            |
 Re>0 不稳定区                   | |z|>1 不稳定
```

**Forward Euler 的映射关系**：`z = 1 + Δt·s`

这个映射把 s 平面的稳定区（左半平面）映射到 z 平面上的一条**射线**：

```
s 从 -∞ 沿实轴移到 0：
  s = -∞  →  z = -∞（左无穷远）
  s = -2  →  z = 1 + Δt·(-2) = 1 - 2Δt
  s = 0   →  z = 1

s平面上左半平面的一条垂线 → z平面上过 (1,0)、斜率为 Δt 的射线
```

**关键问题**：s 平面上靠近虚轴的低频稳定极点（梯度缓变的信息），被映射到 z 平面单位圆外！

```
s = -0.1（稳定，但靠近虚轴 → 变化缓慢的信息）
Δt = 10
→ z = 1 + 10·(-0.1) = 0        |z| < 1 ✅ 稳定

s = -0.1（同样靠近虚轴）
Δt = 21
→ z = 1 + 21·(-0.1) = -1.1     |z| > 1 ❌ 不稳定！
```

**对 HiPPO 矩阵的影响**：

```
HiPPO 矩阵 A 的对角线元素范围: 约 -1 到 -N（如 N=64 则 -1 到 -64）

若 Δt = 0.1：
  最小的稳定极点 s=-1 → z = 1 + 0.1·(-1) = 0.9  ✅
  最大的极点 s=-64  → z = 1 + 0.1·(-64) = -5.4 ❌ 严重失稳！

Δt 必须极小才能压制最大的特征值，但这样小的 Δt 会导致：
  - 离散系统时间常数过大，状态更新极其缓慢
  - 无法捕捉快速变化的序列动态
```

**矩阵形式的稳定性条件**：`ρ(I + Δt·A) < 1`（谱半径小于 1）

- A 的特征值越大（绝对值），Δt 必须越小
- 对 HiPPO 矩阵（特征值覆盖 -1 到 -N 的大范围），Δt 几乎必须极小
- 这在实际训练中是无法接受的约束——没有可行的 Δt 能同时满足所有极点的稳定性

**一句话总结**：Forward Euler 的映射 `z = 1 + Δt·s` 不是保角映射——s 平面的左半稳定区不会完整地映射到 z 平面的单位圆内。只有当 Δt 足够小，使得对所有稳定极点都有 `|1 + Δt·λ| < 1` 时才稳定。实践中对特征值分布广的矩阵（如 HiPPO），不存在这样的 Δt。

#### 3.3.2 指数离散化的稳定性优势

**指数映射**：`dA = exp(Δt·A)` 天然稳定。

- 对任意 Δt > 0，若 A 稳定（特征值实部 < 0），则 `exp(Δt·A)` 的特征值 = `exp(Δt·λ_A)` 永远在单位圆内
- 无需限制 Δt 的取值范围

```
连续稳定: Re(λ_A) < 0
  → |exp(Δt·λ_A)| = exp(Δt·Re(λ_A)) < 1
  → exp(Δt·A) 永远稳定
```

这就是 Mamba 选择 `exp(Δt·A)` 而非 `(I + Δt·A)` 的根本原因。

### 3.4 零阶保持 (Zero-Order Hold, ZOH) —— 精确离散化

假设输入 u(t) 在采样区间内保持常数 u_k，则：

```
∫_{0}^{Δt} e^(A·(Δt-τ)) · B · u_k · dτ
= [∫_{0}^{Δt} e^(A·s) · ds] · B · u_k        (令 s = Δt-τ, ds = -dτ)
= A^(-1) · (e^(A·Δt) - I) · B · u_k
```

因此 **ZOH 精确离散化**：

```
A_d = e^(A·Δt)                                    ← 矩阵指数
B_d = A^(-1) · (e^(A·Δt) - I) · B               ← 依赖于 A^(-1)

x_{k+1} = A_d · x_k + B_d · u_k
```

**问题**：需要计算 A^(-1)，当 A 条件数差时数值不稳定。

### 3.5 Mamba 的离散化选择

Mamba 使用的是**简化版的指数离散化**，而非完整的 ZOH：

```
dA = exp(dt · A)      ← 矩阵指数（与 ZOH 相同）
dB = dt · B           ← 简化：直接用 dt 缩放（不同于 ZOH 的 A^(-1)(e^(A·dt)-I)·B）
```

即 Mamba 选择 `dB = dt · B` 而非 ZOH 的 `B_d = A^(-1) · (e^(A·dt) - I) · B`。

#### 为什么 Mamba 这样做？

这个问题在 GitHub Issue [#114](https://github.com/state-spaces/mamba/issues/114) 中被提出，Albert Gu（论文一作）解释道：

1. **`dt · B` 是一个更简单的近似**：在实践中效果很好
2. **A 是固定的（HiPPO 初始化）**：不随输入变化，而 B 是输入相关的
3. **这个选择是有意的**：简化离散化，同时依靠选择性机制（dt、B、C 的输入依赖性）来补偿

> *"This [dt * B] is more useful"*
> — Albert Gu, [Issue #114](https://github.com/state-spaces/mamba/issues/114)

### 3.6 代码对照

```python
# mamba_ssm 内部实现（简化版）
dt = F.softplus(self.dt_proj(x))      # Δt > 0，用 softplus 保证正值
dA = torch.exp(dt * A)                # exp(Δt · A) — 与 ZOH 相同
dB = dt * B                            # dt · B — 简化近似，非 ZOH

h_new = dA * h_old + dB * x            # 状态更新
```

**关键**：`dt` 是**输入相关的**（由 `x` 通过神经网络生成），这使得模型可以动态调整离散化步长。同时 `B` 也是输入相关的（选择性机制），这才是 Mamba 表达力的核心来源。

---

## 4. Mamba 的核心创新：选择性机制

### 4.1 标准 SSM vs 选择性 SSM

**标准 SSM**（如 S4、DSS）：
```
A, B, C 是**固定参数** — 与输入无关
→ 无法根据内容选择保留哪些信息
```

**选择性 SSM (Mamba)**：
```
dt, B, C 是**输入相关的函数** — 由 x 动态生成
→ 可以选择性传播或遗忘信息
```

### 4.2 选择性机制图示

```
输入 x_t
   │
   ├──→ dt_proj(x_t) → dt  （决定**何时**更新状态）
   ├──→ B_proj(x_t)  → B    （决定**如何**受输入影响）
   └──→ C_proj(x_t)  → C    （决定**如何**产生输出）
```

这让模型能够：
- **选择性遗忘**：某些 token（如填充、噪声）可以快速遗忘
- **选择性记住**：关键 token（如分子关键官能团）可以长期保持

### 4.3 为什么选择性能至关重要？

考虑分子 SMILES 序列：`CC(=O)OC`（乙酸甲酯）

```
标准 SSM：所有 token 平等处理，填充 token 也被记住
选择性 SSM：自动聚焦关键原子/化学键，忽略填充
```

### 4.4 选择性的代码实现

```python
# 简化示意（mamba_ssm 内部逻辑）
x_proj = x  # 输入投影

# 分离出 dt, B, C（每个都是输入的函数）
dt = self.dt_proj(x)        # (B, L, d_inner)
B  = self.B_proj(x)         # (B, L, d_state)
C  = self.C_proj(x)         # (B, L, d_state)

# 这就是"选择性" — 参数随输入变化
```

---

## 5. 并行扫描算法 (Parallel Scan)

### 5.1 RNN 的串行困境

传统 RNN 必须**顺序计算**：

```
h_0 = f(x_0)
h_1 = f(x_1, h_0)   ← 必须等 h_0
h_2 = f(x_2, h_1)   ← 必须等 h_1
...
```

无法并行，O(N) 串行时间。

### 5.2 并行扫描的洞察

观察离散化后的状态更新：

```
h_t = dA_t · h_{t-1} + dB_t · x_t

写成展开形式：
h_2 = dA_2·dA_1·h_0 + dA_2·dB_1·x_1 + dB_2·x_2
```

这类似**前缀和 (prefix sum)** 的结构！前缀和可以通过并行算法加速。

### 5.3 并行扫描原理

将递归转化为**可并行的形式**：

```
定义: 组合算子 (•)
   (A, b) • (A', b') = (A·A', A·b' + b)

则: h_t = dA_t·h_{t-1} + dB_t·x_t
         ↓ 递归展开
    h_t = (dA_t, dB_t·x_t) • (dA_{t-1}, dB_{t-1}·x_{t-1}) • ... • (dA_0, dB_0·x_0)
```

计算所有 `(dA_i, dB_i·x_i)` 可**完全并行**（各时间步独立），然后用**并行扫描**组合。

### 5.4 时间复杂度对比

| 方式 | 时间复杂度 | 说明 |
|------|-----------|------|
| 标准 RNN | O(N) 串行 | N 步必须顺序执行 |
| 卷积 SSM | O(N log N) | FFT 加速卷积 |
| **Mamba 并行扫描** | **O(N)** | 完全并行，仅需 O(log N) 通信 |

### 5.5 硬件感知设计

`mamba_ssm` 库使用**融合 kernel**，避免中间结果写回 HBM：

```
GPU 内存层级: HBM（大但慢）←→ SRAM（小但快）
                            ↑
                    融合 kernel 直接在 SRAM 计算
                    避免反复读写 HBM
```

---

## 6. HiPPO 矩阵初始化

### 6.1 状态空间压缩问题

SSM 的隐藏状态 `h` 维度有限（通常 64-128），但序列可能很长：

```
问题: 如何用有限状态近似所有历史信息？
```

### 6.2 HiPPO 理论

HiPPO (High-order Polynomial Projection Operator) 理论提出：

> 给定历史输入函数 `x(t)`，找到最优的有限维状态 `h(t)` 使
> `||x(t) - projection(x(t))||` 最小

Gu & Dao 证明：使用特定形式的矩阵 `A` 初始化，可以获得**最优的状态空间近似**。

### 6.3 HiPPO 矩阵形式

Mamba 使用的 HiPPO 矩阵（非扫描模式）：

```
A[i,j] ∝ -(i+1)  if i >= j
         = 0      otherwise
```

形式为下三角矩阵，对角线为负：

```
A = [[ -1,   0,   0, ...],
     [ -2,  -2,   0, ...],
     [ -3,  -3,  -3, ...],
     ...]
```

### 6.4 为什么 HiPPO 有效？

```
HiPPO 初始化的 A 矩阵具有性质:
1. 远处的历史信息自然衰减（指数级）
2. 较近的历史信息被保留
3. 状态范数有界（数值稳定）
```

这使得模型**从第一天就能很好地压缩历史**，无需从零学习。

### 6.5 代码对照

```python
# mamba_ssm 自动使用 HiPPO 初始化
# 用户只需指定 d_state（状态维度）
mamba = Mamba2(
    d_model=256,
    d_state=64,      # 状态维度，HiPPO 矩阵大小为 d_model × d_state
    ...
)
# 内部自动生成 HiPPO 矩阵并用于初始化
```

---

## 7. 完整前向传播流程

### 7.1 数据流总览

```
输入 x (B, L, d_model)
   │
   ├── in_proj: x → [x, z]          分割为两份 (d_model → d_inner*2)
   │
   ├── conv1d: x → local_x           因果卷积，捕获局部依赖 (d_conv=4)
   │
   ├── x_proj: local_x → [dt, B, C]  生成选择性参数 (dt, B, C 都是输入的函数)
   │
   ├── SSM: 并行扫描                  用扫描计算 y (硬件感知加速)
   │
   ├── 门控: y = y * silu(z)         门控机制 (z 通过 silu 激活)
   │
   └── out_proj: y → output          输出投影 (d_inner → d_model)
```

### 7.2 各模块详解

#### In_proj（输入投影）

```python
# x: (B, L, d_model)
# → xz: (B, L, d_inner * 2)
xz = self.in_proj(x)
x, z = xz.chunk(2, dim=-1)  # 分割为两份

# x 用于 SSM 计算
# z 用于后面的门控
```

#### 局部卷积（捕获局部依赖）

```python
# 因果卷积：output[i] 只依赖 input[0..i]（不泄露未来信息）
# x: (B, L, d_inner)
# → local_x: (B, L, d_inner)
local_x = self.conv1d(x)[:, :L]  # 截断到原始长度

# d_conv=4 表示每一步只看前4个 token 的局部上下文
```

#### 选择性投影（核心创新）

```python
# local_x: (B, L, d_inner)
# → dt, B, C 都是输入的函数（选择性）

x_proj = self.x_proj(local_x)  # (B, L, d_inner + d_state * 2)
# 假设 d_inner=512, d_state=64
# 则 x_proj 大小 = 512 + 64*2 = 640
# 分割：
#   dt: (B, L, d_inner)      — 离散化步长
#   B:  (B, L, d_state)       — 输入权重
#   C:  (B, L, d_state)       — 输出权重
```

#### SSM 状态更新（并行扫描）

```python
# 1. 离散化
dt = F.softplus(self.dt_proj(local_x))  # (B, L, d_inner)
dt = torch.exp(dt)  # 加速形式

# A 是可学习的 HiPPO 矩阵
dA = torch.exp(torch.einsum('sd,bsd->bsd', dt, A))  # (B, L, d_inner, d_state)
dB = torch.einsum('bsd,bs->bsd', dt, B)              # (B, L, d_state)

# 2. 并行扫描计算 y
# y: (B, L, d_inner)
y = self.selective_scan(x, dA, dB, C)
```

#### 门控机制

```python
# y: (B, L, d_inner)
# z: (B, L, d_inner) — 来自输入投影的"门控信号"
y = y * F.silu(z)  # SiLU/GELU 门控
```

#### 输出投影

```python
# y: (B, L, d_inner) → output: (B, L, d_model)
output = self.out_proj(y)
```

---

## 8. 双向 Mamba (Bi-Mamba)

### 8.1 为什么要双向？

单向 SSM 只能利用**历史信息**（从左到右）：

```
前向:  x_0 → x_1 → x_2 → ... → x_N
       只能看到前缀 [x_0, x_1, ..., x_t] 用于预测 t
```

对于分子性质预测、基因组等任务，**双向信息**至关重要。

### 8.2 Bi-Mamba 结构

```python
class BiMambaEncoder(nn.Module):
    def __init__(self, ...):
        # 两组独立的 Mamba 层
        self.forward_layers = nn.ModuleList([BiMambaBlock(...) for _ in range(n_layers)])
        self.backward_layers = nn.ModuleList([BiMambaBlock(...) for _ in range(n_layers)])
        
        # 双向融合门
        self.fusion_gate = nn.Linear(d_model * 2, d_model * 2)
```

### 8.3 前向传播

```python
def forward(self, input_ids, attention_mask=None):
    # 1. 词嵌入 + 位置嵌入
    hidden_states = token_embeds + position_embeds
    
    # 2. 前向 Mamba（从左到右）
    forward_hidden = hidden_states
    for layer in self.forward_layers:
        forward_hidden = layer(forward_hidden)
    
    # 3. 后向 Mamba（从右到左）
    backward_hidden = torch.flip(hidden_states, [1])  # 反转序列
    for layer in self.backward_layers:
        backward_hidden = layer(backward_hidden)
    backward_hidden = torch.flip(backward_hidden, [1])  # 恢复顺序
    
    # 4. 双向融合
    combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
    gate = torch.sigmoid(self.fusion_gate(combined))
    gate_forward, gate_backward = gate.chunk(2, dim=-1)
    
    # 加权融合
    fused = gate_forward * forward_hidden + gate_backward * backward_hidden
    
    return self.norm(fused)
```

### 8.4 融合门控机制

```
combined = concat([forward_h, backward_h])  # (B, L, d_model*2)
gate = sigmoid(linear(combined))            # (B, L, d_model*2)
gate = [gate_f, gate_b]                     # 分割为前向/后向权重

output = gate_f * forward_h + gate_b * backward_h
```

融合门让模型**动态决定**每个位置更依赖前向还是后向信息。

---

## 9. 代码实现逐行解析

### 9.1 BiMambaBlock

```python
class BiMambaBlock(nn.Module):
    """
    双向 Mamba 块 - 基于 mamba_ssm 的 Mamba2 实现。
    
    Mamba2 核心特性：
    - 选择性状态空间模型（Selective SSM）
    - 并行扫描算法（parallel scan）- O(N) 复杂度
    - HiPPO 矩阵初始化 - 最优的 state 初始化
    - 硬件感知的融合操作
    """
    
    def __init__(
        self,
        d_model: int,        # 模型隐藏层维度（输入输出维度）
        d_state: int = 64,   # SSM 状态维度，通常 64 或 128
        d_conv: int = 4,     # 局部卷积核宽度，用于捕获局部依赖
        expand: int = 2,     # 块扩展因子，控制内部维度（d_inner = expand * d_model）
        use_fast_path: bool = True,  # 是否使用融合 kernel（需要 mamba_ssm 编译安装）
        layer_idx: Optional[int] = None,  # 层索引，用于调试
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # 核心：直接使用 mamba_ssm 库的 Mamba2 实现
        # 这是一个高度优化的融合实现
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **factory_kwargs,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        return self.mamba(hidden_states)
```

**关键点**：`BiMambaBlock` 本身是**单向**的，因为 `Mamba2` 底层是单向扫描。双向性由外层的 `BiMambaEncoder` 提供。

### 9.2 BiMambaEncoder

```python
class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器，同时从左到右和从右到左处理序列。
    """
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, ...):
        super().__init__()
        
        # 词嵌入 + 位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # 两组 Mamba 层
        self.forward_layers = self._make_layers(...)  # 前向
        self.backward_layers = self._make_layers(...) # 后向
        
        # LayerNorm + Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 双向融合门
        self.fusion_gate = nn.Linear(d_model * 2, d_model * 2)
```

### 9.3 BiMambaForPropertyPrediction

```python
class BiMambaForPropertyPrediction(nn.Module):
    """
    Bi-Mamba 分子性质预测模型。
    支持回归任务和分类任务。
    """
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, 
                 task_type="regression", pooling="mean", ...):
        super().__init__()
        
        # 核心编码器
        self.encoder = BiMambaEncoder(...)
        
        # 池化方法选择
        if pooling == "cls":
            # 可学习的 [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 预测头
        self.classifier = nn.Linear(d_model, num_labels)
        
        # 损失函数
        if task_type == "regression":
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()
```

### 9.4 池化方法

```python
def pool_sequence(self, encoder_outputs, attention_mask, pooling):
    """支持三种池化方法"""
    
    if pooling == "mean":
        # 注意力掩码加权平均
        if attention_mask is not None:
            sum_emb = torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1)
            sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
            return sum_emb / sum_mask.clamp(min=1e-9)
        return torch.mean(encoder_outputs, dim=1)
    
    elif pooling == "max":
        # 最大池化（masked）
        if attention_mask is not None:
            masked = encoder_outputs.clone()
            masked[attention_mask == 0] = -1e9
            return torch.max(masked, dim=1)[0]
        return torch.max(encoder_outputs, dim=1)[0]
    
    elif pooling == "cls":
        # [CLS] token（序列第一个位置）
        return encoder_outputs[:, 0]
```

---

## 10. Mamba vs Transformer vs RNN

### 10.1 复杂度对比

| 架构 | 空间复杂度 | 时间复杂度 | 并行化 |
|------|-----------|-----------|--------|
| Transformer | O(N²) | O(N²) | 序列内部并行 |
| Linear Attention | O(N) | O(N) | 序列内部并行 |
| Standard RNN | O(N) | O(N) 串行 | 不可并行 |
| **Mamba** | **O(N)** | **O(N)** | **完全并行** |

### 10.2 核心特性对比

| 特性 | Transformer | Mamba |
|------|-------------|-------|
| 内容选择 | ✅ Full attention | ✅ Selective SSM |
| 线性复杂度 | ❌ O(N²) | ✅ O(N) |
| 因果掩码 | 需要 | 天然因果 |
| 快速推理 | ❌ KV-cache 臃肿 | ✅ 5× higher throughput |
| 变长序列 | 需要 padding | 天然支持 |
| 双向建模 | ✅ Attention | ✅ Bi-Mamba |

### 10.3 Mamba 的优势场景

1. **长序列任务**：基因组（DNA）、音乐、长时间序列
2. **资源受限场景**：边缘设备、MacBook（MPS 加速）
3. **需要快速推理**：Mamba 推理速度是 Transformer 的 5 倍
4. **分子/化学任务**：SMILES、分子图表示

### 10.4 为什么 Mamba 适合分子性质预测？

```
分子表示:
- SMILES 字符串（如 CC(=O)OC 表示乙酸甲酯）
- 序列长度：通常 50-200 tokens
- 关键官能团的位置决定性质

Mamba 的优势:
1. 选择性机制自动聚焦关键化学基团
2. 线性复杂度适合处理大量分子
3. 双向建模捕获完整分子上下文
4. 硬件感知实现，Mac/PC 都能高效运行
```

---

## 附录：关键公式速查

### 状态空间模型

```
连续形式:
    dh/dt = A·h + B·x
    y     = C·h

精确解（积分形式）:
    x(t_b) = e^(A·(t_b-t_a)) · x(t_a) + ∫_{t_a}^{t_b} e^(A·(t_b-τ)) · B · u(τ) · dτ

ZOH 精确离散化:
    A_d = e^(A·Δt)                           ← 矩阵指数
    B_d = A^(-1) · (e^(A·Δt) - I) · B       ← 需要 A^(-1)

Mamba 实际使用:
    dA = exp(dt·A)    ← 矩阵指数（HiPPO 初始化）
    dB = dt·B        ← 简化近似（不经 A^(-1)）
```

### 选择性机制

```
dt, B, C = x_proj(x)     ← 都是输入的函数
h_t = exp(dt·A)·h_{t-1} + dt·B·x_t
y_t = C·h_t
```

### 并行扫描

```
组合算子: (A, b) • (A', b') = (A·A', A·b' + b')
递归:     h_t = (dA_t, dB_t·x_t) • (dA_{t-1}, dB_{t-1}·x_{t-1}) • ... • (dA_0, dB_0·x_0)
```

---

## 参考文献

1. Gu, A., & Dao, T. (2023). **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**. arXiv:2312.00752
2. GoEmotions: 基于双向 Mamba 的分子性质预测实现
3. mamba_ssm 库: https://github.com/state-spaces/mamba

---

*本教程由 AI 生成，基于 Bi-Mamba-Chem 项目源码及 Mamba 原始论文。*
