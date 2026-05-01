# Mamba & Bi-Mamba: Complete Theoretical Derivation

> **目标读者**：理解 Bi-Mamba 模型在分子性质预测中的完整数学原理，从控制理论出发，逐步推导到代码实现。**每一个公式都有从头到尾的推导过程，不会凭空出现。**

---

# 第一部分：状态空间模型——从直觉到数学

## 1.1 什么是"状态空间"？

在理解复杂的数学公式之前，先建立直觉。

想象你在一个迷宫里找出口。在任何时刻，你有一个"状态"——你现在的位置（坐标 x, y），你离出口`还有多远（距离 d）`。这三个数字 `(x, y, d)` 组成的**向量**，就完整描述了你在迷宫中的"状态"。

"**状态空间**"就是所有可能状态构成的集合——迷宫里每个位置+每个距离组合的所有可能性。

"**状态空间模型**"（State Space Model, SSM）要做的事情是：给定当前状态，根据新的信息（输入），预测下一个状态和输出。在数学上，这被表述为两个方程：

- **状态方程**：状态如何随输入而演变
- **输出方程**：如何从状态中"读出"输出

这一思想来源于控制理论，由 Rudolf Kalman 在 1960 年系统化（Kalman Filter）。

## 1.2 连续时间 SSM 的数学推导

### 1.2.1 物理直觉：一个水箱的例子

假设有一个水箱，水位是 $h(t)$。水龙头往里注水（输入 $x(t)$），水箱底部有个小孔漏水（自然衰减）。我们要预测水位的变化和出水口的流量（输出 $y(t)$）。

- **水位变化率** = **漏水速度** + **注水速度**
- 漏水速度∝当前水位（水越多漏得越快）
- 注水速度∝输入水流大小

用数学语言表达（先考虑最简单的一维情况，只有一个水箱）：

$$\frac{dh(t)}{dt} = a \cdot h(t) + b \cdot x(t)$$

其中：
- $a < 0$：负值表示水位越高手衰减越快（漏水）
- $b$：输入对水位变化的贡献系数
- $x(t)$：当前时刻的输入
- $\frac{dh(t)}{dt}$：水位的瞬时变化率（导数）

**输出**就是直接读取水位：

$$y(t) = c \cdot h(t)$$

这里 $c$ 是一个缩放系数。

### 1.2.2 推广到多维：向量与矩阵

真实问题中，水庙不是一维的。可能有 N 个互相连通的水箱，每个水箱有自己的水况、衰减速度、输入管道和输出管道。

在数学上，这意味着 $h(t)$ 不再是一个标量，而是一个 **N 维向量**：

$$h(t) = \begin{bmatrix} h_1(t) \\ h_2(t) \\ \vdots \\ h_N(t) \end{bmatrix}$$

相应地，$a$ 变成一个 $N \times N$ 的**矩阵 A**（描述 N 个水箱之间的互相影响），$b$ 变成一个 $N \times D$ 的**矩阵 B**（描述 D 维输入对 N 个状态的影响），$c$ 变成一个 $D' \times N$ 的**矩阵 C**（描述如何从 N 维状态读取 $D'$ 维输出）。

这就得到了**连续时间状态空间模型的标准形式**（常被称为 LTI 系统——线性时不变系统）：

$$\frac{dh(t)}{dt} = A \cdot h(t) + B \cdot x(t) \qquad \text{(状态方程)}$$

$$y(t) = C \cdot h(t) + D \cdot x(t) \qquad \text{(输出方程)}$$

其中：
| 符号 | 含义 | 形状（深度学习语境） |
|------|------|---------------------|
| $h(t)$ | 隐状态向量（系统的"记忆"） | $(d_{\text{inner}}, d_{\text{state}})$ |
| $x(t)$ | 输入信号（当前 token 的表示） | $(d_{\text{inner}},)$ |
| $y(t)$ | 输出信号 | $(d_{\text{inner}},)$ |
| $A$ | 状态转移矩阵——控制记忆衰减速度 | $(d_{\text{inner}}, d_{\text{state}})$ |
| $B$ | 输入矩阵——控制新信息如何写入记忆 | $(d_{\text{inner}}, d_{\text{state}})$ |
| $C$ | 输出矩阵——控制从记忆读取什么 | $(d_{\text{inner}}, d_{\text{state}})$ |
| $D$ | 直连矩阵（skip connection） | $(d_{\text{inner}},)$ |

**什么是 $d_{\text{inner}}$？**

在继续之前，明确定义这个贯穿全文的关键维度：

- **$d_{\text{inner}}$** = `expand_factor × d_model`（默认 `expand=2, d_model=256` → `d_inner=512`）
- 它是 **SSM 内部的"工作空间"维度**——输入先从 `d_model` 投影到 `d_inner`，在 $d_{\text{inner}}$ 维空间内完成 SSM + 卷积 + 激活，最后再投影回 `d_model`
- 类比 Transformer：就像 FFN 的 inner dimension 是 `4 × d_model`，Mamba 用 `2 × d_model` 作为内部扩展维度
- **$d_{\text{inner}}$ 个独立通道**：$d_{\text{inner}}$ 是"通道数"，每个通道拥有**自己独立的** $d_{\text{state}}$ 维隐状态（即每个通道的 SSM 独立运行，互不干扰）
- 因此 `A` 的形状是 $(d_{\text{inner}}, d_{\text{state}})$：$d_{\text{inner}}$ 个通道 × 每个通道 $d_{\text{state}}$ 个状态维度

**备注**：在 Mamba 的深度学习实现中，我们处理 $d_{\text{inner}}$ 个独立通道，每个通道有自己的 $d_{\text{state}}$ 维状态。所以 $A$ 和 $B$ 的形状不是 $(N, N)$ 而是 $(d_{\text{inner}}, d_{\text{state}})$——这意味着 **A 是对角结构的**（后面会详细解释为什么）。

**什么是 $d_{\text{state}}$？**

- **$d_{\text{state}}$** = SSM 状态维度，默认值 **16**
- 它代表**每个 SSM 通道的"记忆容量"**——即每个通道用多少个数字来压缩历史信息
- **物理直觉**：回到水箱类比，$d_{\text{state}}$ 就是每个水箱内部用来记录历史的"刻度数"。$d_{\text{state}}=16$ 意味着每个通道用 16 个数字来压缩整个序列的历史
- **与 $d_{\text{inner}}$ 的关系**：$d_{\text{inner}}$ 是"通道数"（横向），$d_{\text{state}}$ 是每个通道的"记忆深度"（纵向）。真正的隐状态 $h$ 的形状是 $(B, d_{\text{inner}}, d_{\text{state}})$——`d_inner` 个通道 × 每个通道 `d_state` 维记忆
- **为什么是 16？** 这是一个经验选择。更大会增加记忆容量但也增加计算开销。16 在分子 SMILES（通常长度 < 256）上已经被验证足够。对于更长的序列（如基因组），可以增加到 64 甚至 128
- **在 HiPPO 框架中**：$d_{\text{state}}$ 就是多项式的阶数 $N$——用 N 阶 Legendre 多项式来逼近历史函数。N=16 意味着用 16 个基函数的线性组合来表示历史信息

**为什么需要 $d_{\text{model}}$ 和 $d_{\text{inner}}$ 两个维度？**

如果你已经看到 $d_{\text{inner}}$ 是 SSM 内部的工作维度，可能会问：**为什么不直接用 $d_{\text{inner}}$ 表示每个 token 的向量，省掉 $d_{\text{model}} \to d_{\text{inner}} \to d_{\text{model}}$ 的来回投影？**

答案是两方面的：

**1. 残差连接要求维度一致**

每个 BiMambaBlock 的末尾有残差连接 `out_proj(y) + hidden_states`，两者必须同维度。如果 token 向量从头到尾都是 `512` 维（即用 $d_{\text{inner}}$ 做 $d_{\text{model}}$），那嵌入层的参数也会翻倍：

$$|\text{Token Embedding}| = |\text{Vocab}| \times d_{\text{model}} = 40 \times 256 = 10{,}240 \ \text{（当前）}$$

$$\text{如果改成 } d_{\text{inner}}=512\text{：} \quad 40 \times 512 = 20{,}480 \ \text{（翻倍）}$$

SMILES 词表只有约 40 个字符，用 512 维向量表示每个字符是**过度参数化**——额外参数几乎不带来新的表达能力。

**为什么 40 个字符不需要 512 维？**

从信息论角度：**区分 40 个不同字符，理论上只需要 $\log_2(40) \approx 5.3$ 个二进制位**（因为 $2^5=32 < 40 \leq 2^6=64$，取 6 位足够给每个字符发唯一的二进制编号）。

但 `d_model` 不能只取 6——因为嵌入层不只是给每个字符"发编号"，更要给每个字符留出空间存放**化学语义**：

| d_model | 类比 | 效果 |
|---------|------|------|
| **6 维** | 图书馆里每本书只有 6 位数的索书号 | 能找书，但不知道这本书讲什么 |
| **256 维** | 每本书除了编号，还附带一份 256 字的摘要 | 不翻开就知道主题、风格、难度 |

例如，`d_model=256` 时 `"C"` 的向量可以是这样的：

```
C = [0.12, -0.43, 0.88, ...   ← 前几维：身份（我是碳原子）
     0.91, 0.34, -0.62, ...   ← 中间：周期表第 6 号，常见价态 +4/-4
     ...                       ← 后面：常与 N、O 成键，出现在芳香环/羰基中
     ...]                      ← 更多维：和 "c"（小写芳香碳）是近亲
```

`d_model=6` 的话，`"C"` 只能是 `[0.12, -0.43, 0.88, 0.01, 0.55, -0.29]`——身份够用了，但"碳原子常和氧原子形成双键"这种化学知识根本没地方存。

**用线性代数理解**：嵌入矩阵的形状是 $(40, 512)$，其秩 $\text{rank} \leq \min(40, 512) = 40$。这意味着多出来的 $512-40=472$ 个维度里填不进任何有效的语义信息，只会记住训练集的随机噪音（过拟合），让模型在新 SMILES 上表现更差。

所以：**6 太小（无语义），512 太大（过拟合），256 是实验验证的甜点**。

**2. Expand/Contract 是经典设计模式**

这不是 Mamba 独有的，Transformer 也遵循相同的设计：

| | Transformer | Mamba（本项目） |
|---|---|---|
| **存储维度**（层间传递） | $d_{\text{model}}$ | $d_{\text{model}} = 256$ |
| **工作维度**（层内计算） | $d_{\text{ff}} = 4 \times d_{\text{model}}$ | $d_{\text{inner}} = 2 \times d_{\text{model}} = 512$ |
| **作用** | 思考时展开，存回时压缩 | 同上 |

**直觉类比**：你的长期记忆不需要把每个细节都原样记下来（$d_{\text{model}}$ 是压缩存储），但当你思考一个问题时，需要在脑子里展开更多临时"草稿纸"空间（$d_{\text{inner}}$ 是思考空间）。算完了，再把结果压缩存回去。

更技术地说：`d_model` 决定了模型参数量和层间通信的信息带宽（要小），`d_inner` 决定了每层内部计算的表达容量（要大）。两者的比值 `expand = d_inner / d_model = 2` 是经过实验验证的效率最优解。

**expand 是超参数，不是可学习参数**

你可能会问：`expand` 能不能通过训练自动调整？比如给个学习率让它自己长？

答案是不能。因为 `expand` 决定的是**网络的物理结构**，不是某个系数：

```python
# d_inner = expand × d_model = 2 × 256 = 512

# 以下所有层的形状都由 d_inner 决定：
self.in_proj  = Linear(d_model, 2*d_inner)          # (256 → 1024)
self.out_proj = Linear(d_inner, d_model)             # (512 → 256)
self.x_proj   = Linear(d_inner, dt_rank + 2*d_state) # (512 → 48)
self.conv1d   = Conv1d(d_inner, d_inner, groups=d_inner) # 512 个通道
self.A_log    = (d_inner, d_state)                   # (512, 16) 的矩阵
```

如果训练中途把 `expand` 从 2 改成 3，`d_inner` 从 512 → 768 → **上面所有矩阵的形状全变了**。旧权重无法塞进新形状，已学到的信息全部报废。

**类比**：就像盖房子中途决定加一层楼——不是拧螺丝调参数，而是把天花板拆了重盖。`expand` 是建筑蓝图，不是施工方案：

```python
# expand — 建楼之前决定（超参数）：
expand = 2       # 决定了整栋楼几层、每层多大

# 这些才是训练时调整的（可学习参数）：
learning_rate    # 怎么学
weight_decay     # 是否正则化
dropout          # 是否丢弃
```

**如果想"动态调整" expand**，可行的替代思路：

| 方法 | 做法 | 代价 |
|------|------|------|
| **超参数搜索** | 训练前用 grid search 试 expand=1/2/4/8 | 需要多跑几轮训练 |
| **结构化剪枝** | 训练时用 expand=4，训完后剪掉不重要的通道 | 实现复杂 |
| **每层不同 expand** | 第一层 expand=4，越深层越小 | 需手动设计 |

其中超参数搜索最务实。但 Mamba 论文已经证明 `expand=2` 是效率最优解——更高收益递减，更低表达能力不足。

### 1.2.3 向量化形式的推导回顾

从标量形式推广到向量形式，本质上是把每个维度的独立方程"打包"在一起：

对于第 i 个状态维度：
$$\frac{dh_i(t)}{dt} = \sum_{j=1}^{N} A_{ij} \cdot h_j(t) + \sum_{k=1}^{D} B_{ik} \cdot x_k(t)$$

把所有 i 写成向量形式就是 $\frac{dh}{dt} = Ah + Bx$。

在 Mamba 中，A 是对角矩阵（$A_{ij}=0 \text{ for } i \neq j$），所以每个状态维度独立演化：
$$\frac{dh_i(t)}{dt} = A_{ii} \cdot h_i(t) + B_i \cdot x(t)$$

---

# 第二部分：从连续到离散——离散化

## 2.1 为什么需要离散化？

计算机处理的都是离散序列（token by token），而上面的 SSM 是用微分方程（包含导数 $\frac{dh}{dt}$）描述的连续时间系统。

**核心矛盾**：我们有离散的 token 序列 $\{x_0, x_1, x_2, ..., x_{L-1}\}$，但 SSM 的公式要求输入是连续函数 $x(t)$。

**解决方案**：离散化——用离散的递推公式 $h_{t+1} = \bar{A}h_t + \bar{B}x_t$ 来近似连续系统，这就需要一个**时间步长** $\Delta$（读作 Delta，也记作 dt）来控制采样间隔。

## 2.2 前向欧拉方法（Forward Euler）——最直观的离散化

### 推导过程

回到连续方程：
$$\frac{dh(t)}{dt} = A \cdot h(t) + B \cdot x(t)$$

导数的定义：
$$\frac{dh(t)}{dt} = \lim_{\Delta \to 0} \frac{h(t+\Delta) - h(t)}{\Delta}$$

当 $\Delta$ 很小但不是无穷小时，这是一个近似（前向欧拉近似）：

$$\frac{h(t+\Delta) - h(t)}{\Delta} \approx A \cdot h(t) + B \cdot x(t)$$

两边乘以 $\Delta$：

$$h(t+\Delta) - h(t) \approx \Delta \cdot A \cdot h(t) + \Delta \cdot B \cdot x(t)$$

移项：

$$h(t+\Delta) \approx h(t) + \Delta \cdot A \cdot h(t) + \Delta \cdot B \cdot x(t)$$

提取公因子：

$$h(t+\Delta) \approx (I + \Delta A) \cdot h(t) + (\Delta B) \cdot x(t)$$

写作离散递推形式（用下标 $t$ 和 $t+1$ 替代连续参数 $t$ 和 $t+\Delta$）：

$$\boxed{h_{t+1} = \bar{A} \cdot h_t + \bar{B} \cdot x_t}$$

其中离散化后的矩阵：
$$\bar{A} = I + \Delta A, \quad \bar{B} = \Delta B$$

**优点**：推导简单直观。
**缺点**：当 A 的特征值很大时（即系统很"硬"，stiff），这种线性近似不稳定——误差会随时间步累积。

## 2.3 零阶保持方法（ZOH）——精确离散化

### 关键洞察

前向欧拉用的是**线性近似**。但连续 SSM 的齐次部分 $\frac{dh}{dt} = Ah$ 有**精确解**。

### 推导过程

先考虑没有输入的情况（$B \cdot x(t) = 0$）：

$$\frac{dh(t)}{dt} = A \cdot h(t)$$

这是一个一阶线性常微分方程。假设 A 是标量（推广到对角矩阵是自然的），这个方程有精确解：

$$h(t) = e^{At} \cdot h(0)$$

> **推导验证**：$\frac{d}{dt}[e^{At} \cdot h(0)] = A \cdot e^{At} \cdot h(0) = A \cdot h(t)$ ✓

推导过程：
$$\frac{dh}{dt} = A \cdot h$$
分离变量：
$$\frac{dh}{h} = A \cdot dt$$
两边积分：
$$\int \frac{1}{h} \, dh = \int A \, dt$$
$$\ln|h| = A \cdot t + C$$
提指数：
$$h(t) = e^{A t + C} = e^C \cdot e^{A t}$$
`e^C` 是什么？把 `t=0` 代入：
$$h(0) = e^C \cdot e^{A \cdot 0} = e^C \cdot 1 = e^C$$
所以 `e^C = h(0)`，代回去：
$$h(t) = h(0) \cdot e^{A t}$$
这意味着，如果我们知道 $h(0)$，经过时间 $\Delta$ 后：
$$h(\Delta) = e^{A \cdot \Delta} \cdot h(0)$$

所以离散化的状态转移矩阵应该是：
$$\bar{A} = e^{\Delta \cdot A} \quad \text{而不是前向欧拉的} \quad I + \Delta A$$

### 加入输入项——零阶保持（ZOH）的完整推导

零阶保持的含义：在每个离散采样间隔 $[t, t+\Delta]$ 内，**假设输入 $x(t)$ 保持不变**（"保持"就是"hold"的含义）。

这给了我们以下"分段处理"的方式来解决非齐次方程 $\frac{dh}{dt} = Ah + Bx$：

**第一步**：在区间 $[t, t+\Delta]$ 内，$x(\tau) = x_t$（常量）。方程变为：
$$\frac{dh(\tau)}{d\tau} = A \cdot h(\tau) + B \cdot x_t$$

**第二步推导**：这是一个常系数线性非齐次 ODE。用**积分因子法**（integrating factor）推导通解：

首先将方程写成标准形式：

$$\frac{dh}{d\tau} - A \cdot h = B \cdot x_t$$

关键洞察：乘以 $e^{-A\tau}$ 可以让左边变成乘积求导的形式（$e^{-A\tau}$ 叫做"积分因子"）：

$$\color{blue}{e^{-A\tau}} \cdot \frac{dh}{d\tau} - \color{blue}{e^{-A\tau}} \cdot A \cdot h = e^{-A\tau} \cdot B \cdot x_t$$

左边正好是 $e^{-A\tau} \cdot h(\tau)$ 的导数：

$$\frac{d}{d\tau}\left( e^{-A\tau} \cdot h(\tau) \right) = e^{-A\tau} \cdot \frac{dh}{d\tau} + (-A) \cdot e^{-A\tau} \cdot h(\tau) \;\; \checkmark$$

所以方程简化为：

$$\frac{d}{d\tau}\left( e^{-A\tau} \cdot h(\tau) \right) = e^{-A\tau} \cdot B \cdot x_t$$

两边从 $\tau = 0$ 到 $\tau = \Delta$ 积分：

$$\int_0^{\Delta} \frac{d}{d\tau}\left( e^{-A\tau} \cdot h(\tau) \right) d\tau = \int_0^{\Delta} e^{-A\tau} \cdot B \cdot x_t \; d\tau$$

左边直接算（微积分基本定理）：

$$\left. e^{-A\tau} \cdot h(\tau) \right|_0^{\Delta} = e^{-A\Delta} \cdot h(\Delta) - e^0 \cdot h(0)$$

右边 $B x_t$ 是常量，提出积分外：

$$= B \cdot x_t \cdot \int_0^{\Delta} e^{-A\tau} d\tau$$

整理得：

$$e^{-A\Delta} \cdot h(\Delta) - h(0) = B x_t \cdot \int_0^{\Delta} e^{-A\tau} d\tau$$

两边乘 $e^{A\Delta}$（把积分因子"请回去"）：

$$h(\Delta) - e^{A\Delta} \cdot h(0) = e^{A\Delta} \cdot B x_t \cdot \int_0^{\Delta} e^{-A\tau} d\tau$$

把 $e^{A\Delta}$ 塞进积分（$e^{A\Delta} \cdot e^{-A\tau} = e^{A(\Delta - \tau)}$），并令 $h(t) = h(0)$，$h(t+\Delta) = h(\Delta)$：

$$\boxed{h(t+\Delta) = e^{A\Delta} \cdot h(t) + \int_0^{\Delta} e^{A(\Delta - \tau)} \cdot B \cdot x_t \; d\tau}$$

> **直觉总结**：① 乘 $e^{-A\tau}$ → 左边变导数 ② 积分消导数 ③ 乘 $e^{A\Delta}$ 回来，得到显式解。

> **直觉理解**：右边第一项 $e^{A\Delta}h(t)$ 是旧状态的衰减，第二项是输入 $Bx_t$ 在整个时间间隔 $[0, \Delta]$ 内对各时刻状态的贡献，每个时刻 $\tau$ 的贡献为 $e^{A(\Delta-\tau)}Bx_t d\tau$（因为输入在 $\tau$ 时刻加入，到 $\Delta$ 时刻又衰减了 $\Delta-\tau$ 时间）。

**第三步**：因为 $B$ 和 $x_t$ 在积分区间内是常数，可以提到积分外面：
$$h(t+\Delta) = e^{A\Delta} \cdot h(t) + \left( \int_0^{\Delta} e^{A(\Delta - \tau)} d\tau \right) \cdot B \cdot x_t$$

**第四步**：计算积分（做变量代换 $s = \Delta - \tau$，$ds = -d\tau$）：
$$\int_0^{\Delta} e^{A(\Delta - \tau)} d\tau = \int_0^{\Delta} e^{As} ds = \frac{e^{A\Delta} - I}{A}$$

（当 A 可逆时。对于对角矩阵，A 的对角元素都不为 0 → 可逆。）

> **什么是矩阵可逆？** $A^{-1}$ 就是"矩阵版本的倒数"：$A^{-1} \cdot A = I$，就像 $1/5 \times 5 = 1$。标量不能除以 0（$1/0$ 无意义），矩阵也一样——如果信息在变换中"被压扁了"（行列式为 0），就再也找不到独一无二的逆。
>
> **直观理解**：矩阵 A 像揉面团——把向量变形成别的东西。$A^{-1}$ 就是把面团还原回去。如果某次揉面把面团压成了纸片（某个状态维度坍缩为 0），就再也撑不回原来的面团了（不可逆 = 信息永久丢失）。
>
> **对角矩阵特别简单**：$A = \text{diag}(a_1, a_2, ..., a_N)$，只要每个 $a_i \neq 0$ → 可逆。逆矩阵就是 $\text{diag}(1/a_1, 1/a_2, ..., 1/a_N)$，几乎零开销。
>
> **在 Mamba 中**：A 初始化为 $[-1, -2, ..., -16]$（每行），**全部非零 → 每一行都可逆**。所以 ZOH 公式里的 $A^{-1}$ 永远是合法的，不会崩溃。

**第五步**：代入得到 ZOH 离散化：
$$\boxed{h_{t+1} = e^{\Delta A} \cdot h_t + A^{-1}(e^{\Delta A} - I) \cdot \Delta B \cdot x_t}$$

即离散化参数为：
$$\bar{A} = e^{\Delta \cdot A}, \quad \bar{B} = (\Delta A)^{-1} \cdot (e^{\Delta A} - I) \cdot \Delta B$$

## 2.4 分辨率不变性——为什么要用指数离散化

这是 Mamba 中最重要的理论根基之一。

### 命题

一个连续时间系统天然与采样率无关。指数离散化 $e^{\Delta A}$ 能**严格保持**这一性质，而线性离散化 $(I + \Delta A)$ 不能。

### 证明

**指数离散化**：对于同一系统（相同的 A），用不同采样率采样：
- 快采样 $\Delta = 0.1$（两次）：$e^{0.1A} \cdot e^{0.1A} = e^{0.2A}$
- 慢采样 $\Delta = 0.2$（一次）：$e^{0.2A}$

两次快采样的组合 = 一次慢采样：$e^{0.1A} \cdot e^{0.1A} = e^{0.1A + 0.1A} = e^{0.2A}$ ✓

关键性质：$\exp$ 函数满足 $\exp(p) \cdot \exp(q) = \exp(p+q)$。

**前向欧拉**：同样的情况：
- 快采样 $\Delta = 0.1$（两次）：$(I + 0.1A)(I + 0.1A) = I + 0.2A + 0.01A^2$
- 慢采样 $\Delta = 0.2$（一次）：$I + 0.2A$

两者之差为 $0.01A^2$——**不相等**！✗

### 物理含义

在物理世界中，一个系统以什么采样率被观察，不应该改变系统本身的演化规律。指数离散化保证了这一物理直觉。

## 2.5 本项目的简化离散化——一项实现选择

> **重要说明**：原版 Mamba 论文对所有参数使用**完整的 ZOH 离散化**：$\bar{A} = e^{\Delta A}$，$\bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B$。本文档以下讨论的是**本项目 `bimamba.py` 的实现选择**——使用了简化的 B 离散化。

### 本项目实际使用的公式

本项目对 A 和 B 使用了不对称的处理：

$$\boxed{\bar{A} = e^{\Delta \cdot A} \quad \text{（精确，同 Mamba 论文）}, \qquad \bar{B} = \Delta \cdot B \quad \text{（简化，非论文默认）}}$$

### 为什么可以简化 B？

常见的解释是"ZOH 需要矩阵求逆，复杂度 O(N³)"——**这在 Mamba 中不成立**。因为 A 是对角阵，求逆只是逐元素 `1/x`（O(N)），而 $e^{\Delta A}$ 在计算 $\bar{A}$ 时已经算过了，ZOH 版 $\bar{B}$ 的实际额外开销不过是：

```python
# Ā = exp(ΔA) 已经算过，白嫖 ✓
# ZOH 版 B̄：一次减法 + 一次逐元素除法 + 一次逐元素乘法
B_zoh = ((exp_ΔA - 1) / (dt * A)) * dt * B   # 全是逐元素，O(N)
```

对 `d_state=16`，这是零开销级别的差异。所以**性能不是简化 B 的真正原因**。

真正的原因是：

**1. B 是可学习的——模型能"学回来"**

简化版 $B̄ = Δ·B$ 与 ZOH 版 $B̄ = (e^{ΔA} - 1)/(ΔA) · Δ · B$ 的差距，只是一个缩放因子 $(e^{ΔA} - 1)/(ΔA)$。B 在训练中本来就会被更新——如果这个缩放因子有用，模型自己会把 B 调大/调小来补偿。**Δ 不同——它出现在指数里（$Ā = e^{ΔA}$），无法通过调参弥补**。

**2. 对精度的影响可忽略**

实验表明，在 SMILES 性质预测任务上，简化 B 和完整 ZOH 的精度差异在统计噪声范围内——因为神经网络有很强的容错和自补偿能力，一个逐元素的缩放因子完全可以被后续层吸收。

### 为什么简化 B 而非 A？

两者不能对等看待：

- **A 的误差按指数累积**：$h_t = \bar{A}h_{t-1} + ...$，A 的误差每一步都通过 $\bar{A}$ 乘到下一状态。如果 $\bar{A}$ 有 1% 误差，经 $t=100$ 步，累积误差约为 $(1.01)^{100} \approx 2.7$ 倍。

- **B 的误差按线性累积**：B 的误差只影响当前输入对状态的贡献，不会在后续步骤中被自我放大。多步的 B 误差是相加关系。

**结论**：A 必须精确（用 exp），B 可简化（模型学得回来）。

---

# 第三部分：选择性机制——Mamba 的核心创新

## 3.1 背景：经典 SSM 的局限性

### 什么是"线性时不变"（Linear Time-Invariant, LTI）？

到目前为止我们讨论的 SSM 都是 **LTI 系统**：
- **线性**：状态更新是矩阵乘法（线性变换）
- **时不变**：矩阵 $A, B, C, \Delta$ 在每一个时间步都是相同的，不随输入变化

对于第 t 个 token "Hello"，和对于第 t+1 个 token "World"，SSM 使用完全相同的 A、B、C 参数来处理——模型对所有 token "一视同仁"。

### 为什么这是个问题？

考虑一个**选择性复制任务**：

输入序列：`[A, x, B, y, C, z, (COPY), _, _, _]`
目标输出：`[A, B, C]`（只复制标记的 token，按顺序）

经典 LTI SSM 处理所有 token 用相同的参数，它无法"知道"哪些是标记的 token 需要记住、哪些应该忽略。**它缺乏内容感知能力**。

### Transformer 为什么成功

Transformer 的 self-attention 对每个 token 动态计算与所有其他 token 的相关性权重——不同 token 之间的注意力权重是不同的，取决于 token 的具体内容。

**Mamba 的目标**：在保持 SSM 的高效性（O(N) 复杂度）的同时，引入 Transformer 的内容感知能力。

## 3.2 选择性机制的核心思想

### 原理

让 SSM 的参数 **依赖于输入本身**：

$$\Delta_t = f_{\Delta}(x_t), \quad B_t = f_B(x_t), \quad C_t = f_C(x_t)$$

也就是说，对于不同的输入 token，SSM 使用不同的参数来处理。

### 数学形式

在实际实现中（对应 `bimamba.py` 中的代码），这些函数是简单的线性投影：

```
x(t) → x_proj(x(t)) → [dt(t), B(t), C(t)]
dt(t) → dt_proj(dt(t)) → Softplus → Δ(t)
```

> **与原始 Mamba 论文的区别**：论文中使用 $\Delta = \text{softplus}(\text{Parameter} + \text{Broadcast}_D(\text{Linear}_1(x)))$，即先将 $x$ 投影到**1 维**再广播到 $d_{\text{inner}}$ 维。本项目使用**低秩投影**（$x \to dt\_rank \to d_{\text{inner}}$），参数更灵活但参数量略大。两者最终输出形状相同：$\Delta \in \mathbb{R}^{d_{\text{inner}}}$。

详细的维度变化：

1. **$x \in \mathbb{R}^{d_{\text{inner}}}$**：当前 token 的内部表示
2. **$x\_proj(x) \in \mathbb{R}^{dt\_rank + 2 \cdot d_{\text{state}}}$**：一次线性投影，同时生成 dt、B、C 的原始值
3. **$dt = x\_proj(x)[:dt\_rank] \in \mathbb{R}^{dt\_rank}$**：低秩的时间步参数
4. **$dt\_proj(dt) \in \mathbb{R}^{d_{\text{inner}}}$**：从低秩升到全维度
5. **$\Delta = \text{Softplus}(dt\_proj(dt)) \in \mathbb{R}^{d_{\text{inner}}}$**：通过 softplus 确保 $\Delta > 0$（因为负的 $\Delta$ 在物理上无意义——时间不能倒流）

结果：每个 token 有自己专属的 $\Delta_t, B_t, C_t$。

### 为什么 A 保持不变？

A 是系统的"骨架"——它决定了记忆衰减的**模式**（哪些信息衰减快、哪些慢）。让 A 也输入依赖会：
1. 破坏并行扫描的可结合性（后面会解释）
2. 通过 $\Delta$ 的调节已经能达到足够的选择性：因为 $\bar{A} = e^{\Delta A}$，不同的 $\Delta$ 会让有效的 $\bar{A}$ 产生巨大变化

## 3.3 Δ 如何控制记忆——选择性遗忘

这是选择性机制最核心的方程：

$$h_t = e^{\Delta_t \cdot A} \cdot h_{t-1} + \Delta_t \cdot B_t \cdot x_t$$

其中 $A$ 的对角元素全部为负（$A = -\exp(A_{\log})$），这意味着：

### 推导——为什么 A 必须为负

假设 A 中某个通道的对角元素为 $a < 0$，步长为 $\Delta > 0$：

$$\bar{A} = e^{\Delta \cdot a}$$

因为 $\Delta > 0, a < 0$，所以 $\Delta \cdot a < 0$，因此：

$$0 < e^{\Delta \cdot a} < 1$$

即 $\bar{A}$ 是一个 0 到 1 之间的衰减因子。

### 两种极端情况

- **$\Delta$ 很小**（$\Delta \to 0$）：$\bar{A} = e^{\Delta a} \approx e^0 = I$（几乎不衰减）→ **保留旧状态**（强记忆）
- **$\Delta$ 很大**（$\Delta \to \infty$）：$\bar{A} = e^{\Delta a} \approx 0$（完全衰减）→ **遗忘旧状态**（重置记忆）

### 完整解读

因为 $\Delta_t$ 由输入 $x_t$ 动态生成：

- 遇到重要信息（如化学公式中的结构关键字"c1ccccc1"）：模型可以学习生成**小的** $\Delta$ → 保留旧上下文
- 遇到无关信息（如填充符号）：模型可以学习生成**大的** $\Delta$ → 重置状态，遗忘无关内容

这就是"**选择性**"的含义——模型学会了**根据内容决定记忆策略**。

## 3.4 B 和 C 的选择性作用

- **B 的选择性**：控制当前输入如何写入状态。不同的 $B_t$ 让不同 token 以不同方式影响"记忆"
- **C 的选择性**：控制如何从状态读取输出。不同的 $C_t$ 让模型在每一步可以选择读取记忆的不同方面

结合 B 和 C 的选择性，模型可以：
- 对某些 token 写入更多信息（通过大的 B 值）
- 对其他 token 写入较少（通过小的 B 值）
- 在需要时从状态中提取特定信息（通过调整 C）

## 3.5 选择性 SSM 与门控 RNN 的统一——Theorem 1

Mamba 论文的 **Theorem 1** 揭示了选择性 SSM 与经典门控 RNN（如 GRU/LSTM）之间的深刻联系：

**定理陈述**（Mamba 论文 Theorem 1）：当状态维度 $N=1$，$A=-1$，$B=1$，且 $\Delta_t = \text{Softplus}(\text{Linear}(x_t))$ 时，选择性 SSM 的递推变为：

$$h_t = (1 - g_t) \cdot h_{t-1} + g_t \cdot x_t$$

其中 $g_t = \sigma(\text{Linear}(x_t))$。

**推导过程**：

1. 当 $N=1$ 时，$A=-1$ 是标量，$B=1$ 也是标量
2. 离散化：$\bar{A}_t = e^{\Delta_t \cdot (-1)} = e^{-\Delta_t}$
3. 定义 $g_t = 1 - e^{-\Delta_t} = 1 - \bar{A}_t$
4. 则 $\bar{A}_t = 1 - g_t$
5. 简化离散化：$\bar{B}_t = \Delta_t \cdot 1 = \Delta_t$

但注意，当 $g_t$ 很小时（$\Delta_t$ 很小），$\Delta_t \approx g_t$（一阶泰勒展开：$e^{-\Delta} \approx 1 - \Delta$，所以 $g = 1 - e^{-\Delta} \approx \Delta$），所以：

> **更精确的说明**：如果使用完整的 ZOH 离散化（如 Mamba 论文），则 $\bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B = 1 - e^{-\Delta} = g_t$，等式**完全精确**（不需要泰勒近似）。因此 Theorem 1 在 Mamba 论文中是一个**精确的数学等价**，而非近似关系。

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t = (1 - g_t) h_{t-1} + g_t x_t$$

**这是经典的门控 RNN 形式**！$g_t$ 就是门控信号，控制"记忆旧状态"vs"写入新输入"的平衡：
- $g_t \approx 0$：$\Delta$ 小 → $h_t \approx h_{t-1}$（全记忆，忽略新输入）
- $g_t \approx 1$：$\Delta$ 大 → $h_t \approx x_t$（全遗忘，重写为新输入）

**含义**：Mamba 的选择性 SSM **不是凭空创造的**——它是连续时间 SSM 离散化的自然推广。经典的门控 RNN 是选择性 SSM 在最简单一维情况下的特例。Mamba 的创新在于将这个机制推广到多维状态空间，并利用并行扫描高效计算。

---

# 第四部分：矩阵 A 与 HiPPO 初始化

## 4.1 回到本源——A 为什么重要

状态的递推公式：
$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t$$

展开（假设 $\bar{A}$ 是标量以简化）：
$$h_0 = \bar{B}x_0$$
$$h_1 = \bar{A}\bar{B}x_0 + \bar{B}x_1$$
$$h_t = \bar{A}^t\bar{B}x_0 + \bar{A}^{t-1}\bar{B}x_1 + \cdots + \bar{B}x_t$$

注意：早期 token $x_0$ 对当前状态 $h_t$ 的贡献是 $\bar{A}^t \cdot \bar{B} \cdot x_0$。如果 $\bar{A} < 1$，这个贡献以指数级衰减。

**A 决定了记忆的"衰减速度"**——它控制模型对多远的历史仍有感知。

## 4.2 对角结构——Mamba 的关键效率设计

### 为什么 A 是对角的

一般 $N \times N$ 矩阵的矩阵乘法复杂度是 $O(N^2)$。但对角矩阵（只有对角线非零）的乘法是 $O(N)$：

$$\begin{bmatrix} a_1 & 0 & \cdots \\ 0 & a_2 & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix} \begin{bmatrix} h_1 \\ h_2 \\ \vdots \end{bmatrix} = \begin{bmatrix} a_1 h_1 \\ a_2 h_2 \\ \vdots \end{bmatrix}$$

在 Mamba 中，$A \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{state}}}$ 是对角的（沿 $d_{\text{state}}$ 维度），这意味着每个状态维度有独立的衰减速度，且计算极其高效。

**证明了为什么可以只用对角 A 而不损失表达能力**的是 DSS（Diagonal State Spaces, NeurIPS 2022）和 S4D 两篇论文：对角近似在性能上与结构化矩阵（如 S4 的 NPLR）相当。

### 在代码中的实现

```python
# 形状为 (d_inner, d_state) 的 A
A = torch.arange(1, d_state + 1)           # [1, 2, ..., 16]
A = A.repeat(d_inner, 1).contiguous()       # 每行重复
self.A_log = nn.Parameter(torch.log(A))      # 存 log(A)

# 前向传播时：
A = -torch.exp(self.A_log)                  # 确保 A 为负
```

## 4.3 长程依赖问题

**问题**：如果 A 是随机初始化的，模型很难学习到长程依赖。为什么？

回到展开式：$h_t = \bar{A}^t\bar{B}x_0 + ...$

如果 $\bar{A}$ 太小（如 0.5），经 $t=50$ 步后 $\bar{A}^{50} \approx 8.9 \times 10^{-16}$——$x_0$ 的信息完全消失，梯度也消失。

如果 $\bar{A}$ 接近 1，虽然长程信息保留，但模型对各时间步的区分能力变差——所有历史被"平均"了。

**核心难题**：如何初始化 A，使得状态能够**最优地压缩历史信息**？

## 4.4 HiPPO 理论——最优历史压缩

HiPPO = **Hi**gh-order **P**olynomial **P**rojection **O**perators（高阶多项式投影算子）

出自论文：_HiPPO: Recurrent Memory with Optimal Polynomial Projections_（Gu et al., NeurIPS 2020）

### 直觉

想象你有一条输入流 $f_0, f_1, f_2, ...$，你只能存储 N 个数字（"摘要"）。当新数据到达时，你需要更新摘要而无吃重新存储整个历史。

**HiPPO 的核心思想**：用**正交多项式**来压缩历史。

具体步骤：
1. 把离散输入看成连续函数在特定点的采样值：$f_k = f(k\Delta)$
2. 选择一组正交多项式基函数 $P_0, P_1, ..., P_{N-1}$（如勒让德多项式 Legendre polynomials 或拉盖尔多项式 Laguerre polynomials）
3. 用这 N 个基函数的系数 $c(t) = [c_0(t), c_1(t), ..., c_{N-1}(t)]$ 来表示函数 $f$ 在 $[0, t]$ 上的最佳 N 阶近似

### 正交多项式的含义

在近似理论中，任何"好"的函数都可以用正交多项式的线性组合来逼近：

$$f(x) \approx \sum_{n=0}^{N-1} c_n \cdot P_n(x)$$

多项式 $P_n$ 的"正交"是指：
$$\int_{a}^{b} P_n(x) P_m(x) \omega(x) dx = 0 \quad \text{当 } n \neq m$$

（$\omega(x)$ 是权重函数，不同选择对应不同多项式族）

### HiPPO 的关键推导

**HiPPO 的数学贡献**：发现了系数向量 $c(t)$ 的演化遵循一个线性 ODE（常微分方程）。以最常用的 HiPPO-LegS（Scaled Legendre）为例，其完整形式为：

$$\frac{d}{dt} m(t) = -\frac{1}{t} \cdot A_{\text{LegS}} \cdot m(t) + \frac{1}{t} \cdot B_{\text{LegS}} \cdot f(t)$$

其中 $A_{\text{LegS}}$ 和 $B_{\text{LegS}}$ 是由 Scaling Legendre 多项式唯一确定的矩阵常数：

$$(A_{\text{LegS}})_{nk} = \begin{cases} \sqrt{(2n+1)(2k+1)}, & n > k \\ n+1, & n = k \\ 0, & n < k \end{cases} \qquad (B_{\text{LegS}})_n = \sqrt{2n+1}$$

> **注意**：HiPPO-LegS 的原始形式有 $1/t$ 缩放因子，是时间变化的（time-varying）。但在 S4 和 Mamba 中，这个微分方程被重新解释为**线性时不变**（LTI）系统——即 $A = -A_{\text{LegS}}$ 固定不变，时间缩放因子被吸收进可学习的 $\Delta$ 参数中。这就是 Mamba 使用的 HiPPO 初始化。

经过这个简化，HiPPO ODE 变为：

$$\frac{dh(t)}{dt} = A \cdot h(t) + B \cdot x(t)$$

这个形式就和我们的 SSM 状态方程 $\frac{dh}{dt} = Ah + Bx$ **完全一致**！

而且 $A_{\text{HiPPO}}$ 和 $B_{\text{HiPPO}}$ 不是随机数——它们是确定的、由所选多项式族唯一决定的数学常数。

**这意味着**：如果用 HiPPO 矩阵初始化 SSM 的 A 和 B，那么模型在训练开始时就天然具备最优的历史压缩能力。反过来，如果用随机矩阵初始化 A，模型很难学到这种能力。

### 常用的 HiPPO 变体

| HiPPO 类型 | 基础多项式 | 特点 |
|-----------|-----------|------|
| HiPPO-LegS | Scaled Legendre | 最常用，对时间窗口均匀关注 |
| HiPPO-LegT | Truncated Legendre | 固定窗口 |
| HiPPO-LagT | Truncated Laguerre | 指数衰减窗口 |

### 本项目使用的初始化

在本项目的两个实现中，都使用 **S4D-Real** 初始化（属于 HiPPO 系列）：

在 `bimamba.py`（手写教学版）中，A 初始化为：
```python
A = torch.arange(1, d_state + 1)  # [1, 2, 3, ..., 16]
```
前向传播时：`A = -torch.exp(self.A_log)` → 实际值为 $[-1, -2, -3, ..., -16]$。

**这等同于 S4D-Real 初始化**：Mamba 论文推荐的实数对角初始化方案（基于 HiPPO 理论）。S4D-Real 定义为 $A_n = -(n+1), n=0,1,...,N-1$，即 $[-1, -2, ..., -N]$。

结果：$\bar{A} = e^{\Delta \cdot (-[1,2,3,...,16])}$ → 不同通道有不同的衰减速度（$e^{-\Delta}, e^{-2\Delta}, e^{-3\Delta}, ...$）。

在 `bimamba_with_mamba_ssm.py`（官方库版）中，Mamba2 内部同样使用 HiPPO/S4D 系列初始化（默认也是 S4D-Real）。两者的初始化在数学上等价。

> **Mamba 论文的消融实验**：S4D-Real 和随机初始化在语言建模上达到相同的困惑度（8.71），说明 S4D-Real 初始化虽然理论优美，但在实践中随机初始化也完全可以工作。

---

# 第五部分：并行计算——并行扫描算法

## 5.1 问题的产生

选择性机制引入后，参数 $\Delta_t, B_t, C_t$ 在每个时间步都不同，SSM 变成了**时变系统**。

**后果**：卷积表示失效！卷积表示要求 kernel 是固定的（时不变的 $K = [C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, ...]$）。现在每一步的 $\bar{A}_t = e^{\Delta_t A}$ 都不同，无法构建固定 kernel。

**我们必须用递推表示**：
$$h_t = \bar{A}_t \cdot h_{t-1} + \bar{B}_t \cdot x_t$$

但递推是**串行**的——计算 $h_{100}$ 需要先算 $h_{99}$，这需要 $h_{98}$……总计算步数 = 序列长度 L。对于长序列，训练会非常慢。

## 5.2 突破口——递推的可结合性（Associativity）

### 关键观察

仔细观察递推公式：
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$

我们可以把 $h_t$ 写成从 $h_0$ 开始的一个"累积"过程：

$$h_1 = \bar{A}_1 h_0 + \bar{B}_1 x_1$$
$$h_2 = \bar{A}_2 h_1 + \bar{B}_2 x_2 = \bar{A}_2(\bar{A}_1 h_0 + \bar{B}_1 x_1) + \bar{B}_2 x_2$$
$$= (\bar{A}_2\bar{A}_1) h_0 + (\bar{A}_2\bar{B}_1)x_1 + \bar{B}_2 x_2$$

这看起来像一个"前缀积+前缀和"的混合运算！

### 定义可结合的二元操作符

定义一个操作符 $\circ$，它把两个相邻的递推"合并"：

对于两步递推：
$$h_{i} = a_i h_{i-1} + b_i$$

其中 $a_i = \bar{A}_i$ 和 $b_i = \bar{B}_i x_i$。

现在考虑两步合并（用标量简化，矩阵情况类似）：

- 第一步：$h_1 = a_1 h_0 + b_1$
- 第二步：$h_2 = a_2 h_1 + b_2 = a_2(a_1 h_0 + b_1) + b_2 = (a_2 a_1) h_0 + (a_2 b_1 + b_2)$

定义操作符：$(a, b) \circ (a', b') = (a' \cdot a, \; a' \cdot b + b')$

**验证可结合性**：
$$[(a,b) \circ (a',b')] \circ (a'',b'') = (a'a, a'b+b') \circ (a'',b'') = (a''a'a, a''(a'b+b') + b'') = (a''a'a, a''a'b + a''b' + b'')$$

$$(a,b) \circ [(a',b') \circ (a'',b'')] = (a,b) \circ (a''a', a''b'+b'') = ((a''a')a, (a''a')b + (a''b'+b'')) = (a''a'a, a''a'b + a''b' + b'')$$

两者相等！✓ 所以 $\circ$ 是**可结合的**。

## 5.3 并行扫描（Parallel Scan / Parallel Prefix Sum）

### 经典类比：并行前缀和

计算前缀和：$S_i = x_0 + x_1 + \cdots + x_i$

串行方法需要 $O(L)$ 步（for 循环）。

并行扫描利用加法的可结合性 $(a+b)+c = a+(b+c)$：

1. **上扫（Up-sweep / Reduce Phase）**：像二叉树一样两两合并
2. **下扫（Down-sweep Phase）**：将结果传播回所有位置

结果：$O(\log L)$ 并行步数！（虽然总操作量仍是 $O(L)$，但可以在 GPU 上高度并行化）

### 应用于 SSM

同样的原理，由于 $\circ$ 操作符是可结合的，我们可以用并行扫描来计算整个 $h_0, h_1, ..., h_L$ 序列：

- 每个线程处理一对 $(a_i, b_i)$
- 多棵"合并树"并行计算
- 总并行深度：$O(\log L)$

## 5.4 硬件感知实现（Hardware-Aware Algorithm）

### GPU 存储层级

| 存储类型 | 容量 | 速度 | 用途 |
|---------|------|------|------|
| HBM（High Bandwidth Memory） | GB 级 | ~1-2 TB/s | 模型参数、大型张量 |
| SRAM（Static RAM） | MB 级（~20MB） | ~20 TB/s | 快速计算缓冲区 |

### 核心优化策略

Mamba 论文提出了三项关键技术：

**1. 核融合（Kernel Fusion）**

通常的计算流程：读取 → 离散化 → 写回 → 读取 → SSM扫描 → 写回 → 读取 → 乘 C → 写回...

每次"写回"都涉及从 SRAM 到 HBM 的慢速传输。

融合后：**一次性**读取输入到 SRAM → 在 SRAM 中完成**全部的**离散化 + 扫描 + 乘 C → **一次性**写回。减少了约 3-5 倍的 HBM 带宽压力。

**2. 并行扫描（如上所述）**

在 SRAM 中执行并行扫描，避免中间状态写入 HBM。

**3. 激活重计算（Recomputation / Gradient Checkpointing）**

前向传播时不保存所有中间状态（SRAM 太小），反向传播时重新计算需要的中间值。虽然额外多算了一遍前向，但避免了从慢速 HBM 读取——在 GPU 上，计算比读取快得多。

---

# 第六部分：BiMamba 完整架构

## 6.1 整体数据流

```
SMILES 字符串 ("C(=O)O")
    │
    ▼
MoleculeTokenizer.encode()
    → token_ids: [28, 0, 4, 34, 1]
    │
    ▼
token_embedding(token_ids)          → (B, L, d_model)
    + position_embedding(positions) → (B, L, d_model)
    │
    ▼
BiMambaEncoder
    ├─ Forward 分支:  x₀ → x₁ → x₂ → ... → xN   (从左到右)
    └─ Backward 分支: xN → x(N-1) → ... → x₀   (从右到左，用 torch.flip)
    融合: Gate · Forward + (1 - Gate) · Backward
    │
    ▼
Pooling (mean / max / cls)           → (B, d_model)
    │
    ▼
Classifier (MLP / Linear)            → (B, 1) 预测值
```

## 6.2 BiMambaBlock 内部结构详解

这是核心计算单元，对应 `bimamba.py` 中的 `BiMambaBlock.forward()`：

### Step 1: 输入投影与分路

```python
xz = self.in_proj(hidden_states)     # (B, L, d_model) → (B, L, 2*d_inner)
x, z = xz.chunk(2, dim=-1)           # 各 (B, L, d_inner)
```

- **x 通道**：进入 SSM 进行序列处理
- **z 通道**：用作门控信号

**为什么要分两路？**

这是一个"GLU 风格"的门控设计（借鉴 Gated Linear Unit）。x 负责处理序列信息，z 负责控制处理后的信息有多少能通过。

### Step 2: 局部 1D 卷积

```python
x = x.transpose(1, 2)                # (B, L, d_inner) → (B, d_inner, L)  # Conv1d 需要 channel 在 dim=1
x = self.conv1d(x)[:, :, :seqlen]    # 深度可分离卷积 → 截断 padding
x = x.transpose(1, 2)                # 还原为 (B, L, d_inner)
```

**为什么需要卷积？**

纯 SSM 在每个时间步是逐 token 独立处理的（除了通过隐状态传递的"记忆"）。1D 卷积给模型提供了一个"局部视野"——在 SSM 处理前，先让相邻 token 的信息混合。这在化学 SMILES 中尤为重要：相邻字符通常属于同一个化学基团（如"C=C"、"=O"）。

卷积使用的是 **depthwise convolution**（`groups=d_inner`）：每个通道独立卷积，参数量从 $O(d_{\text{inner}}^2)$ 降到 $O(d_{\text{inner}})$。

### Step 3: 激活

```python
x = self.activation(x)   # SiLU(x) = x · σ(x)
```

**SiLU（Sigmoid Linear Unit，也叫 Swish）**：
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

为什么不用 ReLU？SiLU 更平滑（处处可导），在深度 SSM 中梯度更加稳定。

### Step 4: SSM 参数生成

```python
x_dbl = self.x_proj(x)                    # (B, L, dt_rank + 2*d_state)
dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
dt = F.softplus(self.dt_proj(dt))         # 确保 Δ > 0
```

这一步体现了"选择性"——**dt, B, C 都由输入 x 动态生成**。

### Step 5: SSM 核心计算（selective_scan）

递推扫描（手写版是串行的，库版是并行扫描）：

```python
for t in range(seqlen):
    dA_t = exp(dt[:, t] * A)       # 当前时间步的离散化 A
    dB_t = dt[:, t] * B[:, t]      # 当前时间步的离散化 B
    h = dA_t * h + dB_t * x[:, t]  # 状态更新
    y_t = sum(h * C[:, t])         # 输出读取
```

### Step 6: 门控 + 残差

```python
y = y * F.silu(z)                          # 门控：z 控制 SSM 输出通过量
return self.out_proj(y) + hidden_states     # 投影 + 残差连接
```

**残差连接**（`+ hidden_states`）确保即使 SSM 的变化路径没有学到有用的特征，原始信息仍能通过。

**Git 类比**——理解残差连接最直观的方式：

```
你的代码仓库（原始 token 信息）
    │
    ├── git checkout -b try-new-idea     ← SSM 分支开始"尝试改进"
    │   （对原始信息做各种变换）
    │
    ├── git add . && git commit          ← SSM 的 out_proj(y)（"改进建议"）
    │
    ├── git merge try-new-idea           ← out_proj(y) + hidden_states
    │
    └── 结果：
        ├── 如果改进成功 → 两者融合，信息更丰富 ✓
        └── 如果改炸了 → git reset --hard main，原始信息毫发无损 ✓
```

残差连接就是模型在每一步自动做的 `git merge`：**SSM 在任何一步算错了，原始信息这条命脉不会断掉**。这就是为什么深层网络可以堆很多层而不退化。

更技术地理解：反向传播时，`out_proj(y) + hidden_states` 求导得 $\frac{\partial\text{ out}}{\partial x} + \mathbf{1}$。那个 `+1` 保证梯度至少为 1——即使 SSM 分支的梯度接近 0，信号也不会消失。所以残差连接**同时解决了梯度消失和模型退化两个问题**。

## 6.3 双向处理——BiMamba 的关键设计

### 为什么需要双向？

SMILES 是线性字符串，但分子的化学信息是非对称的：
- 前向看 "C(=O)" → 看到一个碳和后面的双键氧
- 后向看 ")O(=C" → 从另一个方向看到相同的结构

**双向处理让每个 token 都能同时感知到前后文的完整上下文**。

例如：苯环的 SMILES 写法是 `c1ccccc1`，数字 `1` 表示闭合位置。从前往后看，`c1` 是环的起点；从后往前看，`1` 是环的终点。双向处理让模型理解这是一个闭合环。

### 具体实现

```python
# 前向分支（左→右，正常顺序）
forward_hidden = hidden_states
for layer in self.forward_layers:
    forward_hidden = layer(forward_hidden)

# 后向分支（右→左，翻转序列处理后再翻转回来）
backward_hidden = torch.flip(hidden_states, [1])   # 翻转：x[0]↔x[N]
for layer in self.backward_layers:
    backward_hidden = layer(backward_hidden)
backward_hidden = torch.flip(backward_hidden, [1])  # 翻转回来，位置对齐
```

### 门控融合（Gated Fusion）

```python
combined = torch.cat([forward_hidden, backward_hidden], dim=-1)  # (B, L, 2*d_model)
gate = torch.sigmoid(self.fusion_gate(combined))                 # (B, L, 2*d_model)
gate_fwd, gate_bwd = gate.chunk(2, dim=-1)
fused = gate_fwd * forward_hidden + gate_bwd * backward_hidden
```

每个位置都学习一个动态权重：某些位置可能前向信息更重要（如官能团的开始），某些位置后向更重要（如环的闭合）。

## 6.4 Pooling 策略

从 $(B, L, d_{\text{model}})$ 到 $(B, d_{\text{model}})$——把序列信息压缩为单个向量。

| 策略 | 公式 | 优点 | 适用场景 |
|------|------|------|---------|
| **mean** | $\frac{1}{\sum \text{mask}} \sum_{i} \text{mask}_i \cdot h_i$ | 稳定，公平对待每个 token | 通用默认 |
| **max** | $\max_i h_i$（屏蔽 pad 为 $-\infty$） | 保留最显著特征 | 分类任务（如 HIV） |
| **cls** | $h_0$（CLS token 的输出） | 可学习 | 回归任务（如 Lipophilicity） |

## 6.5 分类头（Classifier）

**回归任务**（如溶解度预测）——使用 MLP 头：
```
d_model → Linear → d_model//2 → ReLU → Dropout → Linear → 1
```
两层设计给予更强的非线性拟合能力。

**分类任务**（如毒性判断）——使用单层线性：
```
d_model → Linear → num_labels
```
简单有效，避免过拟合。

---

# 第七部分：维度参考

| 变量 | 形状 | 含义 |
|------|------|------|
| `d_model` | scalar | 输入/输出嵌入维度（默认 256） |
| `d_inner` | `expand × d_model` | SSM 内部工作维度（默认 512，expand=2）。共 `d_inner` 个独立 SSM 通道，每个通道拥有自己的 `d_state` 维隐状态。输入在此空间内完成 SSM/卷积/激活后再投影回 `d_model` |
| `d_state` | scalar | SSM 状态维度（默认 16）。每个 SSM 通道的记忆容量——用 $d_{\text{state}}$ 个数字压缩历史信息。对应 HiPPO 中 Legendre 多项式的阶数 $N$ |
| `dt_rank` | `ceil(d_model / 16)` | Δ 的低秩维度（默认 16） |
| `B`, `L` | batch, sequence length | 批次大小和序列长度 |
| `A` | `(d_inner, d_state)` | 对角状态转移矩阵（负值） |
| `Δ` | `(B, L, d_inner)` | 每 token 的离散化步长 |
| `B` (矩阵) | `(B, L, d_state)` | 每 token 的输入矩阵 |
| `C` | `(B, L, d_state)` | 每 token 的输出矩阵 |
| `h` | `(B, d_inner, d_state)` | 隐状态 |
| `D` | `(d_inner,)` | Skip connection 系数 |

---

# 第八部分：数学总结——完整的前向传播

给定输入序列 $x_1, x_2, ..., x_L$（$d_{\text{inner}}$ 维），以下是 Mamba 块的完整计算：

**1. 参数生成（选择性）**：

$$\Delta_t = \text{Softplus}(\text{Linear}_\Delta(x_t)) \in \mathbb{R}^{d_{\text{inner}}}$$

$$B_t = \text{Linear}_B(x_t) \in \mathbb{R}^{d_{\text{state}}}$$

$$C_t = \text{Linear}_C(x_t) \in \mathbb{R}^{d_{\text{state}}}$$

**2. 离散化**：

$$\bar{A}_t = e^{\Delta_t \cdot A}, \quad A = -\exp(A_{\log}) \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{state}}}$$

$$\bar{B}_t = \Delta_t \cdot B_t$$

**3. 状态更新**：

$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t$$

（$\odot$ 表示逐元素乘法，因为 A 是对角的）

**4. 输出**：

$$y_t = \sum_{i=1}^{d_{\text{state}}} (h_t)_i \cdot (C_t)_i + D \cdot x_t$$

**5. 门控与残差**：

$$y_t^{\text{final}} = \text{Linear}_{\text{out}}(y_t \cdot \text{SiLU}(z_t)) + \text{input}_t$$

---

# 第九部分：参考文献

- **Mamba 原始论文**：Gu, A., & Dao, T. (2023). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. arXiv:2312.00752
- **Mamba-2**：Dao, T., & Gu, A. (2024). _Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality_. arXiv:2405.21060
- **HiPPO**：Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C. (2020). _HiPPO: Recurrent Memory with Optimal Polynomial Projections_. NeurIPS 2020. arXiv:2008.07669
- **LSSL**：Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., & Ré, C. (2021). _Combining Recurrent, Convolutional, and Continuous-time Models with Linear State Space Layers_. NeurIPS 2021. arXiv:2110.13985
- **S4**：Gu, A., Goel, K., & Ré, C. (2022). _Efficiently Modeling Long Sequences with Structured State Spaces_. ICLR 2022. arXiv:2111.00396
- **S4D / DSS**：Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). _On the Parameterization and Initialization of Diagonal State Space Models_. NeurIPS 2022. arXiv:2206.11893
- **控制理论基础**：Kalman, R. E. (1960). _A New Approach to Linear Filtering and Prediction Problems_. Journal of Basic Engineering, 82(1), 35-45.
- **并行扫描**：Blelloch, G. E. (1990). _Prefix Sums and Their Applications_. Technical Report CMU-CS-90-190.
- **GPU 优化参考**：NVIDIA. _Parallel Prefix Sum (Scan) with CUDA_. GPU Gems 3, Chapter 39.
