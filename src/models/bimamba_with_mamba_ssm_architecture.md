 # Bi-Mamba (mamba_ssm 版本) 架构解析

> 本文档解析 `src/models/bimamba_with_mamba_ssm.py` 的代码架构，对比 `bimamba.py`（手动实现）的差异。
>
> **学习路径**：先读 [§1 总览](#1-总览) 建立整体印象 → [§2 核心原理](#2-核心原理) 理解数学基础 → [§3 代码实现](#3-代码实现) 追踪数据流 → [§4 对比 bimambapy](#4-与-bimapmypy-的差异) 理解设计权衡。

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
│      (vocab_size → d_model)                            │
│   ② BiMambaEncoder                                      │
│      ├─ Forward: L× BiMambaBlock                        │
│      └─ Backward: L× BiMambaBlock + flip                │
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
| `d_model` | 256 | 模型维度（输入/输出） |
| `d_state` | 64 | SSM 状态维度（记忆容量） |
| `d_conv` | 4 | 因果卷积核宽度（局部上下文） |
| `expand` | 2 | 内部维度扩展因子，`d_inner = expand * d_model` |
| `n_layers` | 4 | 每方向的 Mamba 层数（总层数 = 2×n_layers） |
| `max_seq_length` | 512 | 最大序列长度（用于 Position Embedding） |

---

## 2. 核心原理

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
dA = exp(dt * A)                  # A 由 HiPPO 初始化
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

| 方式 | 计算方式 | 适用场景 |
|------|---------|---------|
| **mean** | 有效 token 的向量平均 | 通用，默认推荐 |
| **max** | 有效 token 的逐维度最大值 | 提取最显著特征 |
| **cls** | 专用 [CLS] token 的向量 | BERT 风格，需特殊训练 |

**Mean Pooling with Mask**：

```python
# 正确处理 padding
mask = attention_mask.unsqueeze(-1)              # (B, L, 1)
sum_embeds = sum(encoder_outputs * mask, dim=1)  # (B, d_model)
sum_mask  = sum(mask, dim=1).clamp(min=1e-9)      # (B, 1)，防除零
pooled = sum_embeds / sum_mask                     # (B, d_model)
```

**Max Pooling with Mask**：

```python
masked = encoder_outputs.clone()
masked[attention_mask == 0] = -1e9               # padding 位置设极小值
pooled = max(masked, dim=1)                      # 最大值不会被 padding 干扰
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

## 4. 与 bimamba.py 的差异

| 特性 | `bimamba.py`（手动实现） | `bimamba_with_mamba_ssm.py`（本文件） |
|------|------------------------|--------------------------------------|
| SSM 实现 | 手动实现离散化 + 并行扫描 | 使用 `mamba_ssm.Mamba2` 库封装 |
| 前向/后向权重 | **共享**（节省参数） | **独立**（各自有完整参数） |
| 融合方式 | concat / add / **gate** 三种 | 仅 **gate** 一种 |
| 默认 d_state | 16 | 64（更大记忆容量） |
| 适用场景 | 研究/调试（透明但慢） | 生产/部署（高效但黑盒） |

### 4.1 权重共享 vs 独立

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

### 4.2 融合模式

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

## 7. 设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 使用 Mamba2 库 | mamba_ssm.Mamba2 | 硬件感知融合 kernel，比手动实现快 5-10 倍 |
| 双向独立权重 | forward/backward 分离 | 各自有完整表达能力 |
| 融合用 Gate | sigmoid 门控 | 动态学习双向加权，比 add/concat 更灵活 |
| d_state=64 | 默认 64 | 比 bimamba.py 的 16 大 4 倍，记忆容量更强 |
| 支持 CLS pooling | 额外插入 [CLS] token | 与 BERT 风格兼容，方便微调 |
| 损失可选 | `labels=None` 时不计算损失 | 支持推理（不需要 labels）和训练两种用法 |

---

## 8. 参考

- **Mamba 论文**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- **mamba_ssm 库**: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- **Mamba 教程**: [`mamba.tutorial.md`](../../mamba.tutorial.md) — 从零理解 Mamba SSM 的完整指南
