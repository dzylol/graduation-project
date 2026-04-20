"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Bi-Mamba 手写 SSM 版 — 教学注释版                              ║
║   文件：bimamba.py                                                           ║
║   用途：手写实现 SSM，不依赖 mamba_ssm 库，适合理解原理、调试               ║
╚══════════════════════════════════════════════════════════════════════════════╝

【整体架构一览】

    SMILES 字符串
        │
        ▼
    MoleculeTokenizer.encode()   →  token_ids: [C, C, (, =, O, )] → [28, 28, 0, 4, 34, 1]
        │
        ▼
    BiMambaForPropertyPrediction.forward(input_ids)
        │
        ├─ token_embedding(input_ids)       → (B, L, d_model)
        ├─ position_embedding(position_ids) → (B, L, d_model)  加法融合
        │
        ▼
    BiMambaEncoder
        ├─ Forward 分支:  x₀→x₁→x₂→...→xN   (从左到右)
        └─ Backward 分支: xN←x(N-1)←...←x₀  (从右到左，用 torch.flip 实现)
        融合: gate * forward + (1-gate) * backward
        │
        ▼
    Pooling (mean / max / cls)    → (B, d_model)
        │
        ▼
    Classifier (MLP)              → (B, 1)  回归值 或 分类 logit


════════════════════════════════════════════════════════════════
【SSM 核心数学：状态方程】
════════════════════════════════════════════════════════════════

状态空间模型来自控制理论，描述一个动态系统：

    连续形式:
        dh/dt = A · h + B · x     ← h 是系统"记忆"，x 是当前输入
        y = C · h                  ← 从记忆里读出输出

    离散化（Mamba 用指数方法）：
        dA = exp(dt · A)           ← A 决定"遗忘速度"
        dB = dt · B                ← B 决定"怎么写入记忆"
        h_new = dA · h_old + dB · x
        y = C · h_new

    "选择性"：dt, B, C 都由输入 x 动态生成，
    所以模型可以"选择"记住或遗忘某些信息。
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# BiMambaBlock — 核心 SSM 块
# ═══════════════════════════════════════════════════════════

class BiMambaBlock(nn.Module):
    """
    单个 Mamba 块：实现了完整的选择性状态空间模型。

    内部数据流：
        hidden_states (B, L, d_model)
            │
            ├─ in_proj → [x, z]   (d_model → d_inner * 2，然后 chunk 分成两半)
            │
            ├─ conv1d(x)          (局部卷积，捕获相邻 token 的关系)
            │
            ├─ SiLU(x)            (激活函数)
            │
            ├─ ssm(x)             (核心 SSM 计算：参见 selective_scan)
            │
            ├─ y = ssm_out * SiLU(z)   (门控：z 控制信息通过量)
            │
            └─ out_proj(y) + hidden_states   (线性变换 + 残差连接)
    """

    def __init__(
        self,
        d_model: int,          # 输入/输出维度
        d_state: int = 16,     # SSM 状态维度（h 的维度），越大记忆越强
        d_conv: int = 4,       # 局部卷积核大小
        expand: int = 2,       # 内部扩展倍数，d_inner = expand * d_model
        dt_rank: Union[int, str] = "auto",  # dt 的低秩分解维度
        dt_min: float = 0.001,  # dt 初始化最小值
        dt_max: float = 0.1,    # dt 初始化最大值
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # d_inner：块内部实际工作的维度（比 d_model 更大，增强表达力）
        self.d_inner = int(expand * d_model)  # 例：d_model=256, expand=2 → d_inner=512

        # dt_rank：时间步长 dt 的低秩维度（节省参数量）
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.layer_idx = layer_idx

        # ── 1. 输入投影 ──────────────────────────────────────────
        # 把输入映射到 2 * d_inner，然后 chunk 成 x 和 z 两部分
        # x → 进入 SSM 计算
        # z → 门控信号，用 SiLU(z) 控制信息流量
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # ── 2. 局部 1D 卷积 ──────────────────────────────────────
        # depthwise conv（groups=d_inner），每个通道独立卷积，参数量少
        # 作用：在 SSM 处理之前，先捕获局部上下文（类似滑动窗口）
        # padding=d_conv-1 保证输出和输入等长（因果填充）
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,   # depthwise：每通道独立
            padding=d_conv - 1,    # 左侧补零保证因果性
            **factory_kwargs,
        )

        self.activation = nn.SiLU()  # Swish 激活函数，比 ReLU 更平滑

        # ── 3. SSM 参数投影 ───────────────────────────────────────
        # 从 x 动态生成 dt, B, C（这是"选择性"的关键）
        # 输出维度 = dt_rank + d_state * 2（分别对应 dt, B, C）
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs
        )

        # dt 的升维投影：从 dt_rank 升到 d_inner（每个通道各自的时间步长）
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # dt 权重初始化
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt 偏置初始化：用 softplus 反函数保证初始 dt 均匀分布在 [dt_min, dt_max]
        self._init_dt_proj_bias(dt_min, dt_max, dt_init_floor, factory_kwargs)

        # ── 4. SSM 状态矩阵 A（对角形式）────────────────────────────
        # A 控制状态的"衰减速度"：A[i] < 0 → 状态随时间衰减
        # 用 log 存储确保 A 始终为负（exp(A_log) > 0，取负后得到负 A）
        # 初始化为 log(1), log(2), ..., log(d_state) 重复 d_inner 次
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A.repeat(self.d_inner, 1).contiguous()  # shape: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))     # 存 log(A)，前向时取 -exp(A_log)

        # D：skip connection 系数（SSM 输出再加上原始输入 * D）
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))

        # ── 5. 输出投影 ────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def _init_dt_proj_bias(self, dt_min, dt_max, dt_init_floor, factory_kwargs):
        """
        把 dt 偏置初始化为均匀分布在 [dt_min, dt_max] 内。
        用 softplus 反函数（inv_softplus）确保经过 softplus 激活后
        dt 的初始值符合预期范围。
        """
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # inv_softplus(x) = x + log(1 - exp(-x)) ≈ log(exp(x) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # 标记不要被 reinit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            hidden_states: (B, L, d_model)  B=batch, L=序列长度

        Returns:
            output: (B, L, d_model)  与输入同形状
        """
        batch, seqlen, _ = hidden_states.shape

        # Step 1: 输入投影 → 分成 x（进入SSM）和 z（门控信号）
        xz = self.in_proj(hidden_states)            # (B, L, d_inner*2)
        x, z = xz.chunk(2, dim=-1)                  # 各 (B, L, d_inner)

        # Step 2: 1D 卷积（需要 (B, C, L) 格式）
        x = x.transpose(1, 2)                       # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seqlen]           # 截断 padding，保持长度=L
        x = x.transpose(1, 2)                       # (B, L, d_inner) 还原

        # Step 3: 激活
        x = self.activation(x)                      # SiLU(x)

        # Step 4: SSM 核心计算
        y = self.ssm(x)                              # (B, L, d_inner)

        # Step 5: 门控（z 控制信息流量）
        y = y * F.silu(z)                            # 逐元素相乘

        # Step 6: 输出投影 + 残差连接
        return self.out_proj(y) + hidden_states      # 残差保留原始信息

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        SSM 主计算：从输入 x 动态生成 dt, B, C，然后跑 selective_scan。

        Args:
            x: (B, L, d_inner)

        Returns:
            y: (B, L, d_inner)
        """
        # 从 x 动态生成 dt_raw, B, C（这是"选择性"的核心）
        x_dbl = self.x_proj(x)                       # (B, L, dt_rank + d_state*2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt 升维并用 softplus 保证正值（dt > 0 是离散化的数学要求）
        dt = F.softplus(self.dt_proj(dt))             # (B, L, d_inner)

        return self.selective_scan(x, dt, B, C)

    def _discretize(self, dt, B, A):
        """
        零阶保持（ZOH）离散化：连续 SSM → 离散 SSM。

        连续形式: dh/dt = A·h + B·x
        离散形式: h_t = dA·h_{t-1} + dB·x_t

        其中：
            dA = exp(dt · A)
            dB = dt · B

        Args:
            dt: (B, d_inner)      当前时间步的步长
            B:  (B, d_state)      当前时间步的输入矩阵
            A:  (d_inner, d_state) 状态转移矩阵（负值）

        Returns:
            dA: (B, d_inner, d_state)
            dB: (B, d_inner, d_state)
        """
        dt_clamped = torch.clamp(dt, min=-10, max=10)
        # dt * A：每个通道用自己的 dt 缩放 A
        # A.unsqueeze(0): (1, d_inner, d_state)
        dA = torch.exp(torch.clamp(dt_clamped.unsqueeze(-1) * A.unsqueeze(0), min=-50, max=50))
        # dB = dt * B，B 是 (B, d_state)，需要广播到 (B, d_inner, d_state)
        dB = dt_clamped.unsqueeze(-1) * B.unsqueeze(1)
        return dA, dB

    def _single_step(self, h, dt_t, B_t, C_t, x_t, A):
        """
        单步 SSM 更新：
            h_new = dA · h + dB · x    (状态更新)
            y = C · h_new              (输出读取)

        Args:
            h:    (B, d_inner, d_state)  上一步隐藏状态
            dt_t: (B, d_inner)           当前步的 dt
            B_t:  (B, d_state)           当前步的 B
            C_t:  (B, d_state)           当前步的 C
            x_t:  (B, d_inner)           当前输入
            A:    (d_inner, d_state)     状态矩阵

        Returns:
            y_t:  (B, d_inner)           当前步输出
            h_new:(B, d_inner, d_state)  更新后的状态
        """
        dA, dB = self._discretize(dt_t, B_t, A)
        x_t_clamped = torch.clamp(x_t, min=-10, max=10)
        # 状态更新：h_new = dA * h_old + dB * x
        # dB * x: (B, d_inner, d_state) * (B, d_inner, 1)
        h_new = torch.clamp(dA * h + dB * x_t_clamped.unsqueeze(-1), min=-100, max=100)
        # 输出：y = sum(h * C, dim=d_state)  → (B, d_inner)
        y_t = torch.sum(h_new * C_t.unsqueeze(1), dim=2)
        return y_t, h_new

    def selective_scan(self, x, dt, B, C):
        """
        顺序扫描实现（串行，速度慢，但逻辑清晰）。
        mamba_ssm 版本用并行扫描（速度快但代码复杂）。

        ⚠️ 这里是 O(N) 串行循环，是手写版比 mamba_ssm 慢的主要原因。

        Args:
            x:  (B, L, d_inner)
            dt: (B, L, d_inner)
            B:  (B, L, d_state)
            C:  (B, L, d_state)

        Returns:
            y: (B, L, d_inner)
        """
        batch, seqlen, dim = x.shape
        A = -torch.exp(self.A_log)   # A 必须为负（保证系统稳定），shape: (d_inner, d_state)
        D = self.D                   # skip connection 系数，shape: (d_inner,)

        # 初始隐藏状态全零
        h = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        # 逐时间步处理（这里是关键的串行循环）
        for t in range(seqlen):
            y_t, h = self._single_step(
                h,
                dt[:, t, :],   # 第 t 步的 dt: (B, d_inner)
                B[:, t, :],    # 第 t 步的 B:  (B, d_state)
                C[:, t, :],    # 第 t 步的 C:  (B, d_state)
                x[:, t, :],    # 第 t 步的输入:(B, d_inner)
                A,
            )
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)       # (B, L, d_inner)
        # skip connection：y += x * D
        # D.unsqueeze(0) → (1, d_inner) 广播到 (B, L, d_inner)
        return y + x * D.unsqueeze(0)


# ═══════════════════════════════════════════════════════════
# BiMambaEncoder — 双向编码器
# ═══════════════════════════════════════════════════════════

class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器。

    ┌──────────────────────────────────────────────────────┐
    │  token_embedding + position_embedding → hidden       │
    │                                                      │
    │  Forward 分支:                                       │
    │  hidden → Layer0 → Layer1 → ... → forward_hidden    │
    │                                                      │
    │  Backward 分支（序列翻转后处理，再翻转回来）:         │
    │  flip(hidden) → Layer0 → ... → flip(result)         │
    │  = backward_hidden                                   │
    │                                                      │
    │  Fusion Gate:                                        │
    │  combined = cat(forward, backward)                   │
    │  gate = sigmoid(W · combined)                        │
    │  output = gate_f * forward + gate_b * backward       │
    └──────────────────────────────────────────────────────┘

    为什么需要双向？
    - SMILES 是线性字符串，前向看到"C(=O)"，后向看到")O(=C"
    - 双向处理让每个位置都能感知到前后文的完整上下文
    - 例如：判断一个苯环需要从两个方向都看到 'c1ccccc1'
    """

    def __init__(
        self,
        vocab_size: int,      # 词表大小（默认 45 个 SMILES token + 4 个特殊 token）
        d_model: int = 256,   # 模型维度
        n_layers: int = 4,    # 每方向的层数
        d_state: int = 16,    # SSM 状态维度
        d_conv: int = 4,      # 卷积核大小
        expand: int = 2,      # 扩展因子
        max_seq_length: int = 512,  # 最大序列长度（用于 position embedding）
        dropout: float = 0.1,
        pad_token_id: int = 0,      # padding token id（embedding 时该位置梯度=0）
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        # Token Embedding：把 token id 映射到向量
        # padding_idx=pad_token_id：pad 位置的 embedding 梯度为 0
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id, **factory_kwargs
        )

        # Position Embedding：为每个位置加一个可学习的位置向量
        # 注意：max_seq_length 限制了序列长度上限（本项目为 512）
        self.position_embedding = nn.Embedding(max_seq_length, d_model, **factory_kwargs)

        # 前向 Mamba 层（左→右）
        self.forward_layers = nn.ModuleList(
            [BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                          expand=expand, **factory_kwargs)
             for _ in range(n_layers)]
        )

        # 后向 Mamba 层（右→左，权重独立）
        self.backward_layers = nn.ModuleList(
            [BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                          expand=expand, **factory_kwargs)
             for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        # Fusion Gate：用门控方式融合双向表示
        # 输入是 concat(forward, backward) → 2*d_model
        # 输出也是 2*d_model，chunk 后各自作为 gate 权重
        self.fusion_gate = nn.Linear(d_model * 2, d_model * 2, **factory_kwargs)

    def forward(self, input_ids, attention_mask=None, cls_token=None):
        """
        Args:
            input_ids:      (B, L)       token id 序列
            attention_mask: (B, L)       1=真实token, 0=padding
            cls_token:      (B, 1, d)    可选，CLS pooling 专用

        Returns:
            hidden_states: (B, L, d_model)  或 (B, L+1, d_model) 如有 cls_token
        """
        batch_size, seq_len = input_ids.shape

        # 生成位置 id：[0, 1, 2, ..., seq_len-1]，扩展到 batch 维度
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )  # (B, L)

        # Embedding：token 向量 + 位置向量（逐元素相加）
        hidden_states = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(position_ids)
        )  # (B, L, d_model)

        # 如果使用 CLS pooling，在序列最前面插入 CLS token
        if cls_token is not None:
            hidden_states = torch.cat([cls_token, hidden_states], dim=1)  # (B, L+1, d_model)

        # 把 padding 位置清零（防止 padding 干扰计算）
        if attention_mask is not None:
            if cls_token is not None:
                # CLS token 对应的 mask = 1
                attention_mask = torch.cat(
                    [torch.ones((batch_size, 1), dtype=attention_mask.dtype,
                                device=attention_mask.device),
                     attention_mask], dim=1
                )
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        # ── 前向分支 ─────────────────────────────────────────────
        forward_hidden = hidden_states
        for layer in self.forward_layers:
            forward_hidden = layer(forward_hidden)   # 依次经过每个 Mamba 层

        # ── 后向分支 ─────────────────────────────────────────────
        # torch.flip(x, [1])：把序列维度翻转，即 x[0]↔x[N], x[1]↔x[N-1]...
        backward_hidden = torch.flip(hidden_states, [1])
        for layer in self.backward_layers:
            backward_hidden = layer(backward_hidden)
        # 翻转回来，使位置对齐（第 i 个位置对应原始序列第 i 个 token）
        backward_hidden = torch.flip(backward_hidden, [1])

        # ── 双向融合（Gated Fusion）────────────────────────────────
        # 把前向和后向拼接，用可学习的 gate 决定各自的权重
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)  # (B, L, 2*d_model)
        gate = torch.sigmoid(self.fusion_gate(combined))                  # (B, L, 2*d_model)
        gate_fwd, gate_bwd = gate.chunk(2, dim=-1)   # 各 (B, L, d_model)
        # 加权融合（注意：gate_fwd + gate_bwd ≠ 1，各自独立）
        fused_hidden = gate_fwd * forward_hidden + gate_bwd * backward_hidden

        return self.norm(fused_hidden)  # LayerNorm 稳定训练


# ═══════════════════════════════════════════════════════════
# BiMambaForPropertyPrediction — 完整预测模型
# ═══════════════════════════════════════════════════════════

class BiMambaForPropertyPrediction(nn.Module):
    """
    完整的分子性质预测模型。

    架构：
        input_ids → Encoder → Pooling → Dropout → Classifier → logits

    支持三种 pooling 策略（对答辩影响很大，记住）：
    ┌──────────┬──────────────────────────────────────────────────────┐
    │ mean     │ 所有 token 的向量取平均（带 mask）                     │
    │          │ 优点：稳定，对每个 token 公平                          │
    │          │ 适合：通用默认选项                                     │
    ├──────────┼──────────────────────────────────────────────────────┤
    │ max      │ 每维取最大值（带 mask，pad 位置设为 -1e9）              │
    │          │ 优点：保留最显著特征，对分类任务友好                    │
    │          │ 适合：HIV 分类（本项目最佳 AUC=0.787）                 │
    ├──────────┼──────────────────────────────────────────────────────┤
    │ cls      │ 在序列开头插入可学习的 [CLS] token，取其输出            │
    │          │ 优点：全局表示，参数可学习                              │
    │          │ 适合：Lipophilicity 回归（本项目最佳 RMSE=1.19）        │
    │          │ 风险：HIV 分类失败（AUC=0.498），不稳定                 │
    └──────────┴──────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_length: int = 512,
        num_labels: int = 1,         # 回归=1，多分类>1
        task_type: str = "regression",  # "regression" 或 "classification"
        pooling: str = "mean",          # "mean", "max", "cls"
        dropout: float = 0.1,
        pad_token_id: int = 0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_labels = num_labels
        self.task_type = task_type
        self.pooling = pooling
        self.pad_token_id = pad_token_id

        # 编码器（主体）
        self.encoder = BiMambaEncoder(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            max_seq_length=max_seq_length, dropout=dropout,
            pad_token_id=pad_token_id, **factory_kwargs,
        )

        # CLS pooling 专用：可学习的 CLS token 向量
        # shape: (1, 1, d_model)，在 forward 中会 expand 到 (B, 1, d_model)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs))

        self.dropout = nn.Dropout(dropout)

        # 分类头（prediction head）
        # 回归任务用 MLP（两层），增强非线性拟合能力
        # 分类任务用单层线性（简单有效）
        if task_type == "regression":
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2, **factory_kwargs),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_labels, **factory_kwargs),
            )
        else:
            self.classifier = nn.Linear(d_model, num_labels, **factory_kwargs)

        # 损失函数
        # MSELoss：回归任务，最小化预测值与真实值的均方误差
        # BCEWithLogitsLoss：分类任务，数值稳定的二元交叉熵
        self.loss_fct = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Args:
            input_ids:      (B, L)   token id
            attention_mask: (B, L)   1=有效, 0=pad
            labels:         (B,)     真实标签（训练时传入，推理时可不传）

        Returns:
            logits: (B,) 或 (B, num_labels)
            loss:   scalar（仅在 labels 不为 None 时返回）
        """
        batch_size = input_ids.shape[0]

        # CLS pooling：准备 CLS token（expand 到当前 batch 大小）
        cls_token = self.cls_token.expand(batch_size, -1, -1) if self.pooling == "cls" else None

        # 编码
        encoder_outputs = self.encoder(input_ids, attention_mask, cls_token=cls_token)
        # encoder_outputs: (B, L, d_model) 或 (B, L+1, d_model) 如有 CLS

        # ── Pooling ───────────────────────────────────────────────
        if self.pooling == "mean":
            # 只对真实 token（非 pad）取平均
            if attention_mask is not None:
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1e-9)
                pooled_output = (
                    torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1) / sum_mask
                )
                # attention_mask.unsqueeze(-1): (B, L, 1) → 广播到 (B, L, d_model)
            else:
                pooled_output = torch.mean(encoder_outputs, dim=1)

        elif self.pooling == "max":
            # pad 位置设为极小值（-1e9），确保 max 不会选到 pad
            if attention_mask is not None:
                masked = encoder_outputs.clone()
                masked[attention_mask == 0] = -1e9  # pad 位置屏蔽
                pooled_output = torch.max(masked, dim=1)[0]  # [0] 取值，[1] 取索引
            else:
                pooled_output = torch.max(encoder_outputs, dim=1)[0]

        elif self.pooling == "cls":
            # 直接取序列第 0 位（即 CLS token）的输出
            pooled_output = encoder_outputs[:, 0]  # (B, d_model)

        else:
            raise ValueError(f"pooling must be 'mean', 'max', or 'cls', got {self.pooling}")

        # Dropout → Classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)   # (B, num_labels)

        # 单标签回归：squeeze 掉最后一维
        if self.num_labels == 1:
            logits = logits.squeeze(-1)   # (B,)

        # 计算损失（训练时）
        loss = None
        if labels is not None:
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            loss = self.loss_fct(logits, labels)

        return logits, loss


# ═══════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════

def create_bimamba_model(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 4,
    task_type: str = "regression",
    num_labels: int = 1,
    **kwargs,
) -> BiMambaForPropertyPrediction:
    """
    快速创建模型的工厂函数。

    使用示例：
        model = create_bimamba_model(
            vocab_size=45,
            d_model=256,
            n_layers=4,
            task_type="classification",
            num_labels=1,
            pooling="max",       # HIV 分类推荐
            d_state=16,
        )
    """
    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        task_type=task_type,
        num_labels=num_labels,
        **kwargs,
    )
