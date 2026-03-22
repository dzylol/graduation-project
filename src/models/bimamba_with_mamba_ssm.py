"""
Bi-Mamba 模型实现 - 分子性质预测

使用 mamba_ssm 库的优化实现，包含：
- 并行扫描算法（比循环快）
- HiPPO 矩阵初始化（最优的状态空间初始化）
- 融合的卷积和 SSM 操作

================================================================================
Mamba 状态空间模型矩阵计算详解
================================================================================

【选择性状态空间模型 (Selective SSM)】

Mamba 的核心是选择性状态空间模型，用以下方程描述：

    h_{t} = A @ h_{t-1} + B @ x_{t}      (状态更新)
    y_{t} = C @ h_{t}                      (输出)

其中：
    x_{t} : 输入向量 (d_inner 维)
    h_{t} : 隐藏状态向量 (d_state 维)
    y_{t} : 输出向量 (d_inner 维)
    A     : 状态转移矩阵 (d_inner, d_state)
    B     : 输入矩阵 (d_inner, d_state)
    C     : 输出矩阵 (d_inner, d_state)

【为什么需要离散化？】

状态空间模型诞生于控制理论，最初是连续时间形式：

    dh/dt = A @ h + B @ x      (连续微分方程)
     y    = C @ h                (连续输出)

计算机处理的是离散序列，需要把连续方程转为离散方程：

    连续: dh/dt = A @ h + B @ x
    离散: h_{t} = dA @ h_{t-1} + dB @ x_{t}

【离散化数学推导】

用欧拉方法近似导数（Δt → dt）：

    dh/dt ≈ (h_{t} - h_{t-1}) / dt

    代入连续方程:
    (h_{t} - h_{t-1}) / dt = A @ h_{t-1} + B @ x_{t}
    h_{t} - h_{t-1} = dt @ A @ h_{t-1} + dt @ B @ x_{t}
    h_{t} = (I + dt @ A) @ h_{t-1} + dt @ B @ x_{t}

但 Mamba 用指数离散化（更稳定）：

    dA = exp(dt @ A)               # 矩阵指数
    dB = dt @ B                     # 直接缩放

【代码对应】

    dt = softplus(dt_proj(x))      # softplus 保证 dt > 0
    dA = exp(dt * A)               # A 是预先初始化好的（HiPPO）
    dB = dt * B                    # B 也是输入相关的（选择性）

    h_new = dA * h_old + dB * x    # 状态更新

【选择性机制 (Selection)】

Mamba 的核心创新：dt, B, C 是输入相关的（由 x 通过 x_proj 生成）

    x_proj(x) → [dt_proj, B_proj, C_proj] → dt, B, C

这使得模型可以"选择性"地记住或遗忘信息。

【并行扫描 (Parallel Scan)】

原始 RNN 需要顺序计算（O(N) 串行），Mamba 用并行扫描实现 O(N) 并行：

    序列: x_{0}, x_{1}, x_{2}, ..., x_{N}

    并行扫描将递归转化为：
        - 计算各时间步的 dA, dB（可并行）
        - 用并行扫描累积状态（类似前缀和）

【HiPPO 矩阵初始化】

A 矩阵用 HiPPO (High-order Polynomial Projection Operator) 初始化：

    A[i,j] ∝ -(i+1) if i >= j else 0     # 或其他复杂形式

这使得初始状态可以很好地近似历史信息的压缩。

【完整前向传播流程】

    输入: x (B, L, d_model)

    1. in_proj: x → [x, z]  (d_model → d_inner * 2)
    2. conv1d:  x → local_x (因果卷积，捕获局部依赖)
    3. x_proj:  local_x → [dt, B, C]  (选择性)
    4. SSM:     并行扫描计算 y
    5. 门控:    y = y * silu(z)
    6. out_proj: y → output (d_inner → d_model)

【双向 Mamba】

Bi-Mamba 同时建模前向和后向：

    forward:  x_{0} → x_{1} → x_{2} → ... → x_{N}
    backward: x_{N} ← x_{N-1} ← x_{N-2} ← ... ← x_{0}

    最后用 fusion_gate 融合双向表示：

        combined = concat([forward, backward])
        gate = sigmoid(fusion_gate(combined))
        output = gate_f * forward + gate_b * backward

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from mamba_ssm import Mamba2


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
        d_model: int,  # 模型隐藏层维度（输入输出维度）
        d_state: int = 64,  # SSM 状态维度，通常 64 或 128
        d_conv: int = 4,  # 局部卷积核宽度，用于捕获局部依赖
        expand: int = 2,  # 块扩展因子，控制内部维度（d_inner = expand * d_model）
        use_fast_path: bool = True,  # 是否使用融合 kernel（需要 mamba_ssm 编译安装）
        layer_idx: Optional[int] = None,  # 层索引，用于调试和记录
        device: Optional[str] = None,  # 设备（cuda/mps/cpu）
        dtype: Optional[torch.dtype] = None,  # 数据类型（float32/float16）
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **factory_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            hidden_states: (B, L, D)

        Returns:
            output: (B, L, D)
        """
        return self.mamba(hidden_states)


class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器，同时从左到右和从右到左处理序列。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id, **factory_kwargs
        )
        self.position_embedding = nn.Embedding(
            max_seq_length, d_model, **factory_kwargs
        )

        self.forward_layers = self._make_layers(
            d_model, d_state, d_conv, expand, factory_kwargs
        )
        self.backward_layers = self._make_layers(
            d_model, d_state, d_conv, expand, factory_kwargs
        )

        self.norm = nn.LayerNorm(d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.fusion_gate = nn.Linear(d_model * 2, d_model * 2, **factory_kwargs)

    def _make_layers(self, d_model, d_state, d_conv, expand, factory_kwargs):
        return nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **factory_kwargs,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)

        Returns:
            hidden_states: (B, L, D)
        """
        batch_size, seq_len = input_ids.shape

        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        forward_hidden = hidden_states
        for layer in self.forward_layers:
            forward_hidden = layer(forward_hidden)

        backward_hidden = torch.flip(hidden_states, [1])
        for layer in self.backward_layers:
            backward_hidden = layer(backward_hidden)
        backward_hidden = torch.flip(backward_hidden, [1])

        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        gate_forward, gate_backward = gate.chunk(2, dim=-1)
        fused_hidden = gate_forward * forward_hidden + gate_backward * backward_hidden

        output = self.norm(fused_hidden)
        return output


class BiMambaForPropertyPrediction(nn.Module):
    """
    Bi-Mamba 分子性质预测模型。
    支持回归任务和分类任务。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_length: int = 512,
        num_labels: int = 1,
        task_type: str = "regression",
        pooling: str = "mean",
        dropout: float = 0.1,
        pad_token_id: int = 0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_labels = num_labels
        self.task_type = task_type
        self.pooling = pooling
        self.pad_token_id = pad_token_id

        self.encoder = BiMambaEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_token_id=pad_token_id,
            **factory_kwargs,
        )

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels, **factory_kwargs)

        if task_type == "regression":
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            labels: (B,) or (B, num_labels)

        Returns:
            logits, loss
        """
        batch_size = input_ids.shape[0]

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            input_ids = torch.cat(
                [
                    torch.full(
                        (batch_size, 1),
                        self.pad_token_id,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    input_ids,
                ],
                dim=1,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                        attention_mask,
                    ],
                    dim=1,
                )

        encoder_outputs = self.encoder(input_ids, attention_mask)

        if self.pooling == "mean":
            if attention_mask is not None:
                sum_embeddings = torch.sum(
                    encoder_outputs * attention_mask.unsqueeze(-1), dim=1
                )
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                pooled_output = sum_embeddings / sum_mask.clamp(min=1e-9)
            else:
                pooled_output = torch.mean(encoder_outputs, dim=1)

        elif self.pooling == "max":
            if attention_mask is not None:
                masked_embeddings = encoder_outputs.clone()
                masked_embeddings[attention_mask == 0] = -1e9
                pooled_output = torch.max(masked_embeddings, dim=1)[0]
            else:
                pooled_output = torch.max(encoder_outputs, dim=1)[0]

        elif self.pooling == "cls":
            pooled_output = encoder_outputs[:, 0]
        else:
            raise ValueError(f"未知的池化方法: {self.pooling}")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.num_labels == 1:
            logits = logits.squeeze(-1)

        loss = None
        if labels is not None:
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            loss = self.loss_fct(logits, labels)

        return logits, loss


def create_bimamba_model(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 4,
    task_type: str = "regression",
    num_labels: int = 1,
    **kwargs,
) -> BiMambaForPropertyPrediction:
    """工厂函数：创建 BiMamba 模型。"""
    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        task_type=task_type,
        num_labels=num_labels,
        **kwargs,
    )
