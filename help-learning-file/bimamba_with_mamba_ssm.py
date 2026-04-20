"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        bimamba_with_mamba_ssm.py — 教学注释版                               ║
║   用途：用 mamba_ssm 库的优化版本，适合正式训练                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

【与 bimamba.py 的核心区别】

    bimamba.py（手写 SSM）                bimamba_with_mamba_ssm.py（库版）
    ─────────────────────────────────    ──────────────────────────────────
    SSM 核心：Python for 循环（慢）        SSM 核心：Mamba2 CUDA kernel（快）
    离散化：手写 exp(dt·A), dt·B          离散化：库内部处理，不可见
    A 初始化：简单 arange(1..d_state)      A 初始化：HiPPO（数学最优）
    BiMambaBlock.forward：手动拼装        BiMambaBlock.forward：直接调 self.mamba(x)
    双向融合：bimamba.py 内部处理          双向融合：BiMambaEncoder 处理（逻辑相同）
    pool：在 BiMambaForPropertyPrediction  pool：在 BiMambaForPropertyPrediction


【SSM 数学回顾（答辩常考）】

连续状态空间方程：
    dh/dt = A·h + B·x      h 是"记忆"，x 是输入
    y = C·h                 从记忆里读输出

离散化（ZOH 方法）：
    dA = exp(Δ·A)          Δ（delta）= dt，控制遗忘速度
    dB = Δ·B               B 控制怎么写入记忆
    h_t = dA·h_{t-1} + dB·x_t
    y_t = C·h_t

"选择性"（Selective）：
    Δ, B, C 都由输入 x 动态生成（x_proj → [Δ, B, C]）
    不同输入产生不同的记忆策略 → 模型可以"选择性"遗忘

HiPPO 矩阵（mamba_ssm 版特有）：
    A 用 HiPPO（High-order Polynomial Projection）初始化
    直觉：初始 A 能最优地压缩过去历史 → 比 arange 初始化收敛更快

并行扫描（mamba_ssm 版特有）：
    bimamba.py 是 for t in range(seqlen) 串行循环 → O(N) 但慢
    mamba_ssm 用并行前缀扫描（类似并行前缀和）→ O(log N) 深度，GPU 利用率高
    这是 mamba_ssm 版本比手写版快 1.5-2x 的主要原因
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════
# BiMambaBlock（库版）
# ═══════════════════════════════════════════════════════════

class BiMambaBlock(nn.Module):
    """
    单个 Mamba 块（基于 mamba_ssm.Mamba2）。

    与 bimamba.py 中 BiMambaBlock 的对比：
    ┌─────────────────────┬──────────────────────────────────────┐
    │ bimamba.py          │ 本文件                               │
    ├─────────────────────┼──────────────────────────────────────┤
    │ 手写 in_proj        │ Mamba2 内部处理                      │
    │ 手写 conv1d         │ Mamba2 内部处理                      │
    │ 手写 x_proj/dt_proj │ Mamba2 内部处理                      │
    │ 手写 A_log, D 参数  │ Mamba2 内部（HiPPO初始化）           │
    │ 手写 selective_scan │ Mamba2 内部（并行扫描）              │
    │ 手写 out_proj       │ Mamba2 内部处理                      │
    │ 残差：手动 +hidden  │ Mamba2 内部有残差                    │
    └─────────────────────┴──────────────────────────────────────┘

    结论：本文件的 BiMambaBlock 只是一个薄包装，
    所有计算都委托给 Mamba2，代码极简但黑箱化。
    """

    def __init__(
        self,
        d_model: int,           # 模型维度（输入输出相同）
        d_state: int = 16,      # SSM 状态维度（h 的维度）
        d_conv: int = 4,        # 局部卷积核宽度
        expand: int = 2,        # 内部扩展倍数，d_inner = expand * d_model
        use_fast_path: bool = True,   # 是否用 CUDA 融合 kernel
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
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # 引入 mamba_ssm 的 Mamba2（如未安装：pip install mamba-ssm）
        from mamba_ssm import Mamba2

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **factory_kwargs,
        )
        # Mamba2 内部已经包含：
        #   - in_proj, conv1d, x_proj, dt_proj
        #   - A_log（HiPPO初始化）, D 参数
        #   - selective_scan（并行扫描）
        #   - out_proj, 残差连接

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播（极简包装）。

        Args:
            hidden_states: (B, L, d_model)

        Returns:
            output: (B, L, d_model)

        注意：Mamba2 接收 (B, L, D)，与 bimamba.py 中需要
        transpose 到 (B, D, L) 处理 conv1d 不同，这里更直观。
        """
        return self.mamba(hidden_states)


# ═══════════════════════════════════════════════════════════
# BiMambaEncoder（双向编码器）
# ═══════════════════════════════════════════════════════════

class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器。

    与 bimamba.py 中 BiMambaEncoder 的区别：
    - 多了 input_proj（d_model → d_mamba 投影，允许两者不同）
    - 多了 output_proj（d_mamba → d_model 还原）
    - bidirectional 参数：可选单向模式（uni-directional 消融实验用）
    - forward/backward 使用独立权重（bimamba.py 也是独立的，相同）

    前向传播流程：
        input_ids → token_embedding + position_embedding
            → input_proj（可选，d_model→d_mamba）
            → CLS token 拼接（cls pooling 用）
            → attention_mask 清零 padding
            → Forward Mamba 层 × n_layers
            → Backward Mamba 层 × n_layers（flip → 处理 → flip）
            → Fusion Gate 融合双向
            → output_proj + LayerNorm
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,       # token embedding 维度
        d_mamba: int = 256,       # Mamba 内部处理维度（通常等于 d_model）
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bidirectional: bool = True,  # False = 单向（消融实验用）
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_mamba = d_mamba
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Token Embedding（padding_idx 使 pad 位置梯度=0）
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id, **factory_kwargs
        )
        # Position Embedding（可学习，最大长度 max_seq_length）
        self.position_embedding = nn.Embedding(max_seq_length, d_model, **factory_kwargs)

        # 输入投影（d_model → d_mamba，若相等则是恒等映射）
        self.input_proj = nn.Linear(d_model, d_mamba, bias=False, **factory_kwargs)

        self.dropout = nn.Dropout(dropout)

        # 前向 Mamba 层（左→右）
        self.forward_layers = self._make_layers(d_mamba, d_state, d_conv, expand, factory_kwargs)

        if bidirectional:
            # 后向 Mamba 层（右→左，独立权重）
            self.backward_layers = self._make_layers(d_mamba, d_state, d_conv, expand, factory_kwargs)
            # 融合门：concat(forward, backward) → gate → 加权求和
            # 输入 2*d_mamba，输出 2*d_mamba（chunk 后各作为一个方向的权重）
            self.fusion_gate = nn.Linear(d_mamba * 2, d_mamba * 2, **factory_kwargs)

        self.norm = nn.LayerNorm(d_mamba, **factory_kwargs)
        # 输出投影（d_mamba → d_model，还原维度）
        self.output_proj = nn.Linear(d_mamba, d_model, bias=False, **factory_kwargs)

    def _make_layers(self, d_model, d_state, d_conv, expand, factory_kwargs):
        """创建一组 Mamba 层的辅助函数（避免重复代码）。"""
        return nn.ModuleList([
            BiMambaBlock(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, **factory_kwargs,
            )
            for _ in range(self.n_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
        cls_token: Optional[torch.Tensor] = None,       # (B, 1, d_model)
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L)       token id 序列
            attention_mask: (B, L)       1=真实 token, 0=padding
            cls_token:      (B, 1, d)    CLS pooling 专用，prepend 到序列前

        Returns:
            (B, L, d_model) 或 (B, L+1, d_model) 如有 cls_token
        """
        batch_size, seq_len = input_ids.shape

        # Position id：[0, 1, 2, ..., L-1]，expand 到 batch
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0).expand(batch_size, -1)
        )

        # Embedding：token 向量 + 位置向量（逐元素加）
        token_embeds = self.token_embedding(input_ids)       # (B, L, d_model)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)

        # 投影到 Mamba 维度（d_model → d_mamba）
        hidden_states = self.input_proj(hidden_states)        # (B, L, d_mamba)

        # CLS pooling：在序列前插入 CLS token
        if cls_token is not None:
            hidden_states = torch.cat([cls_token, hidden_states], dim=1)  # (B, L+1, d_mamba)

        # Padding 清零（把 pad 位置的向量置 0，防止干扰计算）
        if attention_mask is not None:
            if cls_token is not None:
                # CLS token 对应的 mask 补 1
                attention_mask = torch.cat([
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype,
                               device=attention_mask.device),
                    attention_mask,
                ], dim=1)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        # ── 前向分支（左→右）───────────────────────────────────────
        forward_hidden = hidden_states
        for layer in self.forward_layers:
            forward_hidden = layer(forward_hidden)

        if self.bidirectional:
            # ── 后向分支（右→左）─────────────────────────────────────
            # flip 翻转序列 → 处理 → flip 还原（使位置对齐）
            backward_hidden = torch.flip(hidden_states, dims=[1])
            for layer in self.backward_layers:
                backward_hidden = layer(backward_hidden)
            backward_hidden = torch.flip(backward_hidden, dims=[1])

            # ── 门控融合────────────────────────────────────────────
            # combined: (B, L, 2*d_mamba)
            combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(combined))     # (B, L, 2*d_mamba)
            gate_forward, gate_backward = gate.chunk(2, dim=-1)  # 各 (B, L, d_mamba)
            # 加权融合（gate_forward + gate_backward ≠ 1，各自独立的权重）
            fused_hidden = gate_forward * forward_hidden + gate_backward * backward_hidden
        else:
            # 单向模式（bidirectional=False，消融实验用）
            fused_hidden = forward_hidden

        # LayerNorm → 输出投影（d_mamba → d_model）
        fused_hidden = self.output_proj(self.norm(fused_hidden))
        return fused_hidden


# ═══════════════════════════════════════════════════════════
# BiMambaForPropertyPrediction（完整预测模型）
# ═══════════════════════════════════════════════════════════

class BiMambaForPropertyPrediction(nn.Module):
    """
    Bi-Mamba 分子性质预测完整模型（mamba_ssm 版）。

    与 bimamba.py 版本的差异：
    1. 多了 d_mamba 参数（embedding 维度和 Mamba 处理维度可不同）
    2. 多了 bidirectional 参数（支持单向消融实验）
    3. Encoder 内部有 input_proj/output_proj，外层维度接口统一

    Pooling 策略回顾（答辩重点）：
        mean  → 稳定，适合大多数任务
        max   → 保留最显著特征，HIV 分类最佳 (AUC 0.787)
        cls   → 可学习全局表示，Lipophilicity 回归最佳 (RMSE 1.19)
               但 HIV 分类失败 (AUC 0.498)，原因尚不明确
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_mamba: int = 256,      # Mamba 内部维度（可与 d_model 不同）
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_length: int = 512,
        num_labels: int = 1,
        task_type: str = "regression",
        pooling: str = "mean",
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bidirectional: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_labels = num_labels
        self.task_type = task_type
        self.pooling = pooling
        self.pad_token_id = pad_token_id

        self.encoder = BiMambaEncoder(
            vocab_size=vocab_size, d_model=d_model, d_mamba=d_mamba,
            n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand=expand,
            max_seq_length=max_seq_length, dropout=dropout,
            pad_token_id=pad_token_id, bidirectional=bidirectional,
            **factory_kwargs,
        )

        # CLS pooling 专用：可学习向量 (1, 1, d_model)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs))

        self.dropout = nn.Dropout(dropout)

        # 回归用 MLP head（两层），分类用线性层
        if task_type == "regression":
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2, **factory_kwargs),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_labels, **factory_kwargs),
            )
        else:
            self.classifier = nn.Linear(d_model, num_labels, **factory_kwargs)

        self.loss_fct = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,                   # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
        labels: Optional[torch.Tensor] = None,          # (B,) 或 (B, num_labels)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids:      (B, L)   token id
            attention_mask: (B, L)   1=有效, 0=padding
            labels:         (B,)     真实标签（训练时传，推理时不传）

        Returns:
            logits: (B,)       预测值
            loss:   scalar     损失值（仅训练时）
        """
        batch_size = input_ids.shape[0]

        # CLS token（expand 到当前 batch 大小）
        cls_token = None
        if self.pooling == "cls":
            cls_token = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)

        encoder_outputs = self.encoder(input_ids, attention_mask, cls_token=cls_token)
        # encoder_outputs: (B, L, d_model)

        # ── Pooling（三种策略）──────────────────────────────────────
        if self.pooling == "mean":
            # 有效 token 取平均（排除 padding 位置）
            if attention_mask is not None:
                sum_embeddings = torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1)
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(encoder_outputs, dim=1)

        elif self.pooling == "max":
            # 每维取最大值（padding 位置屏蔽为 -1e9 确保不被选中）
            if attention_mask is not None:
                masked_embeddings = encoder_outputs.clone()
                masked_embeddings[attention_mask == 0] = -1e9
                pooled_output = torch.max(masked_embeddings, dim=1)[0]
            else:
                pooled_output = torch.max(encoder_outputs, dim=1)[0]

        elif self.pooling == "cls":
            # 取序列第 0 位（CLS token 的输出）
            pooled_output = encoder_outputs[:, 0]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Dropout → Classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.num_labels == 1:
            logits = logits.squeeze(-1)  # (B, 1) → (B,)

        # 损失计算（训练时）
        loss = None
        if labels is not None:
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            if self.task_type == "classification" and labels.dtype == torch.long:
                labels = labels.float()   # BCEWithLogitsLoss 需要 float
            loss = self.loss_fct(logits, labels)

        return logits, loss


# ═══════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════

def create_bimamba_model(
    vocab_size: int,
    d_model: int = 256,
    d_mamba: int = 256,
    n_layers: int = 4,
    task_type: str = "regression",
    num_labels: int = 1,
    **kwargs,
) -> BiMambaForPropertyPrediction:
    """
    快速创建 mamba_ssm 版 Bi-Mamba 模型。

    使用示例：
        # HIV 分类（推荐配置）
        model = create_bimamba_model(
            vocab_size=45,
            d_model=256,
            n_layers=4,
            task_type="classification",
            pooling="max",
            d_state=16,
        )

        # Lipophilicity 回归（推荐配置）
        model = create_bimamba_model(
            vocab_size=45,
            task_type="regression",
            pooling="cls",
            d_state=16,
        )
    """
    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        d_mamba=d_mamba,
        n_layers=n_layers,
        task_type=task_type,
        num_labels=num_labels,
        **kwargs,
    )
