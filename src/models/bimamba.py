"""
Bi-Mamba 模型实现 - 分子性质预测

Mamba 是一种状态空间模型（State Space Model），计算复杂度为 O(N)，
比 Transformer 的 O(N^2) 更适合处理长分子序列。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class BiMambaBlock(nn.Module):
    """
    双向 Mamba 块 - 选择性状态空间模型核心组件。

    包含：输入投影、一维卷积、选择性扫描、门控机制、输出投影。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
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
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (
            math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"未知的 dt_init 类型: {dt_init}")

        self._init_dt_proj_bias(dt_min, dt_max, dt_init_floor, factory_kwargs)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        if device is not None:
            A = A.to(device)
        A = A.repeat(self.d_inner, 1).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def _init_dt_proj_bias(self, dt_min, dt_max, dt_init_floor, factory_kwargs):
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            hidden_states: (B, L, D)

        Returns:
            output: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seqlen]
        x = x.transpose(1, 2)

        x = self.activation(x)
        y = self.ssm(x)
        y = y * F.silu(z)

        output = self.out_proj(y)
        return output

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        状态空间模型（选择性扫描）。

        Args:
            x: (B, L, d_inner)

        Returns:
            y: (B, L, d_inner)
        """
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        y = self.selective_scan(x, dt, B, C)
        return y

    def _discretize(self, dt: torch.Tensor, B: torch.Tensor, A: torch.Tensor):
        dt_clamped = torch.clamp(dt, min=-10, max=10)
        dA = torch.exp(
            torch.clamp(dt_clamped.unsqueeze(-1) * A.unsqueeze(0), min=-50, max=50)
        )
        dB = dt_clamped.unsqueeze(-1) * B.unsqueeze(1)
        return dA, dB

    def _single_step(
        self,
        h: torch.Tensor,
        dt_t: torch.Tensor,
        B_t: torch.Tensor,
        C_t: torch.Tensor,
        x_t: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        dA, dB = self._discretize(dt_t, B_t, A)
        x_t_clamped = torch.clamp(x_t, min=-10, max=10)
        h_new = dA * h + dB * x_t_clamped.unsqueeze(-1)
        h_new = torch.clamp(h_new, min=-100, max=100)
        y_t = torch.sum(h_new * C_t.unsqueeze(1), dim=2)
        return y_t, h_new

    def selective_scan(
        self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """
        选择性扫描：依赖输入的状态更新机制。

        离散化: dA = exp(dt * A), dB = dt * B
        状态更新: h_new = dA * h + dB * x
        输出: y = C * h_new

        Args:
            x: (B, L, d_inner)
            dt: (B, L, d_inner)
            B: (B, L, d_state)
            C: (B, L, d_state)

        Returns:
            y: (B, L, d_inner)
        """
        batch, seqlen, dim = x.shape
        A = -torch.exp(self.A_log)
        D = self.D

        h = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seqlen):
            y_t, h = self._single_step(
                h, dt[:, t, :], B[:, t, :], C[:, t, :], x[:, t, :], A
            )
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x * D.unsqueeze(0)
        return y


class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器，同时从左到右和从右到左处理序列。
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
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cls_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            cls_token: (B, 1, D) optional CLS embedding to prepend

        Returns:
            hidden_states: (B, L, D) or (B, L+1, D) if cls_token provided
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

        if cls_token is not None:
            hidden_states = torch.cat([cls_token, hidden_states], dim=1)

        if attention_mask is not None:
            if cls_token is not None:
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
        d_state: int = 16,
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

        cls_token = None
        if self.pooling == "cls":
            cls_token = self.cls_token.expand(batch_size, -1, -1)

        encoder_outputs = self.encoder(input_ids, attention_mask, cls_token=cls_token)

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
            # encoder_outputs now has CLS at position 0
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
