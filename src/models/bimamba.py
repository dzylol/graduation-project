"""
Bi-Mamba 模型实现 - 分子性质预测

Mamba: O(N) 状态空间模型，比 Transformer 的 O(N²) 更适合处理长分子序列。
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiMambaBlock(nn.Module):
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
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.layer_idx = layer_idx

        # Input projection: x -> [x, z]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = nn.SiLU()

        # SSM parameters: dt, B, C
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init must be 'constant' or 'random', got {dt_init}")

        self._init_dt_proj_bias(dt_min, dt_max, dt_init_floor, factory_kwargs)

        # SSM state parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A.repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def _init_dt_proj_bias(
        self, dt_min: float, dt_max: float, dt_init_floor: float, factory_kwargs: dict
    ) -> None:
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seqlen]
        x = x.transpose(1, 2)

        x = self.activation(x)
        y = self.ssm(x)
        y = y * F.silu(z)

        return self.out_proj(y) + hidden_states

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        return self.selective_scan(x, dt, B, C)

    def _discretize(self, dt: torch.Tensor, B: torch.Tensor, A: torch.Tensor):
        dt_clamped = torch.clamp(dt, min=-10, max=10)
        dA = torch.exp(torch.clamp(dt_clamped.unsqueeze(-1) * A.unsqueeze(0), min=-50, max=50))
        dB = (dA - 1) / A.unsqueeze(0) * B.unsqueeze(1)
        return dA, dB

    def _single_step(
        self,
        h: torch.Tensor,
        dt_t: torch.Tensor,
        B_t: torch.Tensor,
        C_t: torch.Tensor,
        x_t: torch.Tensor,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dA, dB = self._discretize(dt_t, B_t, A)
        x_t_clamped = torch.clamp(x_t, min=-10, max=10)
        h_new = torch.clamp(dA * h + dB * x_t_clamped.unsqueeze(-1), min=-100, max=100)
        y_t = torch.sum(h_new * C_t.unsqueeze(1), dim=2)
        return y_t, h_new

    def selective_scan(
        self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        batch, seqlen, dim = x.shape
        A = -torch.exp(self.A_log)
        D = self.D

        h = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seqlen):
            y_t, h = self._single_step(h, dt[:, t, :], B[:, t, :], C[:, t, :], x[:, t, :], A)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y + x * D.unsqueeze(0)


class BiMambaEncoder(nn.Module):
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
        self.position_embedding = nn.Embedding(max_seq_length, d_model, **factory_kwargs)

        self.forward_layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **factory_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.backward_layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **factory_kwargs,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.fusion_gate = nn.Linear(d_model * 2, d_model * 2, **factory_kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cls_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        hidden_states = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(position_ids)
        )

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
        gate_fwd, gate_bwd = gate.chunk(2, dim=-1)
        fused_hidden = gate_fwd * forward_hidden + gate_bwd * backward_hidden

        return self.norm(fused_hidden)


class BiMambaForPropertyPrediction(nn.Module):
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = input_ids.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1) if self.pooling == "cls" else None
        encoder_outputs = self.encoder(input_ids, attention_mask, cls_token=cls_token)

        if self.pooling == "mean":
            if attention_mask is not None:
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1e-9)
                pooled_output = (
                    torch.sum(encoder_outputs * attention_mask.unsqueeze(-1), dim=1) / sum_mask
                )
            else:
                pooled_output = torch.mean(encoder_outputs, dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                masked = encoder_outputs.clone()
                masked[attention_mask == 0] = -1e9
                pooled_output = torch.max(masked, dim=1)[0]
            else:
                pooled_output = torch.max(encoder_outputs, dim=1)[0]
        elif self.pooling == "cls":
            pooled_output = encoder_outputs[:, 0]
        else:
            raise ValueError(f"pooling must be 'mean', 'max', or 'cls', got {self.pooling}")

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
    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        task_type=task_type,
        num_labels=num_labels,
        **kwargs,
    )
