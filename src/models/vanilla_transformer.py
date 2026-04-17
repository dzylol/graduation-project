"""
Vanilla Transformer 模型 - 分子性质预测

标准 Transformer Encoder 实现，用于与 Bi-Mamba 进行效率对比。
"""

from typing import Optional

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    单向 Transformer 编码器。

    架构流程：
        1. Token Embedding + Position Embedding → 初始 hidden_states
        2. N 层 Transformer Encoder → encoded_hidden
        3. 输出 (B, L, D)

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        n_layers: Transformer 层数
        n_heads: 注意力头数
        d_ffn: FFN 隐藏层维度
        max_seq_length: 最大序列长度
        dropout: Dropout 概率
        pad_token_id: padding token id
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ffn: int = 512,
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
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id, **factory_kwargs
        )
        self.position_embedding = nn.Embedding(max_seq_length, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            **factory_kwargs,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_norm = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            input_ids: (B, L) — token ids

        Returns:
            output: (B, L, D) — encoded hidden states
        """
        batch_size, seq_len = input_ids.shape

        token_emb = self.token_embedding(input_ids)

        position_ids = (
            torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_emb = self.position_embedding(position_ids)

        hidden_states = self.dropout(token_emb + position_emb)

        attention_mask = self._generate_causal_mask(seq_len, input_ids.device)

        encoded = self.transformer_encoder(
            hidden_states, src_key_padding_mask=(input_ids == self.pad_token_id)
        )

        return self.output_norm(encoded)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码 (causal mask)。"""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask


class VanillaTransformerForPropertyPrediction(nn.Module):
    """
    Vanilla Transformer 分子性质预测模型。
    支持回归任务（MSELoss）和分类任务（BCEWithLogitsLoss）。

    整体架构：
        Input → Encoder → Pooling → Dropout → Classifier → Output

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        n_layers: 编码器层数
        n_heads: 注意力头数
        d_ffn: FFN 隐藏层维度
        max_seq_length: 最大序列长度
        num_labels: 输出标签数
        task_type: "regression" 或 "classification"
        pooling: "mean" | "max" | "cls"
        dropout: Dropout 概率
        pad_token_id: padding token id
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ffn: int = 512,
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

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ffn=d_ffn,
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
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            input_ids: (B, L) — token ids
            attention_mask: (B, L) — attention mask (未使用，保留接口兼容)
            labels: (B,) 或 (B, num_labels) — 标签

        Returns:
            output: loss 或 logits
        """
        encoder_output = self.encoder(input_ids)

        if self.pooling == "mean":
            pooled = encoder_output.mean(dim=1)
        elif self.pooling == "max":
            pooled = encoder_output.max(dim=1).values
        elif self.pooling == "cls":
            if hasattr(self, "cls_token"):
                cls_token = self.cls_token.expand(encoder_output.size(0), -1, -1)
                pooled = torch.cat([cls_token, encoder_output], dim=1).mean(dim=1)
            else:
                pooled = encoder_output[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if labels is not None:
            loss = self.loss_fct(logits.squeeze(-1), labels)
            return loss

        return logits
