"""
Bi-Mamba Encoder: Bidirectional Mamba for molecular property prediction.
Implements bidirectional scanning to capture chemical environment from both directions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .mamba_block import MambaBlock, MambaLayer


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block.
    Processes sequences in both forward and backward directions.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_mamba: bool = True,
        fusion: str = 'gate',  # 'concat', 'add', 'gate'
    ):
        """
        Initialize Bidirectional Mamba Block.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout rate
            norm_eps: LayerNorm epsilon
            use_mamba: Whether to use mamba-ssm or manual implementation
            fusion: Fusion strategy for combining forward and backward
        """
        super().__init__()

        self.d_model = d_model
        self.fusion = fusion

        # Forward Mamba branch
        self.forward_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
        )

        # Backward Mamba branch
        self.backward_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
        )

        # Fusion layer
        if fusion == 'concat':
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        elif fusion == 'gate':
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            self.value_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with bidirectional processing.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Output tensor (batch, seq, d_model)
        """
        batch, seq_len, _ = x.shape

        # Forward direction (original order)
        forward_out = self.forward_block(x)

        # Backward direction (reverse sequence)
        x_rev = torch.flip(x, dims=[1])
        backward_out = self.backward_block(x_rev)
        backward_out = torch.flip(backward_out, dims=[1])

        # Fusion
        if self.fusion == 'concat':
            combined = torch.cat([forward_out, backward_out], dim=-1)
            output = self.fusion_proj(combined)
        elif self.fusion == 'add':
            output = forward_out + backward_out
        elif self.fusion == 'gate':
            combined = torch.cat([forward_out, backward_out], dim=-1)
            gate = torch.sigmoid(self.gate_proj(combined))
            value = self.value_proj(combined)
            output = gate * value + (1 - gate) * forward_out + (1 - gate) * backward_out
            output = output / 2
        else:
            output = forward_out + backward_out

        return output


class BiMambaEncoder(nn.Module):
    """
    Bidirectional Mamba Encoder.
    Multiple layers of Bidirectional Mamba blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        use_mamba: bool = True,
        fusion: str = 'gate',
        max_len: int = 512,
        padding_idx: int = 0,
    ):
        """
        Initialize Bi-Mamba Encoder.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of layers
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout rate
            norm_eps: LayerNorm epsilon
            use_mamba: Whether to use mamba-ssm or manual implementation
            fusion: Fusion strategy
            max_len: Maximum sequence length
            padding_idx: Padding token index
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx,
        )

        # Position embedding
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Dropout
        self.embedding_dropout = nn.Dropout(dropout)

        # Bi-Mamba layers
        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                norm_eps=norm_eps,
                use_mamba=use_mamba,
                fusion=fusion,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)

        Returns:
            Encoded representations (batch, seq, d_model)
        """
        batch, seq_len = input_ids.shape

        # Token embedding
        x = self.token_embedding(input_ids)  # (batch, seq, d_model)

        # Position embedding
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(position_ids)
        x = x + pos_emb

        # Dropout
        x = self.embedding_dropout(x)

        # Bi-Mamba layers
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        return x

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.d_model


class BiMambaModel(nn.Module):
    """
    Complete Bi-Mamba model for molecular property prediction.
    Includes encoder and pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        use_mamba: bool = True,
        fusion: str = 'gate',
        max_len: int = 512,
        padding_idx: int = 0,
        pool_type: str = 'mean',  # 'mean', 'max', 'cls'
    ):
        """
        Initialize Bi-Mamba model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of layers
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout rate
            norm_eps: LayerNorm epsilon
            use_mamba: Whether to use mamba-ssm or manual implementation
            fusion: Fusion strategy
            max_len: Maximum sequence length
            padding_idx: Padding token index
            pool_type: Pooling type ('mean', 'max', 'cls')
        """
        super().__init__()

        self.encoder = BiMambaEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
            fusion=fusion,
            max_len=max_len,
            padding_idx=padding_idx,
        )

        self.pool_type = pool_type
        self.d_model = d_model

        # [CLS] token for cls pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)

        Returns:
            Pooled representations (batch, d_model)
        """
        # Encode
        hidden = self.encoder(input_ids, attention_mask)

        # Pooling
        if self.pool_type == 'cls':
            # Use [CLS] token
            cls_tokens = self.cls_token.expand(input_ids.size(0), -1, -1)
            pooled = cls_tokens + hidden[:, 0:1, :]
            pooled = pooled.squeeze(1)
        elif self.pool_type == 'mean':
            # Mean pooling (ignoring padding)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)
        elif self.pool_type == 'max':
            # Max pooling
            pooled = hidden.max(dim=1)[0]
        else:
            pooled = hidden.mean(dim=1)

        return pooled


def test_bi_mamba():
    """Test Bi-Mamba implementation."""
    batch = 2
    seq_len = 32
    vocab_size = 100
    d_model = 64

    print("Testing Bi-Mamba Encoder...")

    # Test encoder
    encoder = BiMambaEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        use_mamba=False,  # Use manual implementation
        fusion='gate',
    )

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    out = encoder(input_ids)
    print(f"Encoder input: {input_ids.shape} -> output: {out.shape}")

    # Test full model
    model = BiMambaModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        use_mamba=False,
        pool_type='mean',
    )

    pooled = model(input_ids)
    print(f"Model input: {input_ids.shape} -> pooled: {pooled.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_bi_mamba()
