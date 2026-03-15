"""
Mamba Block implementation using mamba-ssm package.
This provides the official Mamba implementation with CUDA-optimized selective scan.
"""

import torch
import torch.nn as nn
from typing import Optional
import math

# Try to import mamba_ssm, fall back to manual implementation if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba-ssm not found. Using manual SSM implementation.")
    MAMBA_AVAILABLE = False


class MambaBlock(nn.Module):
    """
    Mamba Block wrapper.
    Uses official mamba-ssm if available, otherwise uses manual implementation.
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
    ):
        """
        Initialize Mamba Block.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout rate
            norm_eps: LayerNorm epsilon
            use_mamba: Whether to use mamba-ssm (True) or manual implementation (False)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.use_mamba = use_mamba and MAMBA_AVAILABLE

        # Pre-norm architecture
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)

        if self.use_mamba:
            # Use official mamba-ssm
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fall back to manual implementation
            from .ssm_core import SSMCore
            self.mamba = SSMCore(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual connection.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Output tensor (batch, seq, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        return x


class MambaLayer(nn.Module):
    """
    Single Mamba layer that can be stacked.
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
    ):
        super().__init__()

        self.block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def test_mamba_block():
    """Test Mamba block functionality."""
    batch = 2
    seq_len = 32
    d_model = 64

    print("Testing Mamba Block...")
    print(f"MAMBA_AVAILABLE: {MAMBA_AVAILABLE}")

    # Test with mamba (if available) or manual
    for use_mamba in [False, True]:
        if use_mamba and not MAMBA_AVAILABLE:
            print("Skipping mamba-ssm test (not available)")
            continue

        print(f"\nTesting use_mamba={use_mamba}")
        block = MambaBlock(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_mamba=use_mamba,
        )

        x = torch.randn(batch, seq_len, d_model)
        out = block(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_mamba_block()
