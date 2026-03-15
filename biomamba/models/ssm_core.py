"""
Manual implementation of State Space Model (SSM) core.
This is an alternative implementation that doesn't require mamba-ssm package.
Based on the S4 (Structured State Space Sequence) model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SSMCore(nn.Module):
    """
    Manual SSM core implementation.
    Implements the selective state space model inspired by Mamba.

    This is a simplified implementation for educational purposes and
    as a fallback when mamba-ssm is not available.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize SSM core.

        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolution for local context (applied on input before expansion)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=1,
        )

        # SSM parameters (A, B, C, D)
        # A: state transition matrix
        # B: input to state
        # C: state to output
        # D: direct skip connection (output = input + D * skip)

        # Learnable A matrix (diagonal + low-rank approximation)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Projection for B and C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.o_proj = nn.Linear(self.d_inner, d_model)

        # Selective mechanisms
        self.ssm_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner),
            nn.Sigmoid()
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.conv1d.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def _discretize(self, A, B, C, delta, dt):
        """
        Discretize continuous-time SSM to discrete-time.

        Args:
            A: State matrix (d_inner, d_state)
            B: Input matrix (d_inner, d_state)
            C: Output matrix (d_state, d_inner)
            delta: Input embedding (batch, seq, d_inner)
            dt: Time step (batch, seq, d_inner)

        Returns:
            Discretized B and C matrices
        """
        # Discretization: A_bar = exp(A * dt)
        A_bar = A.unsqueeze(1).unsqueeze(1) * dt.unsqueeze(-1)
        A_bar = torch.exp(A_bar)  # (batch, seq, d_inner, d_state)

        # B_bar = (A_bar - I) * A^{-1} * B * dt (simplified)
        # For diagonal A: B_bar = (exp(A*dt) - I) * B / A * dt
        B_bar = A_bar * B.unsqueeze(1).unsqueeze(1) * dt.unsqueeze(-1)

        return A_bar, B_bar

    def forward(self, x, state=None):
        """
        Forward pass of SSM core.

        Args:
            x: Input tensor (batch, seq, d_model)
            state: Previous state (batch, d_model, d_state)

        Returns:
            Output tensor (batch, seq, d_model)
        """
        batch, seq_len, _ = x.shape

        # Convolution for local context (applied on input)
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # (batch, d_inner, seq)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq, d_inner)

        # Input projection
        xz = self.in_proj(x)  # (batch, seq, d_inner * 2)
        x_inner, z = xz.chunk(2, dim=-1)  # Each: (batch, seq, d_inner)

        # SSM processing
        A = F.softplus(self.A_log)  # Ensure positive

        # Compute B and C from input
        bc = self.x_proj(x_conv)  # (batch, seq, d_state * 2)
        B, C = bc.chunk(2, dim=-1)  # Each: (batch, seq, d_state)

        # Time step (learnable)
        dt = torch.sigmoid(x_conv.mean(dim=-1, keepdim=True))  # (batch, 1, d_inner)
        dt = dt + 0.001  # Avoid zero

        # Selective gate
        gate = self.ssm_gate(x_conv)

        # Compute output using selective scan (simplified)
        # This is a simplified version - real Mamba uses CUDA-optimized selective scan
        output = self._selective_scan(
            x_conv * gate,
            A,
            B,
            C,
            self.D,
            dt
        )

        # Output projection with gating
        output = output * torch.sigmoid(z)
        output = self.o_proj(output)
        output = self.dropout(output)

        return output

    def _selective_scan(
        self,
        x_conv_gate: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified selective scan operation.
        For long sequences, this is more efficient than full attention.

        Args:
            x_conv_gate: Input after conv and gate (batch, seq, d_inner)
            A: State matrix (d_inner, d_state)
            B: Input to state (batch, seq, d_state)
            C: State to output (batch, seq, d_state)
            D: Skip connection (d_inner)
            dt: Time step (batch, 1, d_inner)

        Returns:
            Output (batch, seq, d_inner)
        """
        batch, seq_len, d_inner = x_conv_gate.shape
        d_state = B.shape[-1]

        # Discretize
        A_expanded = A.unsqueeze(0).unsqueeze(1)  # (1, 1, d_inner, d_state)
        dt_expanded = dt.unsqueeze(-1)  # (batch, 1, d_inner, 1)
        A_bar = torch.exp(A_expanded * dt_expanded)  # (batch, seq, d_inner, d_state)

        # B_bar: (batch, seq, d_inner, d_state)
        B_expanded = B.unsqueeze(2)  # (batch, seq, 1, d_state)
        dt_expanded2 = dt.unsqueeze(-1)  # (batch, 1, d_inner, 1)
        B_bar = A_bar * B_expanded * dt_expanded2

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x_conv_gate.device, dtype=x_conv_gate.dtype)

        outputs = []
        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv_gate[:, t].unsqueeze(-1)

            # Output: y = C * h + D * x
            # C[:, t]: (batch, d_state), h: (batch, d_inner, d_state)
            # Need: (batch, d_state) @ (batch, d_inner, d_state)^T -> (batch, d_inner)
            y = torch.matmul(C[:, t].unsqueeze(1), h.transpose(1, 2)).squeeze(1) + D * x_conv_gate[:, t]
            outputs.append(y)

        output = torch.stack(outputs, dim=1)  # (batch, seq, d_inner)

        return output


class SSMBlock(nn.Module):
    """
    Complete SSM Block with normalization and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.ssm = SSMCore(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Output tensor (batch, seq, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = x + residual
        return x


class BidirectionalSSM(nn.Module):
    """
    Bidirectional SSM for molecular sequences.
    Processes sequences in both forward and backward directions.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        fusion: str = 'concat',  # 'concat', 'add', 'gate'
    ):
        super().__init__()

        self.fusion = fusion

        # Forward SSM
        self.forward_ssm = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Backward SSM
        self.backward_ssm = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Fusion layer
        if fusion == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion == 'gate':
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            self.value_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        Forward pass with bidirectional processing.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Output tensor (batch, seq, d_model)
        """
        # Forward direction
        forward_out = self.forward_ssm(x)

        # Backward direction (reverse sequence)
        x_rev = torch.flip(x, dims=[1])
        backward_out = self.backward_ssm(x_rev)
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
            output = gate * value
        else:
            output = forward_out + backward_out

        return output


if __name__ == "__main__":
    # Test SSM implementation
    batch = 2
    seq_len = 32
    d_model = 64

    # Test SSM core
    ssm = SSMCore(d_model=d_model, d_state=16)
    x = torch.randn(batch, seq_len, d_model)
    out = ssm(x)
    print(f"SSM input: {x.shape}, output: {out.shape}")

    # Test bidirectional SSM
    bi_ssm = BidirectionalSSM(d_model=d_model, fusion='gate')
    out = bi_ssm(x)
    print(f"Bidirectional SSM input: {x.shape}, output: {out.shape}")
