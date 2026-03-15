"""
Prediction Head for molecular property prediction.
Supports both regression and classification tasks.
"""

import torch
import torch.nn as nn
from typing import Optional


class PredictionHead(nn.Module):
    """
    Prediction head for molecular property prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        task_type: str = 'regression',  # 'regression' or 'classification'
        dropout: float = 0.1,
        n_layers: int = 2,
    ):
        """
        Initialize prediction head.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension (if None, uses input_dim)
            output_dim: Output dimension
            task_type: 'regression' or 'classification'
            dropout: Dropout rate
            n_layers: Number of hidden layers
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.task_type = task_type
        self.output_dim = output_dim

        # Build MLP
        layers = []
        in_dim = input_dim

        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Output tensor (batch, output_dim)
        """
        return self.mlp(x)


class BiMambaForPrediction(nn.Module):
    """
    Complete Bi-Mamba model with prediction head.
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
        pool_type: str = 'mean',
        # Prediction head parameters
        pred_hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        task_type: str = 'regression',
        pred_dropout: float = 0.1,
    ):
        """
        Initialize complete model.

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
            pool_type: Pooling type
            pred_hidden_dim: Prediction head hidden dimension
            output_dim: Output dimension
            task_type: 'regression' or 'classification'
            pred_dropout: Prediction head dropout
        """
        super().__init__()

        self.task_type = task_type

        # Encoder
        from .bi_mamba import BiMambaModel
        self.encoder = BiMambaModel(
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
            pool_type=pool_type,
        )

        # Prediction head
        self.predictor = PredictionHead(
            input_dim=d_model,
            hidden_dim=pred_hidden_dim,
            output_dim=output_dim,
            task_type=task_type,
            dropout=pred_dropout,
            n_layers=2,
        )

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
            Predictions (batch, output_dim)
        """
        # Encode
        pooled = self.encoder(input_ids, attention_mask)

        # Predict
        predictions = self.predictor(pooled)

        return predictions


class BiMambaForSequenceClassification(nn.Module):
    """
    Bi-Mamba for sequence classification (e.g., BBBP, ClinTox).
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 2,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
        fusion: str = 'gate',
        pool_type: str = 'mean',
    ):
        super().__init__()

        self.num_labels = num_labels

        # Bi-Mamba encoder + classifier
        self.model = BiMambaForPrediction(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_mamba=use_mamba,
            fusion=fusion,
            pool_type=pool_type,
            output_dim=num_labels,
            task_type='classification',
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)

        Returns:
            Dictionary with loss and logits
        """
        logits = self.model(input_ids, attention_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
        }


class BiMambaForRegression(nn.Module):
    """
    Bi-Mamba for regression (e.g., ESOL solubility).
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
        use_mamba: bool = True,
        fusion: str = 'gate',
        pool_type: str = 'mean',
    ):
        super().__init__()

        # Bi-Mamba encoder + regressor
        self.model = BiMambaForPrediction(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_mamba=use_mamba,
            fusion=fusion,
            pool_type=pool_type,
            output_dim=1,
            task_type='regression',
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth values (optional)

        Returns:
            Dictionary with loss and predictions
        """
        predictions = self.model(input_ids, attention_mask)
        predictions = predictions.squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions,
        }


def test_prediction_head():
    """Test prediction head implementation."""
    batch = 2
    input_dim = 128

    print("Testing Prediction Head...")

    # Test regression head
    pred_head = PredictionHead(
        input_dim=input_dim,
        output_dim=1,
        task_type='regression',
    )
    x = torch.randn(batch, input_dim)
    out = pred_head(x)
    print(f"Regression head: input {x.shape} -> output {out.shape}")

    # Test classification head
    pred_head = PredictionHead(
        input_dim=input_dim,
        output_dim=2,
        task_type='classification',
    )
    out = pred_head(x)
    print(f"Classification head: input {x.shape} -> output {out.shape}")

    # Test full model
    model = BiMambaForPrediction(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        use_mamba=False,
        task_type='regression',
    )

    input_ids = torch.randint(0, 100, (batch, 32))
    out = model(input_ids)
    print(f"Full model: input {input_ids.shape} -> output {out.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_prediction_head()
