"""
Multi-task learning module for Bi-Mamba.

This module extends the base Bi-Mamba model to support multi-task learning,
allowing simultaneous prediction of multiple molecular properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from src.models.bimamba import BiMambaEncoder


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head with task-specific layers and shared representation.

    Each task has its own prediction layer, with optional task-specific
    temperature scaling for uncertainty estimation.
    """

    def __init__(
        self,
        d_model: int,
        tasks: Dict[str, Dict[str, Any]],
        dropout: float = 0.1,
        shared_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.task_names = list(tasks.keys())

        hidden_dim = shared_hidden_dim or d_model

        self.task_specific_layers = nn.ModuleDict()
        self.task_losses = {}

        for task_name, task_config in tasks.items():
            task_type = task_config.get("type", "regression")
            num_labels = task_config.get("num_labels", 1)

            if task_type == "regression":
                self.task_losses[task_name] = nn.MSELoss()
                self.task_specific_layers[task_name] = nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_labels),
                )
            else:
                self.task_losses[task_name] = nn.BCEWithLogitsLoss()
                self.task_specific_layers[task_name] = nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_labels),
                )

    def forward(
        self, pooled_output: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for multi-task prediction.

        Args:
            pooled_output: (B, D) pooled representation from encoder

        Returns:
            logits: dict of task_name -> predictions
            losses: dict of task_name -> losses (None during inference)
        """
        logits = {}
        losses = {}

        for task_name in self.task_names:
            logits[task_name] = self.task_specific_layers[task_name](pooled_output)

        return logits, losses

    def compute_loss(
        self, logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss.

        Args:
            logits: dict of task_name -> predictions
            labels: dict of task_name -> targets

        Returns:
            total_loss: weighted sum of task losses
            task_losses: dict of individual task losses
        """
        task_losses = {}
        total_loss = 0.0

        for task_name in self.task_names:
            if task_name not in labels or task_name not in logits:
                continue

            pred = logits[task_name]
            target = labels[task_name]

            if target.dim() > 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            if pred.dim() > 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)

            loss = self.task_losses[task_name](pred, target)
            task_losses[task_name] = loss

            weight = self.tasks[task_name].get("weight", 1.0)
            total_loss += weight * loss

        return total_loss, task_losses


class BiMambaMultiTask(nn.Module):
    """
    Bi-Mamba model for multi-task molecular property prediction.

    Supports:
    - Multiple regression tasks (e.g., solubility, lipophilicity)
    - Multiple classification tasks (e.g., toxicity, BBBP)
    - Mixed task types
    - Task-specific loss weighting
    """

    def __init__(
        self,
        vocab_size: int,
        tasks: Dict[str, Dict[str, Any]],
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_length: int = 512,
        pooling: str = "mean",
        dropout: float = 0.1,
        pad_token_id: int = 0,
        task_strategy: str = "shared",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.tasks = tasks
        self.task_names = list(tasks.keys())
        self.num_tasks = len(tasks)
        self.task_strategy = task_strategy
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

        if task_strategy == "shared":
            self.multi_task_head = MultiTaskHead(
                d_model=d_model,
                tasks=tasks,
                dropout=dropout,
            )
        else:
            self.heads = nn.ModuleDict()
            for task_name, task_config in tasks.items():
                task_type = task_config.get("type", "regression")
                num_labels = task_config.get("num_labels", 1)
                self.heads[task_name] = nn.Linear(d_model, num_labels)

            self.task_losses = {}
            for task_name, task_config in tasks.items():
                task_type = task_config.get("type", "regression")
                if task_type == "regression":
                    self.task_losses[task_name] = nn.MSELoss()
                else:
                    self.task_losses[task_name] = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for multi-task learning.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            labels: optional dict of task_name -> targets

        Returns:
            logits: dict of task_name -> predictions
            loss: total weighted loss if labels provided, else None
        """
        batch_size, seq_len = input_ids.shape

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
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        pooled_output = self.dropout(pooled_output)

        if self.task_strategy == "shared":
            logits, _ = self.multi_task_head(pooled_output)
        else:
            logits = {}
            for task_name in self.task_names:
                logits[task_name] = self.heads[task_name](pooled_output)

        loss = None
        if labels is not None:
            if self.task_strategy == "shared":
                loss, _ = self.multi_task_head.compute_loss(logits, labels)
            else:
                total_loss = 0.0
                for task_name in self.task_names:
                    if task_name not in labels or task_name not in logits:
                        continue
                    pred = logits[task_name]
                    target = labels[task_name]

                    if target.dim() > 1 and target.shape[-1] == 1:
                        target = target.squeeze(-1)
                    if pred.dim() > 1 and pred.shape[-1] == 1:
                        pred = pred.squeeze(-1)

                    task_loss = self.task_losses[task_name](pred, target)
                    weight = self.tasks[task_name].get("weight", 1.0)
                    total_loss += weight * task_loss

                loss = total_loss

        return logits, loss

    def predict(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode prediction.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask

        Returns:
            predictions: dict of task_name -> predictions
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids, attention_mask, labels=None)
        return logits


def create_multitask_model(
    vocab_size: int,
    tasks: Dict[str, Dict[str, Any]],
    d_model: int = 256,
    n_layers: int = 4,
    pooling: str = "mean",
    dropout: float = 0.1,
    pad_token_id: int = 0,
    task_strategy: str = "shared",
) -> BiMambaMultiTask:
    """
    Factory function to create a multi-task Bi-Mamba model.

    Args:
        vocab_size: size of vocabulary
        tasks: dict of task_name -> {type: "regression"|"classification", weight: float}
        d_model: model dimension
        n_layers: number of Mamba layers
        pooling: pooling method
        dropout: dropout rate
        pad_token_id: padding token ID
        task_strategy: "shared" or "separate"

    Returns:
        BiMambaMultiTask model instance
    """
    return BiMambaMultiTask(
        vocab_size=vocab_size,
        tasks=tasks,
        d_model=d_model,
        n_layers=n_layers,
        pooling=pooling,
        dropout=dropout,
        pad_token_id=pad_token_id,
        task_strategy=task_strategy,
    )


def parse_task_string(task_str: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse task configuration from string.

    Format: "task1:regression:1.0,task2:classification:0.5"

    Args:
        task_str: task configuration string

    Returns:
        dict of task_name -> {type, weight, num_labels}
    """
    tasks = {}
    if not task_str:
        return tasks

    for task_spec in task_str.split(","):
        parts = task_spec.strip().split(":")
        task_name = parts[0]
        task_type = parts[1] if len(parts) > 1 else "regression"
        weight = float(parts[2]) if len(parts) > 2 else 1.0
        num_labels = int(parts[3]) if len(parts) > 3 else 1

        tasks[task_name] = {
            "type": task_type,
            "weight": weight,
            "num_labels": num_labels,
        }

    return tasks
