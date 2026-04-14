"""DataLoader creation and label normalization utilities."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from .dataset import DatabaseMoleculeDataset, MoleculeDataset


class LabelNormalizer:
    """Z-score normalization for regression targets.

    Transforms labels to zero-mean, unit-variance.
    Fit on training set only, apply to val/test sets.
    """

    def __init__(self) -> None:
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self._fitted: bool = False

    def fit(self, labels: np.ndarray) -> "LabelNormalizer":
        self.mean = float(np.mean(labels))
        self.std = float(np.std(labels))
        if self.std < 1e-8:
            self.std = 1.0
        self._fitted = True
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (labels - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return normalized * self.std + self.mean

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class NormalizedDataset(Dataset):
    """Wraps MoleculeDataset to apply z-score normalization to labels.

    Use this wrapper when training with normalized regression labels.
    The normalizer should be fitted on the training set only.
    """

    def __init__(
        self,
        base_dataset: MoleculeDataset,
        normalizer: LabelNormalizer,
    ) -> None:
        self.base_dataset = base_dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.base_dataset[idx]
        normalized_labels = self.normalizer.transform(labels.numpy())
        return input_ids, torch.tensor(normalized_labels, dtype=torch.float)


def create_data_loaders(
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    task_type: str = "regression",
    max_length: int = 512,
    num_workers: int = 4,
    normalize: bool = True,
    train_dataset_name: Optional[str] = None,
    val_dataset_name: Optional[str] = None,
    test_dataset_name: Optional[str] = None,
    db_path: str = "bi_mamba_chem.db",
    property_name: Optional[str] = None,
    smiles_col: Optional[str] = None,
    label_cols: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
) -> Tuple:
    """Create train/val/test DataLoaders from file paths or database datasets.

    Args:
        train_path: Training set file path (CSV/JSON/TXT) - mutually exclusive with train_dataset_name
        val_path: Validation set file path (optional)
        test_path: Test set file path (optional)
        train_dataset_name: Database dataset name for training - mutually exclusive with train_path
        val_dataset_name: Database dataset name for validation (optional)
        test_dataset_name: Database dataset name for testing (optional)
        batch_size: Samples per batch, default 32
        task_type: "regression" or "classification"
        max_length: Max SMILES token sequence length
        num_workers: Data loading worker processes
        normalize: Apply z-score normalization for regression labels (default True)
        db_path: Path to SQLite database (when using dataset_name)
        property_name: Property to use as label when loading from database

    Returns:
        (train_loader, val_loader, test_loader, normalizer) tuple
        normalizer is LabelNormalizer or None (classification or normalize=False)
    """
    normalizer: Optional[LabelNormalizer] = None

    if train_path and train_dataset_name:
        raise ValueError("Cannot specify both train_path and train_dataset_name")
    if train_path is None and train_dataset_name is None:
        raise ValueError("Must specify either train_path or train_dataset_name")

    def _load_split_meta(output_dir: str) -> Optional[dict]:
        meta_path = os.path.join(output_dir, "split_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return None

    # Try to load column mapping from split metadata if files come from a scaffold split
    if smiles_col is None and label_cols is None and train_path:
        train_dir = os.path.dirname(train_path)
        meta = _load_split_meta(train_dir)
        if meta:
            smiles_col = meta.get("smiles_col")
            label_col_from_meta = meta.get("label_col")
            if label_col_from_meta:
                label_cols = [label_col_from_meta]

    def make_file_loader(path: str, is_train: bool = False) -> torch.utils.data.DataLoader:
        dataset = MoleculeDataset(
            data_file_path=path,
            task_type=task_type,
            max_length=max_length,
            smiles_col=smiles_col,
            label_cols=label_cols,
            dataset_name=dataset_name,
        )
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=(path == train_path),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 and is_train else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)

    def make_db_loader(dataset_name: str, is_train: bool = False) -> torch.utils.data.DataLoader:
        dataset = DatabaseMoleculeDataset(
            dataset_name=dataset_name,
            db_path=db_path,
            task_type=task_type,
            max_length=max_length,
            property_name=property_name,
        )
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 and is_train else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)

    if train_dataset_name:
        train_loader = make_db_loader(train_dataset_name, is_train=True)
    else:
        assert train_path is not None, "train_path required when train_dataset_name not provided"
        train_loader = make_file_loader(train_path, is_train=True)

    if normalize and task_type == "regression":
        normalizer = LabelNormalizer()
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy().flatten())
        normalizer.fit(np.array(train_labels))

        normalized_train_dataset = NormalizedDataset(
            base_dataset=train_loader.dataset,
            normalizer=normalizer,
        )
        train_loader = torch.utils.data.DataLoader(
            normalized_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    if val_dataset_name:
        val_loader = make_db_loader(val_dataset_name)
    elif val_path:
        val_loader = make_file_loader(val_path) if os.path.exists(val_path) else None
    else:
        val_loader = None

    if test_dataset_name:
        test_loader = make_db_loader(test_dataset_name)
    elif test_path:
        test_loader = make_file_loader(test_path) if os.path.exists(test_path) else None
    else:
        test_loader = None

    # Also wrap val/test loaders in NormalizedDataset for consistent label space
    if normalize and task_type == "regression" and normalizer is not None:
        if val_loader is not None:
            normalized_val_dataset = NormalizedDataset(
                base_dataset=val_loader.dataset,
                normalizer=normalizer,
            )
            val_loader = torch.utils.data.DataLoader(
                normalized_val_dataset,
                batch_size=batch_size,
                shuffle=False,  # Val/test should not be shuffled
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=False,  # Disable to avoid temp file issues
                prefetch_factor=4 if num_workers > 0 else None,
            )
        if test_loader is not None:
            normalized_test_dataset = NormalizedDataset(
                base_dataset=test_loader.dataset,
                normalizer=normalizer,
            )
            test_loader = torch.utils.data.DataLoader(
                normalized_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=4 if num_workers > 0 else None,
            )

    return train_loader, val_loader, test_loader, normalizer
