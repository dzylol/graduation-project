"""
Multi-task molecular dataset handling module.

This module provides functionality for loading and processing multi-task
molecular datasets where each molecule may have multiple properties.
"""

from typing import Any, Dict, List, Optional, Tuple

import json
import os
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MultiTaskMoleculeDataset(Dataset):
    """
    Multi-task molecular dataset.

    Each molecule can have multiple labels from different tasks
    (regression or classification).
    """

    def __init__(
        self,
        data_path: str,
        tasks: Dict[str, Dict[str, Any]],
        max_length: int = 512,
        tokenizer: Optional[object] = None,
        cache_smiles: bool = True,
        validate_smiles: bool = True,
    ):
        self.tasks = tasks
        self.task_names = list(tasks.keys())
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache_smiles = cache_smiles
        self.validate_smiles = validate_smiles

        self.vocab = self._init_vocab()
        self.vocab_size = len(self.vocab)

        self.data = self._load_data(data_path)
        self.smiles_cache: Dict[str, List[int]] = {} if cache_smiles else {}

    def _init_vocab(self) -> Dict[str, int]:
        chars: List[str] = [
            "(",
            ")",
            "[",
            "]",
            "=",
            "#",
            "%",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "+",
            "-",
            "/",
            ".",
            ":",
            ";",
            "<",
            ">",
            "@",
            "B",
            "Br",
            "C",
            "Cl",
            "F",
            "H",
            "I",
            "N",
            "O",
            "P",
            "S",
            "Si",
            "Te",
            "Se",
            "At",
        ]
        special_tokens: List[str] = ["<pad>", ">", "<bos>", "<eos>"]
        vocab: Dict[str, int] = {token: idx for idx, token in enumerate(special_tokens)}
        vocab.update(
            {char: idx + len(special_tokens) for idx, char in enumerate(chars)}
        )
        return vocab

    def _load_data(self, data_path: str) -> List[dict]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data: List[dict] = []

        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
            smiles_col = df.columns[0]

            for _, row in df.iterrows():
                smiles = str(row[smiles_col])
                labels = {}
                for task_name in self.task_names:
                    task_col = task_name
                    if task_col in df.columns:
                        labels[task_name] = float(row[task_col])
                    else:
                        labels[task_name] = 0.0
                data.append({"smiles": smiles, "labels": labels})

        elif data_path.endswith(".json"):
            with open(data_path, "r") as f:
                raw_data = json.load(f)
                for item in raw_data:
                    smiles = item.get("smiles", "")
                    labels = {}
                    for task_name in self.task_names:
                        labels[task_name] = item.get(task_name, 0.0)
                    data.append({"smiles": smiles, "labels": labels})

        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        if self.validate_smiles:
            from rdkit import Chem

            original_len = len(data)
            data = [item for item in data if self._validate_smiles(item["smiles"])]
            if len(data) < original_len:
                print(f"Filtered {original_len - len(data)} invalid SMILES")

        return data

    def _validate_smiles(self, smiles: str) -> bool:
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def _tokenize_smiles(self, smiles: str) -> List[int]:
        if self.cache_smiles and smiles in self.smiles_cache:
            return self.smiles_cache[smiles]

        tokens: List[int] = []
        i: int = 0

        while i < len(smiles):
            if i + 1 < len(smiles) and smiles[i : i + 2] in self.vocab:
                tokens.append(self.vocab[smiles[i : i + 2]])
                i += 2
            elif smiles[i] in self.vocab:
                tokens.append(self.vocab[smiles[i]])
                i += 1
            else:
                tokens.append(self.vocab["<pad>"])
                i += 1

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            pad_token: int = self.vocab["<pad>"]
            tokens = tokens + [pad_token] * (self.max_length - len(tokens))

        if self.cache_smiles:
            self.smiles_cache[smiles] = tokens

        return tokens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        item = self.data[idx]
        smiles = item["smiles"]
        raw_labels = item["labels"]

        token_ids = self._tokenize_smiles(smiles)
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        labels = {}
        for task_name in self.task_names:
            task_config = self.tasks[task_name]
            task_type = task_config.get("type", "regression")

            if task_name in raw_labels:
                label_value = raw_labels[task_name]
            else:
                label_value = 0.0

            if task_type == "regression":
                labels[task_name] = torch.tensor(label_value, dtype=torch.float)
            else:
                labels[task_name] = torch.tensor(label_value, dtype=torch.float)

        return input_ids, labels

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_pad_token_id(self) -> int:
        return self.vocab["<pad>"]


def create_multitask_loaders(
    train_path: str,
    tasks: Dict[str, Dict[str, Any]],
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create data loaders for multi-task learning.

    Args:
        train_path: training data path
        tasks: dict of task_name -> {type, weight, num_labels}
        val_path: validation data path (optional)
        test_path: test data path (optional)
        batch_size: batch size
        max_length: maximum sequence length
        num_workers: number of data loading workers

    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    train_dataset = MultiTaskMoleculeDataset(
        data_path=train_path,
        tasks=tasks,
        max_length=max_length,
    )

    val_dataset = None
    if val_path and os.path.exists(val_path):
        val_dataset = MultiTaskMoleculeDataset(
            data_path=val_path,
            tasks=tasks,
            max_length=max_length,
        )

    test_dataset = None
    if test_path and os.path.exists(test_path):
        test_dataset = MultiTaskMoleculeDataset(
            data_path=test_path,
            tasks=tasks,
            max_length=max_length,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
