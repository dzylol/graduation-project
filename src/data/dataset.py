"""Molecular dataset classes for PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from rdkit import Chem

from .tokenizer import MoleculeTokenizer, default_vocab
from .column_mapping import validate_smiles_internal


@dataclass
class Data:
    smiles: str
    labels: List[float]


class MoleculeDataset(Dataset):
    """Molecular dataset with CSV/JSON/TXT loading, RDKit validation, tokenization."""

    def __init__(
        self,
        data_file_path: str,
        task_type: str = "regression",
        max_length: int = 512,
        validate_smiles: bool = True,
        smiles_col: Optional[str] = None,
        label_cols: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.task_type = task_type
        self.max_length = max_length
        self.validate_smiles = validate_smiles
        self.smiles_col = smiles_col
        self.label_cols = label_cols
        self.dataset_name = dataset_name
        self.tokenizer = MoleculeTokenizer()
        self.vocab_id = id(default_vocab)
        self.data = self.load_data_internal(data_file_path)

    def load_csv_internal(self, path: str) -> List[Data]:
        from .column_mapping import detect_column_mapping

        df = pd.read_csv(path)

        if self.smiles_col is not None:
            smiles_col = self.smiles_col
            label_cols = self.label_cols if self.label_cols else []
        else:
            mapping = detect_column_mapping(df, dataset_name=self.dataset_name)
            smiles_col = mapping.smiles_col
            label_cols = mapping.label_cols

        smiles_list = df[smiles_col].astype(str).tolist()

        if label_cols:
            labels_list = []
            for idx in range(len(smiles_list)):
                row_labels = []
                for col in label_cols:
                    val = df[col].iloc[idx]
                    if pd.isna(val):
                        continue
                    row_labels.append(float(val))
                if row_labels:
                    labels_list.append(row_labels)
                else:
                    labels_list.append([0.0])
        else:
            labels_list = [[0.0]] * len(smiles_list)

        return [Data(smiles=s, labels=l) for s, l in zip(smiles_list, labels_list)]

    def load_json_internal(self, path: str) -> List[Data]:
        import json

        with open(path, "r") as file:
            json_raw_data = json.load(file)
        return [Data(smiles=item["smiles"], labels=item["labels"]) for item in json_raw_data]

    def load_txt_internal(self, path: str) -> List[Data]:
        data: List[Data] = []
        with open(path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                data.append(
                    Data(
                        smiles=parts[0],
                        labels=[float(x) for x in parts[1:]] if len(parts) >= 2 else [0.0],
                    )
                )
        return data

    def load_data_internal(self, data_file_path: str) -> List[Data]:
        import os

        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found: {data_file_path}")
        file_extension = (
            data_file_path[data_file_path.rfind(".") :] if "." in data_file_path else ""
        )
        if file_extension == ".csv":
            data = self.load_csv_internal(data_file_path)
        elif file_extension == ".json":
            data = self.load_json_internal(data_file_path)
        elif file_extension == ".txt":
            data = self.load_txt_internal(data_file_path)
        else:
            raise ValueError(f"Unsupported file format: {data_file_path}")
        if self.validate_smiles:
            original_len = len(data)
            data = [item for item in data if validate_smiles_internal(item.smiles)]
            if len(data) < original_len:
                print(f"Filtered {original_len - len(data)} invalid SMILES strings")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        from .tokenizer import tokenize_smiles_cached_internal

        item = self.data[idx]
        token_ids = tokenize_smiles_cached_internal(
            item.smiles,
            self.vocab_id,
            self.max_length,
        )
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels_tensor = torch.tensor(
            item.labels,
            dtype=torch.float,
        )
        return input_ids, labels_tensor

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_pad_token_id(self) -> int:
        return self.tokenizer.vocab["<pad>"]


class DatabaseMoleculeDataset(Dataset):
    """Molecular dataset loaded from SQLite database.

    Uses MoleculeRepository to fetch molecules by dataset_name.
    """

    def __init__(
        self,
        dataset_name: str,
        db_path: str = "bi_mamba_chem.db",
        task_type: str = "regression",
        max_length: int = 512,
        property_name: Optional[str] = None,
    ) -> None:
        self.task_type = task_type
        self.max_length = max_length
        self.property_name = property_name
        self.tokenizer = MoleculeTokenizer()
        self.vocab_id = id(default_vocab)
        from src.db.molecule_repo import MoleculeRepository

        self.repo = MoleculeRepository(db_path)
        self.molecules = self.repo.get_dataset(dataset_name)
        if not self.molecules:
            raise ValueError(f"No molecules found for dataset: {dataset_name}")

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        from .tokenizer import tokenize_smiles_cached_internal

        mol = self.molecules[idx]
        token_ids = tokenize_smiles_cached_internal(
            mol.smiles,
            self.vocab_id,
            self.max_length,
        )
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        if self.property_name and self.property_name in mol.properties:
            label = mol.properties[self.property_name]
        elif mol.properties:
            label = list(mol.properties.values())[0]
        else:
            label = 0.0
        labels_tensor = torch.tensor(
            [label],
            dtype=torch.float if self.task_type == "regression" else torch.long,
        )
        return input_ids, labels_tensor

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_pad_token_id(self) -> int:
        return self.tokenizer.vocab["<pad>"]
