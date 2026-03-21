"""
Molecular dataset handling module.
Loads CSV/JSON/TXT, tokenizes SMILES, validates with RDKit, creates DataLoaders.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rdkit import Chem


_SMILES_ELEMENTS: List[str] = [
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

_SPECIAL_TOKENS: List[str] = ["<pad>", ">", "<bos>", "<eos>"]


def _build_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {token: idx for idx, token in enumerate(_SPECIAL_TOKENS)}
    vocab.update(
        {char: idx + len(_SPECIAL_TOKENS) for idx, char in enumerate(_SMILES_ELEMENTS)}
    )
    return vocab


_VOCAB: Dict[str, int] = _build_vocab()
_VOCAB_SIZE: int = len(_VOCAB)


@dataclass
class MoleculeSample:
    smiles: str
    labels: List[float]


class MoleculeTokenizer:
    """SMILES分词器，提供encode/decode方法。"""

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None) -> None:
        if vocab_dict is None:
            self.vocab: Dict[str, int] = _VOCAB
        else:
            self.vocab = vocab_dict
        self.inverse_vocab: Dict[int, str] = {
            idx: token for token, idx in self.vocab.items()
        }
        self.vocab_size: int = len(self.vocab)

    def encode(self, smiles: str, max_length: int = 512) -> List[int]:
        return _tokenize_smiles(smiles, self.vocab, max_length)

    def decode(self, token_ids: List[int]) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token: str = self.inverse_vocab.get(token_id, "")
            if token not in ["<pad>", ">", "<bos>", "<eos>"]:
                tokens.append(token)
        return "".join(tokens)


@functools.lru_cache(maxsize=500000)
def _tokenize_smiles_cached(
    smiles: str, vocab_id: int, max_length: int
) -> Tuple[int, ...]:
    """Tokenize SMILES string with caching.

    Note: vocab_id is passed to make cache key unique per vocab.
    Returns Tuple for hashability (required by lru_cache).
    """
    vocab: Dict[str, int] = _VOCAB if vocab_id == id(_VOCAB) else {}
    tokens: List[int] = []
    i: int = 0
    while i < len(smiles):
        if i + 1 < len(smiles) and smiles[i : i + 2] in vocab:
            tokens.append(vocab[smiles[i : i + 2]])
            i += 2
        elif smiles[i] in vocab:
            tokens.append(vocab[smiles[i]])
            i += 1
        else:
            tokens.append(vocab["<pad>"])
            i += 1
    pad_token_id: int = vocab["<pad>"]
    if len(tokens) > max_length:
        return tuple(tokens[:max_length])
    return tuple(tokens + [pad_token_id] * (max_length - len(tokens)))


def _tokenize_smiles(smiles: str, vocab_id: int, max_length: int) -> List[int]:
    """Tokenize SMILES string (wrapper for caching)."""
    return list(_tokenize_smiles_cached(smiles, vocab_id, max_length))


class MoleculeDataset(Dataset):
    """分子数据集，支持CSV/JSON/TXT格式加载、RDKit验证、分词。"""

    def __init__(
        self,
        data_path: str,
        task_type: str = "regression",
        max_length: int = 512,
        validate_smiles: bool = True,
    ) -> None:
        self.task_type = task_type
        self.max_length = max_length
        self.validate_smiles = validate_smiles
        self.tokenizer = MoleculeTokenizer()
        self.vocab_id = id(_VOCAB)
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[MoleculeSample]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        file_extension = data_path[data_path.rfind(".") :] if "." in data_path else ""
        match file_extension:
            case ".csv":
                data = self._load_csv(data_path)
            case ".json":
                data = self._load_json(data_path)
            case ".txt":
                data = self._load_txt(data_path)
            case _:
                raise ValueError(f"不支持的文件格式: {data_path}")
        if self.validate_smiles:
            original_len = len(data)
            data = [item for item in data if _validate_smiles(item.smiles)]
            if len(data) < original_len:
                print(f"已过滤{original_len - len(data)}个无效SMILES字符串")
        return data

    def _load_csv(self, path: str) -> List[MoleculeSample]:
        df = pd.read_csv(path)
        smiles_col = df.columns[0]
        label_cols = df.columns[1:].tolist() if len(df.columns) > 1 else []
        smiles_list = df[smiles_col].astype(str).tolist()
        labels_list = (
            df[label_cols].to_numpy().tolist()
            if label_cols
            else [[0.0]] * len(smiles_list)
        )
        return [
            MoleculeSample(smiles=s, labels=l) for s, l in zip(smiles_list, labels_list)
        ]

    def _load_json(self, path: str) -> List[MoleculeSample]:
        with open(path, "r") as file:
            raw_data = json.load(file)
        return [
            MoleculeSample(smiles=item["smiles"], labels=item["labels"])
            for item in raw_data
        ]

    def _load_txt(self, path: str) -> List[MoleculeSample]:
        data: List[MoleculeSample] = []
        with open(path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                data.append(
                    MoleculeSample(
                        smiles=parts[0],
                        labels=[float(x) for x in parts[1:]]
                        if len(parts) >= 2
                        else [0.0],
                    )
                )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        item = self.data[idx]
        token_ids = _tokenize_smiles(item.smiles, self.vocab_id, self.max_length)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels_tensor = torch.tensor(
            item.labels,
            dtype=torch.float if self.task_type == "regression" else torch.long,
        )
        return input_ids, labels_tensor

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_pad_token_id(self) -> int:
        return self.tokenizer.vocab["<pad>"]


def _validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    task_type: str = "regression",
    max_length: int = 512,
    num_workers: int = 4,
) -> Tuple:
    def make_loader(path: str) -> torch.utils.data.DataLoader:
        dataset = MoleculeDataset(
            data_path=path, task_type=task_type, max_length=max_length
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(path == train_path),
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = make_loader(train_path)
    val_loader = (
        make_loader(val_path) if val_path and os.path.exists(val_path) else None
    )
    test_loader = (
        make_loader(test_path) if test_path and os.path.exists(test_path) else None
    )
    return train_loader, val_loader, test_loader
