"""
Molecular dataset handling module.
Loads CSV/JSON/TXT, tokenizes SMILES, validates with RDKit, creates DataLoaders.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rdkit import Chem


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

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        input_ids, labels = self.base_dataset[idx]
        normalized_labels = self.normalizer.transform(labels.numpy())
        return input_ids, torch.tensor(normalized_labels, dtype=torch.float)


smiles_token_tuple: tuple[str, ...] = (
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
)

special_token_tuple: tuple[str, ...] = (
    "<pad>",  # 填充标记
    "<unk>",  # 未知字符
    "<bos>",  # 句子开始
    "<eos>",  # 句子结束
)


def build_default_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {
        token: idx for idx, token in enumerate(special_token_tuple)
    }
    vocab.update(
        {
            char: idx + len(special_token_tuple)
            for idx, char in enumerate(smiles_token_tuple)
        }
    )
    return vocab  # Dict[str, int]


default_vocab: Dict[str, int] = build_default_vocab()
default_vocab_size: int = len(default_vocab)


@dataclass  # 开启简化class模式(只适合纯数据类（只有赋值）)
class Data:  # 开头大写的是类名
    smiles: str  # ---> self.smiles = smiles,下同
    labels: List[float]


class MoleculeTokenizer:
    """SMILES分词器，提供encode/decode方法。"""

    def __init__(  # 有 if-else 逻辑，无法简化。
        self,
        given_vocab_dict: Optional[
            Dict[str, int]
        ] = None,  # OptionalType 表示参数可以是 Type 或者 None，不传参数也不会报错
    ) -> None:
        if given_vocab_dict is None:
            self.vocab: Dict[str, int] = default_vocab
        else:
            self.vocab = given_vocab_dict
        self.inverse_vocab: Dict[int, str] = {
            idx: token for token, idx in self.vocab.items()
        }
        self.vocab_size: int = len(self.vocab)

    def encode(self, smiles: str, max_length: int = 512) -> Tuple[int, ...]:
        # id() 是 Python 内置函数，返回对象的内存地址（整数）。
        return tokenize_smiles_cached_internal(
            smiles, id(self.vocab), max_length
        )  # 校验缓存是否命中

    def decode(self, token_ids: List[int]) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token: str = self.inverse_vocab.get(token_id, "")
            if token not in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                tokens.append(token)
        return "".join(tokens)


@functools.lru_cache(maxsize=500000)
def tokenize_smiles_cached_internal(
    smiles: str, vocab_id: int, max_length: int
) -> Tuple[int, ...]:
    """Tokenize SMILES string with caching.

    Note: vocab_id is passed to make cache key unique per vocab.
    Returns Tuple for hashability (required by lru_cache).
    """
    given_vocab_dict: Dict[str, int] = (
        default_vocab if vocab_id == id(default_vocab) else {}
    )
    tokens: List[int] = []
    i: int = 0
    while i < len(smiles):
        if i + 1 < len(smiles) and smiles[i : i + 2] in given_vocab_dict:
            tokens.append(given_vocab_dict[smiles[i : i + 2]])
            i += 2
        elif smiles[i] in given_vocab_dict:
            tokens.append(given_vocab_dict[smiles[i]])
            i += 1
        else:
            tokens.append(given_vocab_dict["<pad>"])
            i += 1
    pad_token_id: int = given_vocab_dict["<pad>"]
    if len(tokens) > max_length:
        return tuple(tokens[:max_length])
    return tuple(tokens + [pad_token_id] * (max_length - len(tokens)))


class MoleculeDataset(Dataset):
    """分子数据集，支持CSV/JSON/TXT格式加载、RDKit验证、分词。"""

    def __init__(
        self,
        data_file_path: str,
        task_type: str = "regression",
        max_length: int = 512,
        validate_smiles: bool = True,
    ) -> None:
        self.task_type = task_type
        self.max_length = max_length
        self.validate_smiles = validate_smiles
        self.tokenizer = MoleculeTokenizer()
        self.vocab_id = id(default_vocab)
        self.data = self.load_data_internal(data_file_path)

    def load_csv_internal(self, path: str) -> List[Data]:
        df = pd.read_csv(path)
        smiles_col = df.columns[0]
        label_cols = df.columns[1:].tolist() if len(df.columns) > 1 else []
        smiles_list = df[smiles_col].astype(str).tolist()
        labels_list = (
            df[label_cols].to_numpy().tolist()
            if label_cols
            else [[0.0]] * len(smiles_list)
        )
        return [Data(smiles=s, labels=l) for s, l in zip(smiles_list, labels_list)]

    def load_json_internal(self, path: str) -> List[Data]:
        with open(path, "r") as file:
            json_raw_data = json.load(file)
        return [
            Data(smiles=item["smiles"], labels=item["labels"]) for item in json_raw_data
        ]

    def load_txt_internal(self, path: str) -> List[Data]:
        data: List[Data] = []
        with open(path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                data.append(
                    Data(
                        smiles=parts[0],
                        labels=[float(x) for x in parts[1:]]
                        if len(parts) >= 2
                        else [0.0],
                    )
                )
        return data

    def load_data_internal(self, data_file_path: str) -> List[Data]:
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"数据文件不存在: {data_file_path}")
        file_extension = (
            data_file_path[data_file_path.rfind(".") :] if "." in data_file_path else ""
        )
        match file_extension:
            case ".csv":
                data = self.load_csv_internal(data_file_path)
            case ".json":
                data = self.load_json_internal(data_file_path)
            case ".txt":
                data = self.load_txt_internal(data_file_path)
            case _:
                raise ValueError(f"不支持的文件格式: {data_file_path}")
        if self.validate_smiles:
            original_len = len(data)
            data = [item for item in data if validate_smiles_internal(item.smiles)]
            if len(data) < original_len:
                print(f"已过滤{original_len - len(data)}个无效SMILES字符串")
        return data

    def __len__(self) -> int:  # 让对象支持 len() 函数，返回数据集的样本数量。
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tensor, Tensor]:  # 让对象支持索引操作，返回指定索引的样本。
        item = self.data[idx]
        # 将 SMILES 字符串转换为 token 整数序列（字符->整数映射），不足 max_length 的用 <pad> 填充
        token_ids = tokenize_smiles_cached_internal(
            item.smiles,  # SMILES 字符串，如 "CCO"
            self.vocab_id,  # 词汇表 ID，用于缓存
            self.max_length,  # 最大长度，不足则 padding
        )
        # 将 token 整数序列转换为 PyTorch Tensor（long 类型用于 embedding 查找）
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        # 标签类型根据任务类型决定：回归用 float（连续值）:溶解度、毒性数值，分类用 long（离散类别）比如是否有毒
        labels_tensor = torch.tensor(
            item.labels,
            dtype=torch.float if self.task_type == "regression" else torch.long,
        )
        return input_ids, labels_tensor

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_pad_token_id(self) -> int:
        return self.tokenizer.vocab["<pad>"]


def validate_smiles_internal(smiles: str) -> bool:
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
    normalize: bool = True,
) -> Tuple:
    """创建训练/验证/测试数据加载器。

    Args:
        train_path: 训练集文件路径（必需）
        val_path: 验证集文件路径（可选）
        test_path: 测试集文件路径（可选）
        batch_size: 每批样本数，默认 32
        task_type: 任务类型，"regression" 或 "classification"
        max_length: SMILES token 序列最大长度
        num_workers: 数据加载的进程数
        normalize: 是否对回归标签进行 z-score 归一化（默认 True）

    Returns:
        (train_loader, val_loader, test_loader, normalizer) 元组
        normalizer 为 LabelNormalizer 或 None（分类任务或不 normalize 时）
    """
    normalizer: Optional[LabelNormalizer] = None

    def make_loader(path: str, is_train: bool = False) -> torch.utils.data.DataLoader:
        dataset = MoleculeDataset(
            data_file_path=path, task_type=task_type, max_length=max_length
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

    train_loader = make_loader(train_path, is_train=True)

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

    val_loader = (
        make_loader(val_path) if val_path and os.path.exists(val_path) else None
    )
    test_loader = (
        make_loader(test_path) if test_path and os.path.exists(test_path) else None
    )
    return train_loader, val_loader, test_loader, normalizer


def list_available_databases(
    database_dir: str = "src/data/database",
) -> list[str]:
    """列出数据库目录中的所有可用数据库文件。

    Args:
        database_dir: 数据库文件夹路径

    Returns:
        数据库文件路径列表
    """
    if not os.path.exists(database_dir):
        return []
    return sorted(
        [
            os.path.join(database_dir, f)
            for f in os.listdir(database_dir)
            if f.endswith(".db")
        ]
    )


def select_database(
    database_dir: str = "src/data/database",
) -> str:
    """交互式选择数据库文件。

    Args:
        database_dir: 数据库文件夹路径

    Returns:
        选择的数据库文件路径

    Raises:
        FileNotFoundError: 没有找到任何数据库文件
    """
    db_files = list_available_databases(database_dir)

    if not db_files:
        raise FileNotFoundError(f"数据库目录为空或不存在: {database_dir}")

    print("可用数据库：")
    for i, db_path in enumerate(db_files, 1):
        print(f"  [{i}] {os.path.basename(db_path)}")

    while True:
        try:
            choice = int(input("\n请选择数据库编号: "))
            if 1 <= choice <= len(db_files):
                return db_files[choice - 1]
            print(f"无效选择，请输入 1-{len(db_files)}")
        except ValueError:
            print("请输入有效数字")
