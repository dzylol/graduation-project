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
from sklearn.model_selection import train_test_split

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

# 常见的 SMILES 列名（不区分大小写，去空白后匹配）
SMILES_COLUMNS: set[str] = {
    "smiles",
    "smile",
    "canonical_smiles",
    "canonical",
    "smi",
    "structure",
    "molecule",
    "mol",
    "chem_smile",
}

# 非 SMILES 数据集（明确跳过）
IGNORED_DATASETS: set[str] = {
    "sider",  # 药物副作用数据库，无 SMILES 列
}

special_token_tuple: tuple[str, ...] = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
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
    return vocab


default_vocab: Dict[str, int] = build_default_vocab()
default_vocab_size: int = len(default_vocab)

# 预定义数据集的列配置（用于自动检测失败或列名非常规的数据集）
DATASET_CONFIG: Dict[str, Dict[str, List[str]]] = {
    "ESOL": {
        "smiles_col": "SMILES",
        "label_cols": ["measured log(solubility:mol/L)"],
    },
    "BBBP": {
        "smiles_col": "smiles",
        "label_cols": ["p_np"],
    },
    "ZINC250K": {
        "smiles_col": "smiles",
        "label_cols": ["logP", "qed", "SAS"],
    },
    "FreeSolv": {
        "smiles_col": "smiles",
        "label_cols": ["freesolv"],
    },
    "Lipophilicity": {
        "smiles_col": "smiles",
        "label_cols": ["exp"],
    },
    "ClinTox": {
        "smiles_col": "smiles",
        "label_cols": ["FDA_APPROVED", "CT_TOX"],
    },
    "HIV": {
        "smiles_col": "smiles",
        "label_cols": ["HIV_active"],
    },
    "MUV": {
        "smiles_col": "smiles",
        "label_cols": [],  # 动态检测
    },
    "mpro": {
        "smiles_col": "smiles",
        "label_cols": ["IC50"],
    },
    "EGFR": {
        "smiles_col": "smiles",
        "label_cols": ["value"],
    },
    "bace": {
        "smiles_col": "smiles",
        "label_cols": ["pIC50"],
    },
}


@dataclass
class ColumnMapping:
    """CSV column mapping result for auto-detection."""

    smiles_col: str
    label_cols: List[str]
    detection_method: str
    confidence: float = 1.0


def detect_column_mapping(
    df: pd.DataFrame, dataset_name: Optional[str] = None
) -> ColumnMapping:
    """Auto-detect CSV column mapping (smiles_col + label_cols).

    Detection order:
    1. DATASET_CONFIG check (for known datasets)
    2. Whitelist matching (case-insensitive SMILES_COLUMNS)
    3. RDKit validation (sample rows, check >80% valid)
    4. Fallback to first column as SMILES

    Args:
        df: Loaded pandas DataFrame
        dataset_name: Optional dataset name to match against DATASET_CONFIG

    Returns:
        ColumnMapping: smiles_col, label_cols, detection_method, confidence

    Raises:
        ValueError: If no valid mapping can be detected
    """
    # 1. DATASET_CONFIG check (case-insensitive dataset name match)
    if dataset_name:
        for name, config in DATASET_CONFIG.items():
            if name.lower() == dataset_name.lower():
                smiles_col = config["smiles_col"]
                label_cols = config.get("label_cols", [])
                if smiles_col in df.columns:
                    return ColumnMapping(
                        smiles_col=smiles_col,
                        label_cols=label_cols,
                        detection_method="dataset_config",
                        confidence=1.0,
                    )

    # 2. Whitelist matching (case-insensitive)
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in SMILES_COLUMNS:
            label_cols = [
                c
                for c in df.columns
                if c != col and pd.api.types.is_numeric_dtype(df[c])
            ]
            return ColumnMapping(
                smiles_col=col,
                label_cols=label_cols,
                detection_method="whitelist",
                confidence=1.0,
            )

    # 2. RDKit validation (sample first 20 rows)
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20)
            valid_count = sum(validate_smiles_internal(s) for s in sample)
            if valid_count >= 16:  # >80%
                label_cols = [
                    c
                    for c in df.columns
                    if c != col and pd.api.types.is_numeric_dtype(df[c])
                ]
                return ColumnMapping(
                    smiles_col=col,
                    label_cols=label_cols,
                    detection_method="rdkit_validation",
                    confidence=valid_count / max(len(sample), 1),
                )

    # 3. Fallback: first column is SMILES, rest are numeric labels
    if len(df.columns) >= 2:
        smiles_col = df.columns[0]
        label_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
        return ColumnMapping(
            smiles_col=smiles_col,
            label_cols=label_cols,
            detection_method="fallback_first_column",
            confidence=0.5,
        )

    raise ValueError(
        f"Cannot detect column mapping. DataFrame columns: {df.columns.tolist()}"
    )


@dataclass
class Data:
    smiles: str
    labels: List[float]


class MoleculeTokenizer:
    """SMILES tokenizer with encode/decode methods."""

    def __init__(
        self,
        given_vocab_dict: Optional[Dict[str, int]] = None,
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
        return tokenize_smiles_cached_internal(smiles, id(self.vocab), max_length)

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
            raise FileNotFoundError(f"Data file not found: {data_file_path}")
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
        item = self.data[idx]
        token_ids = tokenize_smiles_cached_internal(
            item.smiles,
            self.vocab_id,
            self.max_length,
        )
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


def validate_smiles_internal(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


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

    def make_file_loader(
        path: str, is_train: bool = False
    ) -> torch.utils.data.DataLoader:
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

    def make_db_loader(
        dataset_name: str, is_train: bool = False
    ) -> torch.utils.data.DataLoader:
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
        assert train_path is not None, (
            "train_path required when train_dataset_name not provided"
        )
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

    return train_loader, val_loader, test_loader, normalizer


_SPLIT_SEED_FILE = ".split_seed"


def get_next_split_seed() -> int:
    """Get and increment next split seed."""
    try:
        with open(_SPLIT_SEED_FILE, "r") as f:
            seed = int(f.read().strip())
    except FileNotFoundError:
        seed = 42

    with open(_SPLIT_SEED_FILE, "w") as f:
        f.write(str(seed + 1))

    return seed


def get_current_split_seed() -> int:
    """Get current split seed without incrementing."""
    try:
        with open(_SPLIT_SEED_FILE, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 42


def random_split_dataset(
    input_csv: str,
    output_dir: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split CSV dataset into train/val/test.

    Args:
        input_csv: Input CSV file path
        output_dir: Output directory (optional, saves files if specified)
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.1)
        seed: Random seed (default None uses numpy default)
        n_jobs: Number of threads for CSV reading.
                Default None uses os.cpu_count() or 4.

    Returns:
        (train_df, val_df, test_df) tuple of DataFrames

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    if n_jobs is None:
        n_jobs = os.cpu_count() or 4

    from concurrent.futures import ThreadPoolExecutor

    def read_chunk(args):
        start, end = args
        return pd.read_csv(
            input_csv,
            skiprows=range(1, start + 1) if start > 0 else None,
            nrows=end - start,
        )

    total_lines = sum(1 for _ in open(input_csv)) - 1
    chunk_size = max(1, total_lines // n_jobs)
    chunks = []
    for i in range(n_jobs):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_lines) if i < n_jobs - 1 else total_lines
        if start < total_lines:
            chunks.append((start, end))

    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(read_chunk, chunks))
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.read_csv(input_csv)

    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    relative_val_ratio = val_ratio / val_test_ratio if val_test_ratio > 0 else 0.5
    val_df, test_df = train_test_split(
        val_test_df,
        train_size=relative_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.csv")
        val_path = os.path.join(output_dir, "val.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df


def list_available_databases(
    database_dir: str = "src/data/database",
) -> list[str]:
    """List all available database files in directory.

    Args:
        database_dir: Path to database directory

    Returns:
        List of database file paths
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
    """Interactively select a database file.

    Args:
        database_dir: Path to database directory

    Returns:
        Selected database file path

    Raises:
        FileNotFoundError: No database files found
    """
    db_files = list_available_databases(database_dir)

    if not db_files:
        raise FileNotFoundError(
            f"Database directory empty or not found: {database_dir}"
        )

    print("Available databases:")
    for i, db_path in enumerate(db_files, 1):
        print(f"  [{i}] {os.path.basename(db_path)}")

    while True:
        try:
            choice = int(input("\nSelect database number: "))
            if 1 <= choice <= len(db_files):
                return db_files[choice - 1]
            print(f"Invalid choice, enter 1-{len(db_files)}")
        except ValueError:
            print("Enter a valid number")
