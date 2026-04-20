"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              molecule_dataset.py — 教学注释版                               ║
║   用途：SMILES 数据加载、词表、tokenizer、Dataset、DataLoader、归一化        ║
╚══════════════════════════════════════════════════════════════════════════════╝

【数据流全景】

    CSV 文件 (smiles, label)
        │
        ▼
    scaffold_split_dataset()     ← 按分子骨架切分 train/val/test（比随机更严格）
        │
        ▼
    MoleculeDataset.__init__()   ← 读取 CSV，RDKit 验证 SMILES，存为 List[Data]
        │
        ▼
    MoleculeDataset.__getitem__()
        ├─ tokenize_smiles_cached_internal()  ← 把 SMILES 字符串 → token id tuple
        └─ torch.tensor(token_ids), torch.tensor(labels)
        │
        ▼
    NormalizedDataset.__getitem__()   ← （可选）对 labels 做 Z-score 归一化
        │
        ▼
    DataLoader(dataset, batch_size=32, shuffle=True)
        │
        ▼
    训练循环 for input_ids, labels in train_loader: ...
"""

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


# ═══════════════════════════════════════════════════════════
# 词表（Vocabulary）
# ═══════════════════════════════════════════════════════════

# SMILES 化学符号词表：41 个 token
# 注意"Br"、"Cl"、"Si"等双字符 token 排在单字符前面
# tokenizer 会优先匹配双字符（贪心匹配）
smiles_token_tuple: tuple[str, ...] = (
    "(", ")", "[", "]", "=", "#", "%",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "/", ".", ":", ";", "<", ">", "@",
    "B", "Br",   # ← Br 在 B 后面，tokenizer 需要先试匹配 2 字符
    "C", "Cl",   # ← 同理
    "F", "H", "I", "N", "O", "P", "S", "Si", "Te", "Se", "At",
)

# 特殊 token（id 0~3）
# <pad> = 0：序列补全用
# <unk> = 1：未知字符用（当前 tokenizer 实际用 <pad> 代替未知）
# <bos> = 2：序列开始（当前未显式添加）
# <eos> = 3：序列结束（当前未显式添加）
special_token_tuple: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")

# 常见的 SMILES 列名（不区分大小写）
SMILES_COLUMNS: set[str] = {
    "smiles", "smile", "canonical_smiles", "canonical",
    "smi", "structure", "molecule", "mol", "chem_smile",
}

# 忽略的数据集（没有 SMILES 列）
IGNORED_DATASETS: set[str] = {"sider"}


def build_default_vocab() -> Dict[str, int]:
    """
    构建默认词表：特殊 token (id 0-3) + SMILES token (id 4-44)。

    例：
        "<pad>" → 0
        "<unk>" → 1
        "C"     → 28   (特殊4个 + C在smiles_token_tuple中的索引24 = 28)
        "Cl"    → 29
    """
    vocab: Dict[str, int] = {token: idx for idx, token in enumerate(special_token_tuple)}
    vocab.update(
        {char: idx + len(special_token_tuple) for idx, char in enumerate(smiles_token_tuple)}
    )
    return vocab


default_vocab: Dict[str, int] = build_default_vocab()
default_vocab_size: int = len(default_vocab)  # = 4 + 41 = 45

# 已知数据集的列配置（用于自动检测）
DATASET_CONFIG: Dict[str, Dict] = {
    "ESOL":          {"smiles_col": "SMILES",  "label_cols": ["measured log(solubility:mol/L)"]},
    "HIV":           {"smiles_col": "smiles",  "label_cols": ["HIV_active"]},
    "Lipophilicity": {"smiles_col": "smiles",  "label_cols": ["exp"]},
    "ZINC250K":      {"smiles_col": "smiles",  "label_cols": ["logP", "qed", "SAS"]},
    "FreeSolv":      {"smiles_col": "smiles",  "label_cols": ["freesolv"]},
    "BBBP":          {"smiles_col": "smiles",  "label_cols": ["p_np"]},
    "ClinTox":       {"smiles_col": "smiles",  "label_cols": ["FDA_APPROVED", "CT_TOX"]},
    "MUV":           {"smiles_col": "smiles",  "label_cols": []},
    "mpro":          {"smiles_col": "smiles",  "label_cols": ["IC50"]},
    "EGFR":          {"smiles_col": "smiles",  "label_cols": ["value"]},
    "bace":          {"smiles_col": "smiles",  "label_cols": ["pIC50"]},
}


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class ColumnMapping:
    """CSV 列映射检测结果（哪列是 SMILES，哪列是标签）。"""
    smiles_col: str
    label_cols: List[str]
    detection_method: str  # "dataset_config" / "whitelist" / "rdkit_validation" / "fallback"
    confidence: float = 1.0


@dataclass
class Data:
    """单条分子数据。"""
    smiles: str
    labels: List[float]


# ═══════════════════════════════════════════════════════════
# 列自动检测
# ═══════════════════════════════════════════════════════════

def detect_column_mapping(df: pd.DataFrame, dataset_name: Optional[str] = None) -> ColumnMapping:
    """
    自动检测 CSV 的 SMILES 列和标签列，检测顺序：
        1. DATASET_CONFIG（已知数据集，直接查表）
        2. 白名单匹配（列名在 SMILES_COLUMNS 中）
        3. RDKit 验证（取前 20 行，>80% 有效则认为是 SMILES 列）
        4. 兜底：第一列是 SMILES，其余数值列是标签
    """
    # 1. 已知数据集直接查配置
    if dataset_name:
        for name, config in DATASET_CONFIG.items():
            if name.lower() == dataset_name.lower():
                smiles_col = config["smiles_col"]
                if smiles_col in df.columns:
                    return ColumnMapping(
                        smiles_col=smiles_col,
                        label_cols=config.get("label_cols", []),
                        detection_method="dataset_config",
                    )

    # 2. 白名单匹配（列名包含常见 SMILES 关键词）
    for col in df.columns:
        if col.strip().lower() in SMILES_COLUMNS:
            label_cols = [c for c in df.columns
                          if c != col and pd.api.types.is_numeric_dtype(df[c])]
            return ColumnMapping(smiles_col=col, label_cols=label_cols,
                                 detection_method="whitelist")

    # 3. RDKit 验证（取样检测哪列是合法 SMILES）
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(20)
            valid_count = sum(validate_smiles_internal(s) for s in sample)
            if valid_count >= 16:  # >80% 有效
                label_cols = [c for c in df.columns
                              if c != col and pd.api.types.is_numeric_dtype(df[c])]
                return ColumnMapping(smiles_col=col, label_cols=label_cols,
                                     detection_method="rdkit_validation",
                                     confidence=valid_count / max(len(sample), 1))

    # 4. 兜底：第一列是 SMILES
    if len(df.columns) >= 2:
        smiles_col = df.columns[0]
        label_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
        return ColumnMapping(smiles_col=smiles_col, label_cols=label_cols,
                             detection_method="fallback_first_column", confidence=0.5)

    raise ValueError(f"Cannot detect column mapping. Columns: {df.columns.tolist()}")


# ═══════════════════════════════════════════════════════════
# SMILES 验证
# ═══════════════════════════════════════════════════════════

def validate_smiles_internal(smiles: str) -> bool:
    """
    用 RDKit 验证 SMILES 是否合法。
    RDKit 会尝试解析分子图，失败则返回 None。

    例：
        validate_smiles_internal("CCO")   → True  (乙醇)
        validate_smiles_internal("XYZ")   → False (无效)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=500000)
def tokenize_smiles_cached_internal(smiles: str, vocab_id: int, max_length: int) -> Tuple[int, ...]:
    """
    SMILES 字符串 → token id 元组（可缓存）。

    ⚠️ 返回 Tuple（不是 List），因为 lru_cache 要求参数和返回值都可哈希。
    ⚠️ vocab_id 用 id(vocab) 传入，确保不同词表不混用缓存。

    贪心双字符优先匹配：
        "ClC" → 先匹配 "Cl"（双字符），再匹配 "C"（单字符）
        不贪心的话 "Cl" 会被错误拆成 "C" + "l"

    处理逻辑：
        1. 尝试匹配 smiles[i:i+2]（双字符）
        2. 失败则匹配 smiles[i]（单字符）
        3. 再失败则用 <pad> 代替（处理未知字符）
        4. 补 padding 到 max_length

    例（简化词表）：
        "CCO" → [28, 28, 34, 0, 0, ..., 0]  (28=C, 34=O, 0=<pad>)
    """
    given_vocab_dict = default_vocab if vocab_id == id(default_vocab) else {}
    tokens: List[int] = []
    i = 0

    while i < len(smiles):
        # 优先匹配双字符（如 Br, Cl, Si）
        if i + 1 < len(smiles) and smiles[i: i + 2] in given_vocab_dict:
            tokens.append(given_vocab_dict[smiles[i: i + 2]])
            i += 2
        elif smiles[i] in given_vocab_dict:
            tokens.append(given_vocab_dict[smiles[i]])
            i += 1
        else:
            # 未知字符用 <pad> 代替（id=0）
            tokens.append(given_vocab_dict["<pad>"])
            i += 1

    pad_token_id = given_vocab_dict["<pad>"]   # = 0

    # 截断或补 padding
    if len(tokens) > max_length:
        return tuple(tokens[:max_length])
    return tuple(tokens + [pad_token_id] * (max_length - len(tokens)))


class MoleculeTokenizer:
    """
    SMILES tokenizer：字符串 ↔ token id 序列。

    示例：
        tokenizer = MoleculeTokenizer()
        ids = tokenizer.encode("CCO")        # (28, 28, 34, 0, 0, ...) 长512
        smiles = tokenizer.decode(ids)       # "CCO"（去掉 pad/特殊 token）
    """

    def __init__(self, given_vocab_dict: Optional[Dict[str, int]] = None) -> None:
        self.vocab = given_vocab_dict if given_vocab_dict else default_vocab
        # 反向词表：id → token，用于 decode
        self.inverse_vocab: Dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, smiles: str, max_length: int = 512) -> Tuple[int, ...]:
        """SMILES → token id tuple（带 lru_cache 加速）。"""
        return tokenize_smiles_cached_internal(smiles, id(self.vocab), max_length)

    def decode(self, token_ids: List[int]) -> str:
        """token id list → SMILES（跳过特殊 token）。"""
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, "")
            if token not in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                tokens.append(token)
        return "".join(tokens)


# ═══════════════════════════════════════════════════════════
# 归一化
# ═══════════════════════════════════════════════════════════

class LabelNormalizer:
    """
    Z-score 归一化：将标签变换到均值=0、方差=1。

    为什么需要归一化？
        - 不同数据集的标签范围差异很大（logP: -5~10, IC50: 0~1000）
        - 归一化后 MSELoss 的量纲一致，学习率等超参数更好调
        - 预测时用 inverse_transform 还原到原始量纲（报告中的 RMSE 是还原后的）

    ⚠️ 只在训练集上 fit，然后用同一个 normalizer transform val/test
       （防止数据泄露）

    使用流程：
        normalizer = LabelNormalizer()
        normalizer.fit(train_labels)          # 从训练集计算 mean, std
        y_train_norm = normalizer.transform(train_labels)
        y_val_norm   = normalizer.transform(val_labels)  # 用训练集的统计量
        # 预测完后：
        y_pred_orig = normalizer.inverse_transform(y_pred_norm)
    """

    def __init__(self) -> None:
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self._fitted: bool = False

    def fit(self, labels: np.ndarray) -> "LabelNormalizer":
        self.mean = float(np.mean(labels))
        self.std = float(np.std(labels))
        if self.std < 1e-8:       # 防止除以 0（标签全相同的极端情况）
            self.std = 1.0
        self._fitted = True
        return self                # 支持链式调用：normalizer.fit(x).transform(x)

    def transform(self, labels: np.ndarray) -> np.ndarray:
        """(label - mean) / std"""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (labels - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """normalized * std + mean（还原到原始量纲）"""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return normalized * self.std + self.mean

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class NormalizedDataset(Dataset):
    """
    对 MoleculeDataset 的 labels 应用 Z-score 归一化的包装器。

    设计模式：Decorator（装饰器）
    - 不修改原始 dataset，只在 __getitem__ 时动态归一化标签
    - 原始 SMILES token 不变，只有 labels 被归一化

    使用示例：
        base_dataset = MoleculeDataset("train.csv")
        normalizer = LabelNormalizer().fit(all_train_labels)
        norm_dataset = NormalizedDataset(base_dataset, normalizer)
        # norm_dataset[i] 返回的 labels 已经归一化
    """

    def __init__(self, base_dataset: "MoleculeDataset", normalizer: LabelNormalizer) -> None:
        self.base_dataset = base_dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        input_ids, labels = self.base_dataset[idx]
        # 对 labels 做归一化（labels 是 float tensor，先转 numpy 再转回）
        normalized_labels = self.normalizer.transform(labels.numpy())
        return input_ids, torch.tensor(normalized_labels, dtype=torch.float)


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class MoleculeDataset(Dataset):
    """
    分子数据集：从 CSV/JSON/TXT 加载，支持 RDKit 验证和自动列检测。

    PyTorch Dataset 协议（必须实现）：
        __len__()     → 数据集大小
        __getitem__(i)→ 第 i 条数据 (input_ids, labels)

    数据加载流程：
        __init__ → load_data_internal → load_csv_internal
                → (可选) RDKit 过滤无效 SMILES
                → 存为 self.data: List[Data]

        __getitem__ → tokenize_smiles_cached_internal → torch.tensor
    """

    def __init__(
        self,
        data_file_path: str,
        task_type: str = "regression",
        max_length: int = 512,
        validate_smiles: bool = True,   # 是否用 RDKit 过滤无效 SMILES
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
        self.vocab_id = id(default_vocab)   # 用 id 而非 dict 本身（lru_cache 需要可哈希 key）
        self.data = self.load_data_internal(data_file_path)

    def load_csv_internal(self, path: str) -> List[Data]:
        """
        读取 CSV，自动检测列，返回 List[Data]。

        如果没有指定 smiles_col/label_cols，调用 detect_column_mapping 自动检测。
        NaN 标签行跳过，缺标签的行用 [0.0] 占位。
        """
        df = pd.read_csv(path)

        if self.smiles_col is not None:
            smiles_col = self.smiles_col
            label_cols = self.label_cols if self.label_cols else []
        else:
            # 自动检测列（见 detect_column_mapping）
            mapping = detect_column_mapping(df, dataset_name=self.dataset_name)
            smiles_col = mapping.smiles_col
            label_cols = mapping.label_cols

        smiles_list = df[smiles_col].astype(str).tolist()

        if label_cols:
            labels_list = []
            for idx in range(len(smiles_list)):
                row_labels = [float(df[col].iloc[idx])
                              for col in label_cols
                              if not pd.isna(df[col].iloc[idx])]
                labels_list.append(row_labels if row_labels else [0.0])
        else:
            labels_list = [[0.0]] * len(smiles_list)

        return [Data(smiles=s, labels=l) for s, l in zip(smiles_list, labels_list)]

    def load_json_internal(self, path: str) -> List[Data]:
        """读取 JSON 格式：[{"smiles": "CCO", "labels": [-2.5]}, ...]"""
        with open(path, "r") as f:
            raw = json.load(f)
        return [Data(smiles=item["smiles"], labels=item["labels"]) for item in raw]

    def load_txt_internal(self, path: str) -> List[Data]:
        """读取 TXT 格式：每行 smiles,label1,label2,..."""
        data = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                data.append(Data(
                    smiles=parts[0],
                    labels=[float(x) for x in parts[1:]] if len(parts) >= 2 else [0.0],
                ))
        return data

    def load_data_internal(self, data_file_path: str) -> List[Data]:
        """
        根据文件扩展名选择加载方式，并可选做 RDKit 验证。

        RDKit 验证会过滤掉无效 SMILES（返回 None 的）。
        训练集约 41127 条（HIV），验证通常过滤 <0.1%。
        """
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found: {data_file_path}")

        ext = data_file_path[data_file_path.rfind("."):] if "." in data_file_path else ""
        if ext == ".csv":
            data = self.load_csv_internal(data_file_path)
        elif ext == ".json":
            data = self.load_json_internal(data_file_path)
        elif ext == ".txt":
            data = self.load_txt_internal(data_file_path)
        else:
            raise ValueError(f"Unsupported file format: {data_file_path}")

        if self.validate_smiles:
            original_len = len(data)
            data = [item for item in data if validate_smiles_internal(item.smiles)]
            removed = original_len - len(data)
            if removed > 0:
                print(f"Filtered {removed} invalid SMILES strings")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        返回第 idx 条数据的 (input_ids, labels)。

        input_ids: LongTensor (max_length,)   token id 序列（含 padding）
        labels:    FloatTensor (num_labels,)   真实标签值
        """
        item = self.data[idx]

        # tokenize：SMILES → token id tuple（带 lru_cache，同一 SMILES 只算一次）
        token_ids = tokenize_smiles_cached_internal(
            item.smiles, self.vocab_id, self.max_length
        )
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels_tensor = torch.tensor(item.labels, dtype=torch.float)
        return input_ids, labels_tensor

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size   # 45

    def get_pad_token_id(self) -> int:
        return self.tokenizer.vocab["<pad>"]   # 0


# ═══════════════════════════════════════════════════════════
# DataLoader 工厂函数
# ═══════════════════════════════════════════════════════════

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
    """
    创建 train/val/test DataLoader，并处理 Z-score 归一化。

    两种模式（互斥）：
        文件模式：  train_path="dataset/HIV/train.csv"
        数据库模式：train_dataset_name="HIV", db_path="bi_mamba_chem.db"

    归一化流程：
        1. 遍历 train_loader 收集所有 labels
        2. LabelNormalizer.fit(train_labels)  ← 只从训练集计算 mean/std
        3. 用 NormalizedDataset 包装 train/val/test
           val/test 用训练集的 mean/std（防止数据泄露）

    返回：
        (train_loader, val_loader, test_loader, normalizer)
        回归 + normalize=True  → normalizer 是 LabelNormalizer
        分类 或 normalize=False → normalizer 是 None

    使用示例：
        train_loader, val_loader, test_loader, normalizer = create_data_loaders(
            train_path="dataset/HIV/train.csv",
            val_path="dataset/HIV/val.csv",
            test_path="dataset/HIV/test.csv",
            batch_size=32,
            task_type="classification",
            normalize=False,
        )
    """
    normalizer: Optional[LabelNormalizer] = None

    if train_path and train_dataset_name:
        raise ValueError("Cannot specify both train_path and train_dataset_name")
    if train_path is None and train_dataset_name is None:
        raise ValueError("Must specify either train_path or train_dataset_name")

    def _load_split_meta(output_dir: str) -> Optional[dict]:
        """从 scaffold_split 生成的 split_meta.json 读取列映射。"""
        meta_path = os.path.join(output_dir, "split_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return None

    # 尝试从 scaffold_split 的 meta 文件恢复列映射
    if smiles_col is None and label_cols is None and train_path:
        meta = _load_split_meta(os.path.dirname(train_path))
        if meta:
            smiles_col = meta.get("smiles_col")
            label_col_from_meta = meta.get("label_col")
            if label_col_from_meta:
                label_cols = [label_col_from_meta]

    def make_file_loader(path: str, is_train: bool = False) -> torch.utils.data.DataLoader:
        dataset = MoleculeDataset(
            data_file_path=path, task_type=task_type, max_length=max_length,
            smiles_col=smiles_col, label_cols=label_cols, dataset_name=dataset_name,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(path == train_path),  # 只有训练集 shuffle
            num_workers=num_workers,
            pin_memory=True,               # 锁定内存，加速 GPU 数据传输
            persistent_workers=True if num_workers > 0 and is_train else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    # 创建训练 loader
    if train_dataset_name:
        from src.data.molecule_dataset import DatabaseMoleculeDataset
        train_ds = DatabaseMoleculeDataset(
            dataset_name=train_dataset_name, db_path=db_path,
            task_type=task_type, max_length=max_length, property_name=property_name,
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers,
                                                   pin_memory=True)
    else:
        train_loader = make_file_loader(train_path, is_train=True)

    # Z-score 归一化（仅回归任务）
    if normalize and task_type == "regression":
        normalizer = LabelNormalizer()
        train_labels = []
        # 遍历一次训练集收集所有标签（用于计算 mean/std）
        for _, labels in train_loader:
            train_labels.extend(labels.numpy().flatten())
        normalizer.fit(np.array(train_labels))

        # 用 NormalizedDataset 重新包装训练集
        train_loader = torch.utils.data.DataLoader(
            NormalizedDataset(train_loader.dataset, normalizer),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    # 创建 val/test loader
    val_loader = make_file_loader(val_path) if val_path and os.path.exists(val_path) else None
    test_loader = make_file_loader(test_path) if test_path and os.path.exists(test_path) else None

    # 对 val/test 也应用同一个 normalizer
    if normalize and task_type == "regression" and normalizer is not None:
        for loader_ref, path in [(val_loader, val_path), (test_loader, test_path)]:
            if loader_ref is not None:
                wrapped = NormalizedDataset(loader_ref.dataset, normalizer)
                new_loader = torch.utils.data.DataLoader(
                    wrapped, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, persistent_workers=False,
                    prefetch_factor=4 if num_workers > 0 else None,
                )
                if path == val_path:
                    val_loader = new_loader
                else:
                    test_loader = new_loader

    return train_loader, val_loader, test_loader, normalizer


# ═══════════════════════════════════════════════════════════
# Scaffold Split（数据集划分）
# ═══════════════════════════════════════════════════════════

def scaffold_split_dataset(
    input_csv: str,
    output_dir: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
    smiles_col: str = "smiles",
    label_col: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按分子骨架（Murcko Scaffold）划分数据集。

    为什么用 Scaffold Split 而非随机划分？
    ┌─────────────────────────────────────────────────────────────┐
    │ 随机划分：训练集和测试集可能含大量相似分子（同一骨架）        │
    │ → 模型容易"记住"类似分子，AUC 虚高                          │
    │                                                             │
    │ Scaffold Split：不同骨架严格分开                            │
    │ → 测试集含训练时从未见过的化学骨架                          │
    │ → 更真实地反映新药发现场景（对全新骨架的预测能力）           │
    └─────────────────────────────────────────────────────────────┘

    Murcko Scaffold：保留分子的环系骨架，去掉侧链。
    例：阿司匹林和布洛芬的骨架不同 → 分入不同 split

    算法：
        1. 计算每个分子的 scaffold
        2. 按 scaffold 分组
        3. 打乱 scaffold 组的顺序（seed 固定可复现）
        4. 轮询分配（0→train, 1→val, 2→test）直到达到目标比例
    """
    from collections import defaultdict

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    df = pd.read_csv(input_csv)

    # 自动检测 SMILES 列名
    if smiles_col not in df.columns:
        for col in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
            if col in df.columns:
                smiles_col = col
                break
        else:
            raise ValueError(f"SMILES column not found. Available: {df.columns.tolist()}")

    def get_scaffold(smiles: str) -> str:
        """用 RDKit 提取 Murcko Scaffold SMILES。"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "INVALID"
            scaffold = Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            return scaffold if scaffold else "NO_SCAFFOLD"
        except Exception:
            return "ERROR"

    # 计算所有分子的 scaffold
    scaffolds = [get_scaffold(s) for s in df[smiles_col].tolist()]

    # 按 scaffold 分组：{scaffold_smiles: [index1, index2, ...]}
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices[scaffold].append(idx)

    # 打乱 scaffold 组的顺序
    np.random.seed(seed)
    shuffled = list(scaffold_to_indices.items())
    np.random.shuffle(shuffled)

    # 轮询分配到三个 split
    train_indices, val_indices, test_indices = [], [], []
    train_count = val_count = test_count = 0
    total = len(df)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    for scaff_idx, (scaffold, indices) in enumerate(shuffled):
        mod = scaff_idx % 3    # 0→train, 1→val, 2→test（循环分配）
        n = len(indices)
        if mod == 0 and train_count < train_target:
            train_indices.extend(indices); train_count += n
        elif mod == 1 and val_count < val_target:
            val_indices.extend(indices); val_count += n
        else:
            test_indices.extend(indices); test_count += n

    # 如果某个 split 为空（数据量太少时），回退到随机划分
    if val_count == 0 or test_count == 0:
        all_indices = train_indices + val_indices + test_indices
        np.random.shuffle(all_indices)
        n = len(all_indices)
        train_indices = all_indices[:train_target]
        val_indices   = all_indices[train_target: train_target + val_target]
        test_indices  = all_indices[train_target + val_target:]

    # 各 split 内部再打乱
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # 自动检测标签列
    if label_col is None:
        numeric_cols = [c for c in df.columns
                        if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1:
            label_col = numeric_cols[0]
        else:
            for col in ["measured log(solubility:mol/L)", "solubility", "label", "value"]:
                if col in numeric_cols:
                    label_col = col
                    break
            if label_col is None:
                raise ValueError(f"Could not auto-detect label column. Found: {numeric_cols}")

    cols_to_keep = [smiles_col, label_col]
    train_df = df.iloc[train_indices][cols_to_keep].reset_index(drop=True)
    val_df   = df.iloc[val_indices][cols_to_keep].reset_index(drop=True)
    test_df  = df.iloc[test_indices][cols_to_keep].reset_index(drop=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        # 保存元数据（供 create_data_loaders 自动读取列配置）
        with open(os.path.join(output_dir, "split_meta.json"), "w") as f:
            json.dump({"smiles_col": smiles_col, "label_col": label_col}, f)

    return train_df, val_df, test_df
