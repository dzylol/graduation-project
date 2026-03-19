"""
Molecular dataset handling module.

This module provides functionality for:
1. Loading molecular data (CSV, JSON, TXT formats)
2. Tokenizing SMILES strings
3. Validating molecular structures with RDKit
4. Creating PyTorch data loaders
"""

# ============================================================================
# Imports
# ============================================================================
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import os
import pandas as pd
import numpy as np
import torch
from torch import Tensor  # PyTorch张量类型，用于模型输入输出
from torch.utils.data import Dataset  # PyTorch数据集基类

# RDKit用于化学结构验证和SMILES处理
from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================================
# MoleculeDataset class
# ============================================================================
class MoleculeDataset(Dataset):
    """
    分子数据集类,继承自PyTorch的--Dataset--基类。from torch.utils.data.Dataset

    - 加载CSV/JSON/TXT格式的分子数据
    - 使用RDKit验证SMILES字符串的有效性
    - 将SMILES字符串分词为固定长度的token ID序列
    - 返回PyTorch张量用于模型训练
    """

    def __init__(
        self,
        data_path: str,
        task_type: str = "regression",
        max_length: int = 512,
        tokenizer: Optional[object] = None,
        cache_smiles: bool = True,
        validate_smiles: bool = True,
    ) -> None:
        """
        初始化数据集。

        Args:
            data_path: 数据文件路径（支持CSV、JSON、TXT格式）
            task_twype: 任务类型（"regression"回归或"classification"分类）
            max_length: 最大序列长度（超过此长度的序列会被截断）
            tokenizer: 自定义分词器（默认为内置分词器）
            cache_smiles: 是否缓存已处理的SMILES以加速读取
            validate_smiles: 是否使用RDKit验证SMILES字符串的有效性
        """
        self.task_type = task_type
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache_smiles = cache_smiles
        self.validate_smiles = validate_smiles

        # 初始化词汇表
        self.vocab = self._init_vocab()
        self.vocab_size = len(self.vocab)

        # 加载数据
        self.data = self._load_data(data_path)

        # 初始化SMILES缓存（始终为字典，启用缓存时用于存储已处理的SMILES）
        self.smiles_cache: Dict[str, List[int]] = {} if cache_smiles else {}

    def _init_vocab(self) -> Dict[str, int]:
        """
        初始化SMILES字符到ID的映射词汇表。

        Returns:
            词汇表字典 {token: id}
        """
        # 常见SMILES字符（包括原子符号、化学键、括号等）
        smiles_elements: List[str] = [
            "(",  # 左括号
            ")",  # 右括号
            "[",  # 左方括号
            "]",  # 右方括号
            "=",  # 双键
            "#",  # 三键
            "%",  # 环编号开始
            "0",  # 环编号
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "+",  # 正电荷
            "-",  # 负电荷/单键
            "/",  # 立体化学信息
            ".",  # 断键
            ":",  # 芳香键
            ";",  # 立体信息
            "<",  # 特殊环开始
            ">",  # 特殊环结束
            "@",  # 立体化学标记
            # 元素符号
            "B",
            "Br",  # 溴（双字符需优先匹配）
            "C",
            "Cl",  # 氯（双字符需优先匹配）
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

        # 特殊token（填充符、开始符、结束符等）
        special_tokens: List[str] = ["<pad>", ">", "<bos>", "<eos>"]
        # <pad> (Padding) 填充符
        # > (Separator) 分隔符。用于分隔不同类型的token。
        # <bos> (Beginning of Sequence) 开始序列
        # <eos> (End of Sequence) 结束序列

        # 构建词汇表：特殊token在前，普通字符在后
        vocab: Dict[str, int] = {token: idx for idx, token in enumerate(special_tokens)}
        '''enumerate 是 Python 的一个内置函数，它的作用是**“边数数边取值”。
        假设 special_tokens 是 ["<pad>", ">", "<bos>", "<eos>"],
        那么 enumerate 会把它变成一组组带有编号的配对：

        (0, "<pad>")

        (1, ">")

        (2, "<bos>")

        (3, "<eos>")
        token: idx:意思是“把字符(token)作为键(Key)，把编号(idx)作为值(Value)”。
        '''
        vocab.update(
            {char: idx + len(special_tokens) for idx, char in enumerate(smiles_elements)}
        )
        '''offset = len(special_tokens) 

# 2. 开始给化学符号排队
# 我们用最原始的 range(len(...)) 方式来数数
        for i in range(len(chars)):
    # 取出当前的化学符号，比如 "(" 或 "C"
            char = chars[i]
    # 计算这个符号应该分配的 ID
    # 逻辑：当前的序号 + 之前被占掉的位置
            new_id = i + offset
    # 把这个对应关系存进词汇表字典里
    # 这行等价于 vocab.update(...)
            vocab[char] = new_id
    '''

        return vocab

    def _load_data(self, data_path: str) -> List[dict]:
        """
        从CSV、JSON或TXT文件加载分子数据。

        Args:
            data_path: 数据文件路径

        Returns:
            包含"smiles"和"labels"键的字典列表
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        data: List[dict] = []  #List: 说明 data 是一个列表（数组）。
                               #[dict]: 说明这个列表里的每一个元素都必须是一个字典。

        # 根据文件扩展名加载数据
        if data_path.endswith(".csv"):
            # CSV格式：第一列为SMILES，后续列为标签
            df = pd.read_csv(data_path)
            smiles_col = df.columns[0]
            label_cols = df.columns[1:] if len(df.columns) > 1 else []

            for _, row in df.iterrows():
                smiles = str(row[smiles_col])
                if len(label_cols) > 0:
                    labels = [float(row[col]) for col in label_cols]
                else:
                    labels = [0.0]
                data.append({"smiles": smiles, "labels": labels})

        elif data_path.endswith(".json"):
            # JSON格式：{"smiles": "...", "labels": [...]}对象列表
            with open(data_path, "r") as f:
                data = json.load(f)

        elif data_path.endswith(".txt"):
            # TXT格式：每行"SMILES,label1,label2,..."
            with open(data_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        smiles = parts[0]
                        labels = [float(x) for x in parts[1:]]
                        data.append({"smiles": smiles, "labels": labels})
                    else:
                        data.append({"smiles": parts[0], "labels": [0.0]})

        else:
            raise ValueError(f"不支持的文件格式: {data_path}")

        # 如果启用验证，使用RDKit过滤无效SMILES
        if self.validate_smiles:
            original_len = len(data)
            data = [item for item in data if self._validate_smiles(item["smiles"])]
            if len(data) < original_len:
                print(f"已过滤{original_len - len(data)}个无效SMILES字符串")

        return data

    def _validate_smiles(self, smiles: str) -> bool:
        """
        使用RDKit验证SMILES字符串的有效性。

        Args:
            smiles: SMILES字符串

        Returns:
            有效返回True，无效返回False
        """
        try:
            # MolFromSmiles将SMILES转换为分子对象，失败返回None
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def _tokenize_smiles(self, smiles: str) -> List[int]:
        """
        将SMILES字符串转换为token ID列表。

        Args:
            smiles: SMILES字符串

        Returns:
            token ID列表
        """
        # 优先检查缓存，避免重复分词
        if self.cache_smiles and smiles in self.smiles_cache:
            return self.smiles_cache[smiles]

        tokens: List[int] = []
        i: int = 0

        # 逐字符遍历SMILES字符串进行分词
        while i < len(smiles):
            # 优先匹配双字符token（如"Br"、"Cl"等元素符号）
            if i + 1 < len(smiles) and smiles[i : i + 2] in self.vocab:
                tokens.append(self.vocab[smiles[i : i + 2]])
                i += 2
            # 然后匹配单字符token
            elif smiles[i] in self.vocab:
                tokens.append(self.vocab[smiles[i]])
                i += 1
            # 未知字符用<pad>代替
            else:
                tokens.append(self.vocab["<pad>"])
                i += 1

        # 处理序列长度：截断或填充到max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            pad_token: int = self.vocab["<pad>"]
            tokens = tokens + [pad_token] * (self.max_length - len(tokens))

        # 更新缓存
        if self.cache_smiles:
            self.smiles_cache[smiles] = tokens

        return tokens

    def __len__(self) -> int:
        """返回数据集中样本的数量。"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            (input_ids, labels)张量元组
        """
        item = self.data[idx]
        smiles = item["smiles"]
        labels = item["labels"]

        # 将SMILES分词为token ID序列
        token_ids = self._tokenize_smiles(smiles)

        # 转换为PyTorch张量
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # 根据任务类型选择标签张量的数据类型
        if self.task_type == "regression":
            labels_tensor = torch.tensor(labels, dtype=torch.float)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels_tensor

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def get_pad_token_id(self) -> int:
        """Return the ID of the padding token."""
        return self.vocab["<pad>"]


# ============================================================================
# Data loader creation function
# ============================================================================
def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    task_type: str = "regression",
    max_length: int = 512,
    num_workers: int = 4,
    tokenizer: Optional[object] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    创建训练、验证和测试用的PyTorch DataLoader。

    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径（可选）
        test_path: 测试数据路径（可选）
        batch_size: 每批样本数量
        task_type: 任务类型（"regression"或"classification"）
        max_length: 最大序列长度
        num_workers: 数据加载的工作进程数
        tokenizer: 自定义分词器

    Returns:
        (train_loader, val_loader, test_loader)元组
    """
    # 创建训练数据集和DataLoader
    train_dataset = MoleculeDataset(
        data_path=train_path,
        task_type=task_type,
        max_length=max_length,
        tokenizer=tokenizer,
    )

    # 验证数据集（可选）
    val_dataset: Optional[MoleculeDataset] = None
    if val_path and os.path.exists(val_path):
        val_dataset = MoleculeDataset(
            data_path=val_path,
            task_type=task_type,
            max_length=max_length,
            tokenizer=tokenizer,
        )

    # 测试数据集（可选）
    test_dataset: Optional[MoleculeDataset] = None
    if test_path and os.path.exists(test_path):
        test_dataset = MoleculeDataset(
            data_path=test_path,
            task_type=task_type,
            max_length=max_length,
            tokenizer=tokenizer,
        )

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=True,  # 加速CPU到GPU的数据传输
    )

    val_loader: Optional[torch.utils.data.DataLoader] = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证时不打乱
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader: Optional[torch.utils.data.DataLoader] = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # 测试时不打乱
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


# ============================================================================
# Collate function for dynamic padding
# ============================================================================
def collate_fn(batch):
    """
    自定义collate函数，用于处理变长序列的批量填充。

    Args:
        batch: (input_ids, labels)元组列表

    Returns:
        (input_ids_batch, labels_batch)元组
    """
    input_ids, labels = zip(*batch)

    # 使用pad_sequence将不同长度的序列填充到相同长度
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )

    # 堆叠标签张量
    labels = torch.stack(labels, dim=0)

    return input_ids, labels


# ============================================================================
# Independent tokenizer class
# ============================================================================
class MoleculeTokenizer:
    """
    独立的SMILES分词器类，提供encode/decode方法。

    该类可单独使用，不依赖MoleculeDataset。
    """

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None) -> None:
        """
        初始化分词器。

        Args:
            vocab_dict: 可选的词汇表字典（token字符串 -> ID映射）
        """
        if vocab_dict is None:
            # 使用默认词汇表
            smiles_elements: List[str] = [
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

            vocab_local: Dict[str, int] = {
                token: idx for idx, token in enumerate(special_tokens)
            }
            vocab_local.update(
                {char: idx + len(special_tokens) for idx, char in enumerate(smiles_elements)}
            )
        else:
            vocab_local = vocab_dict

        # 正向词汇表：token字符串 -> 整数ID
        self.vocab: Dict[str, int] = vocab_local
        # 反向词汇表：整数ID -> token字符串
        self.inverse_vocab: Dict[int, str] = {
            idx: token for token, idx in vocab_local.items()
        }
        self.vocab_size: int = len(vocab_local)

    def encode(self, smiles: str, max_length: int = 512) -> List[int]:
        """
        将SMILES字符串编码为token ID列表。

        Args:
            smiles: SMILES字符串
            max_length: 输出序列的最大长度

        Returns:
            token ID列表
        """
        tokens: List[int] = []
        i: int = 0
        while i < len(smiles):
            # 优先匹配双字符token（如"Br"、"Cl"）
            if i + 1 < len(smiles) and smiles[i : i + 2] in self.vocab:
                tokens.append(self.vocab[smiles[i : i + 2]])
                i += 2
            # 匹配单字符token
            elif smiles[i] in self.vocab:
                tokens.append(self.vocab[smiles[i]])
                i += 1
            # 未知字符用<pad>代替
            else:
                tokens.append(self.vocab["<pad>"])
                i += 1

        # 处理序列长度
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            pad_token_id: int = self.vocab["<pad>"]
            tokens = tokens + [pad_token_id] * (max_length - len(tokens))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        将token ID列表解码回SMILES字符串。

        Args:
            token_ids: token ID列表

        Returns:
            解码后的SMILES字符串
        """
        tokens: List[str] = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token: str = self.inverse_vocab[token_id]
                # 解码时跳过特殊token
                if token not in ["<pad>", ">", "<bos>", "<eos>"]:
                    tokens.append(token)
            else:
                tokens.append("")
        return "".join(tokens)
