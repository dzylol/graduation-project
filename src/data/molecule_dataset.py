"""
Molecular dataset handling module.

This module provides functionality for:
1. Loading molecular data (CSV, JSON, TXT formats)
2. Tokenizing SMILES strings
3. Validating molecular structures with RDKit
4. Creating PyTorch data loaders
"""

from typing import Dict, List, Optional, Tuple

import json
import os
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rdkit import Chem


# SMILES vocabulary (shared by MoleculeTokenizer and MoleculeDataset)

# 常见SMILES字符（包括原子符号、化学键、括号等）
_SMILES_ELEMENTS: List[str] = [
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
_SPECIAL_TOKENS: List[str] = ["<pad>", ">", "<bos>", "<eos>"]
# <pad> (Padding) 填充符
# > (Separator) 分隔符。用于分隔不同类型的token。
# <bos> (Beginning of Sequence) 开始序列
# <eos> (End of Sequence) 结束序列


def _build_vocab() -> Dict[str, int]:
    """
    构建SMILES字符到ID的映射词汇表。

    返回类型注解 `-> Dict[str, int]` 说明这个函数返回一个字典：
    - key: str类型（token字符串，如 "C", "<pad>"）
    - value: int类型（对应的ID，如 0, 1, 2...）
    """
    vocab: Dict[str, int] = {token: idx for idx, token in enumerate(_SPECIAL_TOKENS)}
    vocab.update(
        {char: idx + len(_SPECIAL_TOKENS) for idx, char in enumerate(_SMILES_ELEMENTS)}
    )
    return vocab


# 全局词汇表（单例，所有实例共享）
_VOCAB: Dict[str, int] = _build_vocab()
_VOCAB_SIZE: int = len(_VOCAB)


# MoleculeTokenizer class //


class MoleculeTokenizer:
    """
    独立的SMILES分词器类，提供encode/decode方法。

    类与对象的关系：
    - 类(Class): 模板，定义有哪些属性(数据)和方法(函数)
    - 对象(Object): 类的实例，通过类创建的具体个体
    - self: 当前实例对象的引用，谁调用方法，self就是谁

    属性的定义与访问：
    - 定义: self.xxx = 值（在__init__等方法中）
    - 访问: self.xxx（在类的任何方法中）
    - 属性属于对象，每个实例的属性相互独立

    示例：
        tokenizer = MoleculeTokenizer()  # tokenizer是MoleculeTokenizer的对象
        tokenizer.encode("CC")           # 调用encode方法，self指向tokenizer
        tokenizer.inverse_vocab          # 访问inverse_vocab属性
    """

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None) -> None:
        """
        初始化分词器。

        Args:
            vocab_dict: 可选的词汇表字典（token字符串 -> ID映射）
        """
        if vocab_dict is None:
            # 使用全局默认词汇表
            self.vocab: Dict[str, int] = _VOCAB
        else:
            self.vocab = vocab_dict

        # 正向词汇表：token字符串 -> 整数ID
        # 反向词汇表：整数ID -> token字符串

        # vocab.items() 返回字典所有键值对，格式: dict_items([(key1,val1), (key2,val2)...])
        # 遍历时依次取出 (token, idx) 元组
        # 字典推导式：交换key和value位置，原始 {"C":4, "N":5} -> 反向 {4:"C", 5:"N"}
        self.inverse_vocab: Dict[int, str] = {
            idx: token for token, idx in self.vocab.items()
        }
        self.vocab_size: int = len(self.vocab)

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

        # 处理序列长度：截断或填充到max_length
        if len(tokens) > max_length:
            # 如果token数量超过max_length，截断多余部分
            # tokens[:max_length] 取前max_length个元素
            tokens = tokens[:max_length]
        else:
            # 如果token数量不足max_length，用<pad>填充到固定长度
            # 例如: len(tokens)=10, max_length=20, 差10个padding
            # [1,2,3] + [0]*2 -> [1,2,3,0,0]
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
            # 使用 dict.get()：key存在返回value，不存在返回默认值""
            token: str = self.inverse_vocab.get(token_id, "")
            # 解码时跳过特殊token（<pad>填充符、>分隔符、<bos>开始符、<eos>结束符）
            if token not in ["<pad>", ">", "<bos>", "<eos>"]:
                tokens.append(token)
        return "".join(tokens)


# MoleculeDataset class


class MoleculeDataset(Dataset):
    """
    分子数据集类,继承自PyTorch的Dataset基类。

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
        validate_smiles: bool = True,
    ) -> None:
        """
        初始化数据集。

        Args:
            data_path: 数据文件路径（支持CSV、JSON、TXT格式）
            task_type: 任务类型（"regression"回归或"classification"分类）
            max_length: 最大序列长度（超过此长度的序列会被截断）
            validate_smiles: 是否使用RDKit验证SMILES字符串的有效性
        """
        self.task_type = task_type
        self.max_length = max_length
        self.validate_smiles = validate_smiles

        # 使用全局共享词汇表
        self.vocab = _VOCAB
        self.vocab_size = _VOCAB_SIZE

        # 加载数据
        self.data = self._load_data(data_path)

        # SMILES缓存（始终为字典，启用时用于存储已处理的SMILES）
        self.smiles_cache: Dict[str, List[int]] = {}

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

        data: List[dict] = []

        # 根据文件扩展名加载数据
        # rfind(".") 找到最后一个"."的位置（防止文件名含多个点如 data.train.csv）
        # data_path[位置:] 切片获取"."及之后的内容，即扩展名
        # 如果无"."则返回空字符串""
        
        if "." in data_path:
            file_extension = data_path[data_path.rfind(".") :] 
        else: 
            file_extension = ""



        match file_extension:
            case ".csv":
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

            case ".json":
                with open(data_path, "r") as f:
                    data = json.load(f)

            case ".txt":
                with open(data_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            smiles = parts[0]
                            labels = [float(x) for x in parts[1:]]
                            data.append({"smiles": smiles, "labels": labels})
                        else:
                            data.append({"smiles": parts[0], "labels": [0.0]})

            case _:
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
        if smiles in self.smiles_cache:
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
) -> Tuple:
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

    Returns:
        (train_loader, val_loader, test_loader)元组
    """

    def make_loader(path: str) -> torch.utils.data.DataLoader:
        """为给定数据路径创建DataLoader。"""
        dataset = MoleculeDataset(
            data_path=path, task_type=task_type, max_length=max_length
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(path == train_path),  # 训练时打乱，验证/测试不打乱
            num_workers=num_workers,
            pin_memory=True,  # 加速CPU到GPU的数据传输
        )

    train_loader = make_loader(train_path)

    # 验证/测试数据集（可选，路径不存在时为None）
    val_loader = (
        make_loader(val_path) if val_path and os.path.exists(val_path) else None
    )
    test_loader = (
        make_loader(test_path) if test_path and os.path.exists(test_path) else None
    )

    return train_loader, val_loader, test_loader
