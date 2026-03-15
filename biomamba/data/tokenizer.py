"""
SMILES 分词器 (Tokenizer)

本文件实现了分子 SMILES 字符串的分词功能。

什么是分词 (Tokenization)?
=========================
分词是把文本/字符串拆分成更小的单元( token )的过程。

示例:
-----
原始文本: "CCO" (乙醇的 SMILES)
分词后: ["C", "C", "O"]

为什么需要分词?
--------------
1. 计算机只能处理数字,不能直接处理字符
2. 分词后可以把每个 token 转换成一个数字 ID
3. 这样就能用神经网络处理文本了!

SMILES 分词的特殊性:
-------------------
SMILES 字符串有一些特殊的 token:
- 双字符元素: Cl (氯), Br (溴), Fe (铁) 等
- 括号原子: [Cl], [N], [O] 等
- 环结构: 1, 2, ..., %10, %11 等
- 化学键: = (双键), # (三键), : (芳香键) 等
"""

import re
from typing import List, Optional


class AtomTokenizer:
    """
    SMILES 原子级分词器

    功能:
    -----
    1. tokenize(): 把 SMILES 字符串拆分成 token 列表
    2. encode(): 把 token 转换为数字 ID
    3. decode(): 把数字 ID 转换回 SMILES 字符串

    示例:
    -----
    >>> tokenizer = AtomTokenizer()
    >>> tokenizer.tokenize("CCO")
    ['C', 'C', 'O']
    >>> tokenizer.tokenize("ClCC")
    ['Cl', 'C', 'C']
    >>> tokenizer.tokenize("[Cl]CC")
    ['[Cl]', 'C', 'C']
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        初始化分词器

        参数:
        -----
        vocab : Optional[List[str]]
            预定义的词表。如果为 None,则使用默认词表。
        """
        if vocab is not None:
            self.vocab = vocab
        else:
            # 构建默认词表
            self.vocab = self._build_default_vocab()

        # 创建 token -> ID 的映射表
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        # 创建 ID -> token 的映射表
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        # 特殊 token
        self.pad_token = '[PAD]'      # 填充 token,用于补齐序列
        self.unk_token = '[UNK]'      # 未知 token,用于处理未见过的字符
        self.cls_token = '[CLS]'      # 分类 token,用于表示整个序列
        self.sep_token = '[SEP]'      # 分隔 token,用于分隔不同部分
        self.mask_token = '[MASK]'    # 掩码 token,用于掩码语言模型

        # 特殊 token 对应的 ID
        self.pad_id = self.token_to_id.get(self.pad_token, 0)
        self.unk_id = self.token_to_id.get(self.unk_token, 1)
        self.cls_id = self.token_to_id.get(self.cls_token, 2)
        self.sep_id = self.token_to_id.get(self.sep_token, 3)
        self.mask_id = self.token_to_id.get(self.mask_token, 4)

    def _build_default_vocab(self) -> List[str]:
        """
        构建默认 SMILES 词表

        词表包含:
        ---------
        1. 特殊 token: [PAD], [UNK], [CLS], [SEP], [MASK]
        2. 单字符原子: C, N, O, S, P, B, F, I, H
        3. 双字符元素: Cl, Br, Li, Na, K, Ca, Mg, Zn, Fe, Cu, ...
        4. 括号原子: [Cl], [Br], [C@@H], [C@H], [CH], [NH], ...
        5. 特殊字符: =, #, +, -, (, ), [, ], %, /, \, @, :, .
        6. 环数字: 0-9
        """
        # 1. 特殊 token (必须放在最前面!)
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

        # 2. 单字符原子
        single_atoms = ['C', 'N', 'O', 'S', 'P', 'B', 'F', 'I', 'H']

        # 3. 双字符元素 (过渡金属和卤素)
        double_atoms = [
            'Cl', 'Br', 'Li', 'Na', 'K', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu',
            'Al', 'Si', 'Ge', 'Sn', 'Pb', 'Hg', 'Ag', 'Au', 'Pt', 'Ni'
        ]

        # 4. 括号原子 (有机子集)
        bracket_atoms = [
            '[Cl]', '[Br]', '[C@@H]', '[C@H]', '[CH]', '[NH]',
            '[O]', '[N]', '[S]', '[P]', '[B]', '[Si]', '[Se]'
        ]

        # 5. 特殊化学字符
        special_chars = [
            '=', '#', '+', '-', '(', ')', '[', ']', '%',
            '/', '\\', '@', '+', '-', ':', '.'
        ]

        # 6. 环闭合数字
        ring_digits = [str(i) for i in range(10)]

        # 合并所有词表
        vocab = (special_tokens + single_atoms + double_atoms +
                 bracket_atoms + special_chars + ring_digits)

        return vocab

    def tokenize(self, smiles: str) -> List[str]:
        """
        把 SMILES 字符串分词

        这是分词的核心逻辑:
        -------------------
        1. 优先匹配双字符元素 (如 Cl, Br)
        2. 匹配括号原子 (如 [Cl], [N])
        3. 匹配双位数字环 (如 %10, %11)
        4. 最后匹配单字符

        参数:
        -----
        smiles : str
            SMILES 字符串

        返回:
        -----
        List[str]: token 列表
        """
        tokens = []
        i = 0
        smiles = smiles.strip()  # 去除首尾空格

        while i < len(smiles):
            # ====== 情况 1: 双字符元素 ======
            # 例如 "Cl", "Br", "Li", "Na" 等
            if i + 1 < len(smiles):
                two_char = smiles[i:i+2]
                # 检查是否是有效的双字符元素
                # 注意: [C, [O 等开头的是括号原子,不是双字符元素
                if (two_char in self.vocab and
                    two_char not in ['[C', '[O', '[N', '[S', '[P', '[B', '[S']):
                    tokens.append(two_char)
                    i += 2
                    continue

            # ====== 情况 2: 括号原子 ======
            # 例如 "[Cl]", "[N]", "[O-]", "[C@@H]" 等
            if smiles[i] == '[':
                # 找到匹配的 closing bracket
                j = i + 1
                while j < len(smiles) and smiles[j] != ']':
                    j += 1
                if j < len(smiles):
                    token = smiles[i:j+1]  # 包含 []
                    tokens.append(token)
                    i = j + 1
                    continue

            # ====== 情况 3: 双位数字环 ======
            # 例如 "%10", "%11" 表示两位数的环编号
            if smiles[i] == '%':
                if i + 2 < len(smiles):
                    tokens.append(smiles[i:i+3])
                    i += 3
                    continue

            # ====== 情况 4: 单字符 ======
            token = smiles[i]
            # 检查是否是有效的 SMILES 字符
            if (token in self.vocab or token.isalnum() or
                token in '()[]=#+-@%/\\.:'):
                tokens.append(token)
            else:
                # 未知字符
                tokens.append(token)
            i += 1

        return tokens

    def encode(
        self,
        smiles: str,
        max_length: int = 128,
        padding: bool = True,
        truncation: bool = True
    ) -> List[int]:
        """
        把 SMILES 字符串转换为数字 ID 序列

        流程:
        -----
        1. 分词 (tokenize)
        2. 转换为 ID (查表)
        3. 截断 (如果太长)
        4. 添加特殊 token ([CLS], [SEP])
        5. 填充 ([PAD])

        参数:
        -----
        smiles : str
            SMILES 字符串

        max_length : int
            最大序列长度

        padding : bool
            是否填充到 max_length

        truncation : bool
            是否截断超过长度的序列

        返回:
        -----
        List[int]: ID 列表
        """
        # 步骤 1: 分词
        tokens = self.tokenize(smiles)

        # 步骤 2: 转换为 ID
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # 尝试小写
                if token.lower() in self.token_to_id:
                    ids.append(self.token_to_id[token.lower()])
                else:
                    # 未知 token
                    ids.append(self.unk_id)

        # 步骤 3: 截断
        # 预留位置给 [CLS] 和 [SEP]
        if truncation and len(ids) > max_length - 2:
            ids = ids[:max_length - 2]

        # 步骤 4: 添加特殊 token
        # [CLS] 在开头, [SEP] 在结尾
        # [CLS] 的作用: 类似于 "句子开头",用于分类任务
        # [SEP] 的作用: 类似于 "句子结束"
        ids = [self.cls_id] + ids + [self.sep_id]

        # 步骤 5: 填充
        if padding:
            if len(ids) < max_length:
                # 补 [PAD] 到指定长度
                ids = ids + [self.pad_id] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        把数字 ID 序列转换回 SMILES 字符串

        参数:
        -----
        ids : List[int]
            ID 列表

        返回:
        -----
        str: SMILES 字符串
        """
        tokens = []
        for id_ in ids:
            # 跳过 padding
            if id_ == self.pad_id:
                continue
            # 查表转换
            if id_ in self.id_to_token:
                tokens.append(self.id_to_token[id_])
            else:
                tokens.append(self.unk_token)

        # 移除特殊 token
        tokens = [
            t for t in tokens
            if t not in [self.cls_token, self.sep_token,
                         self.pad_token, self.mask_token]
        ]

        # 拼接成字符串
        return ''.join(tokens)

    def __len__(self) -> int:
        """返回词表大小"""
        return len(self.vocab)


def build_vocab_from_dataset(smiles_list: List[str]) -> List[str]:
    """
    从 SMILES 列表构建词表

    这个函数会:
    1. 使用默认分词器对所有 SMILES 分词
    2. 收集所有出现过的 token
    3. 排序后返回词表

    参数:
    -----
    smiles_list : List[str]
        SMILES 字符串列表

    返回:
    -----
    List[str]: 排序后的词表
    """
    tokenizer = AtomTokenizer()
    vocab_set = set(tokenizer.vocab)

    # 遍历所有 SMILES,收集 token
    for smiles in smiles_list:
        tokens = tokenizer.tokenize(smiles)
        vocab_set.update(tokens)

    # 排序,确保结果可复现
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    vocab_list = special_tokens + sorted(list(vocab_set - set(special_tokens)))

    return vocab_list


if __name__ == "__main__":
    """测试分词器"""
    tokenizer = AtomTokenizer()

    # 测试用例
    test_smiles = [
        'CCO',           # 乙醇
        'c1ccccc1',      # 苯
        'CC(=O)O',       # 乙酸
        'CC(C)C',        # 异丁烷
        '[Cl]CC(C)C',    # 2-氯-2-甲基丙烷
    ]

    print("=" * 60)
    print("SMILES Tokenizer 测试")
    print("=" * 60)

    for smiles in test_smiles:
        tokens = tokenizer.tokenize(smiles)       # 分词
        ids = tokenizer.encode(smiles)           # 编码
        decoded = tokenizer.decode(ids)           # 解码

        print(f"\n原始 SMILES: {smiles}")
        print(f"分词结果:    {tokens}")
        print(f"ID 序列:     {ids}")
        print(f"解码结果:    {decoded}")
        print("-" * 60)

    print(f"\n词表大小: {len(tokenizer)}")
