"""
分子数据集加载模块

本文件负责加载和处理分子数据集 (MoleculeNet)。

支持的数据库:
-----------
1. ESOL: 溶解度预测 (回归任务)
2. BBBP: 血脑屏障穿透性 (二分类)
3. ClinTox: 临床毒性 (二分类)

核心概念:
---------
1. SMILES: 分子的字符串表示
   - 例如: "CCO" 表示乙醇

2. Dataset: PyTorch 的数据容器
   - 提供 __len__ 和 __getitem__ 方法
   - 可以被 DataLoader 使用

3. Train/Val/Test Split: 训练/验证/测试集划分
   - 训练集: 用于训练模型
   - 验证集: 用于调参和早停
   - 测试集: 用于最终评估

数据划分比例通常是:
- 训练集: 80%
- 验证集: 10%
- 测试集: 10%
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from rdkit import Chem  # RDKit: 化学信息学库
from rdkit.Chem import Descriptors

from .tokenizer import AtomTokenizer


class MoleculeDataset(Dataset):
    """
    分子数据集类

    继承自 PyTorch 的 Dataset,用于存储分子数据。

    功能:
    -----
    1. 存储 SMILES 字符串和标签
    2. 使用 RDKit 验证分子有效性
    3. 提供分词和编码功能

    为什么要验证分子?
    -----------------
    不是所有字符串都是有效的 SMILES!
    例如 "XXX" 不是有效的分子表示。
    我们用 RDKit 检查,无效的分子会被过滤掉。
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: List[float],
        tokenizer: AtomTokenizer,
        max_length: int = 128
    ):
        """
        初始化数据集

        参数:
        -----
        smiles_list : List[str]
            SMILES 字符串列表

        labels : List[float]
            标签列表 (回归: 实数, 分类: 0 或 1)

        tokenizer : AtomTokenizer
            分词器,用于把 SMILES 转换为数字

        max_length : int
            最大序列长度
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ====== 验证 SMILES 有效性 ======
        # 使用 RDKit 检查每个 SMILES 是否能解析为有效分子
        self.valid_indices = []  # 有效分子的索引
        self.valid_smiles = []   # 有效的 SMILES
        self.valid_labels = []  # 有效的标签

        for i, smiles in enumerate(smiles_list):
            # Chem.MolFromSmiles() 尝试解析 SMILES
            # 如果成功,返回分子对象
            # 如果失败,返回 None
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                self.valid_indices.append(i)
                self.valid_smiles.append(smiles)
                self.valid_labels.append(labels[i])

        print(f"有效分子: {len(self.valid_smiles)}/{len(smiles_list)}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.valid_smiles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个数据样本

        参数:
        -----
        idx : int
            索引

        返回:
        -----
        Tuple[torch.Tensor, torch.Tensor]:
            (input_ids, label) - (输入ID, 标签)
        """
        # 获取 SMILES 和标签
        smiles = self.valid_smiles[idx]
        label = self.valid_labels[idx]

        # ====== 分词和编码 ======
        # SMILES 字符串 -> token IDs
        input_ids = self.tokenizer.encode(
            smiles,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )

        # 转换为 PyTorch Tensor
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )


def load_esol() -> Tuple[List[str], List[float]]:
    """
    加载 ESOL 数据集 (溶解度预测)

    ESOL 是 MoleculeNet 的一个经典数据集:
    - 任务: 预测化合物在水中的溶解度
    - 标签: logS (溶解度的对数)
    - 数值范围: 通常在 -6 到 1 之间

    负值含义:
    --------
    logS = -2 表示溶解度为 10^(-2) = 0.01 mol/L
    负值越大,溶解度越低!

    返回:
    -----
    Tuple[List[str], List[float]]: (SMILES列表, 溶解度标签)
    """
    # ESOL 数据集 (约 1128 个分子)
    # 这里提供一个简化版本用于演示
    data = [
        ('CCO', -0.77),  # 乙醇
        ('CC(C)C', -2.18),  # 2-甲基丙烷
        ('CC(=O)O', -0.28),  # 乙酸
        ('c1ccccc1', -1.88),  # 苯
        ('CC(C)CC(C)C', -3.21),  # 2,4-二甲基戊烷
        ('CCCCC', -2.67),  # 戊烷
        ('CCCC', -1.85),  # 丁烷
        ('CCC', -1.29),  # 丙烷
        ('CC', -0.39),  # 乙烷
        ('C', -0.12),  # 甲烷
        ('CCC(C)C', -2.92),  # 2-甲基丁烷
        ('c1ccc(C)cc1', -2.59),  # 甲苯
        ('c1ccc(cc1)C(C)(C)C', -3.42),  # 叔丁基苯
        ('c1ccc2ccccc2c1', -2.82),  # 萘
        ('c1ccc2c(c1)ccc3c2cccc3', -4.03),  # 蒽
        ('c1ccc2c(c1)ccc3c2ccc4c3cccc4', -4.58),  # 并四苯
        ('c1ccncc1', -1.06),  # 吡啶
        ('c1cnccn1', -1.23),  # 嘧啶
        ('C1CCCCC1', -1.83),  # 环己烷
        ('C1CCCC1', -1.68),  # 环戊烷
        ('C1CCC1', -1.15),  # 环丁烷
        ('c1ccoc1', -1.27),  # 呋喃
        ('c1cc[nH]c1', -1.38),  # 吡咯
        ('c1cscc1', -1.41),  # 噻酚
        ('CC=O', -0.58),  # 乙醛
        ('CC(C)=O', -0.32),  # 丙酮
        ('CCC(=O)C', -0.69),  # 丁酮
        ('CCC(=O)CCC', -1.22),  # 2-戊酮
        ('CC(=O)CC(C)C', -1.03),  # 甲基异丁基酮
        ('CCOC(=O)C', -0.22),  # 乙酸乙酯
        ('CCCOC(=O)C', -0.68),  # 乙酸丙酯
        ('CCOC(=O)CC', -0.63),  # 丙酸乙酯
        ('c1ccc(N)cc1', -1.52),  # 苯胺
        ('c1ccc(Nc2ccccc2)cc1', -2.47),  # 二苯胺
        ('c1ccc(O)cc1', -1.50),  # 苯酚
        ('c1ccc(Oc2ccccc2)cc1', -2.88),  # 二苯醚
        ('CC(O)C', -0.90),  # 2-丙醇
        ('CCCO', -0.76),  # 1-丙醇
        ('CCCCO', -1.24),  # 1-丁醇
        ('CC(C)O', -0.89),  # 2-丙醇
        ('CC(C)(C)O', -0.93),  # 叔丁醇
        ('c1ccc2c(c1)C(=O)O2', -1.82),  # 邻苯二甲酸酐
        ('CC(=O)OC(=O)C', -0.44),  # 乙酸酐
        ('C#N', -0.25),  # 氰化氢
        ('CC#N', -0.27),  # 乙腈
        ('CCC#N', -0.36),  # 丙腈
        ('c1cccnc1', -1.04),  # 吡啶
        ('c1cncnc1', -1.20),  # 吡嗪
        ('c1cnc(N)nc1', -1.55),  # 嘧啶胺
    ]

    smiles_list = [d[0] for d in data]
    labels = [d[1] for d in data]

    return smiles_list, labels


def load_bbbp() -> Tuple[List[str], List[int]]:
    """
    加载 BBBP 数据集 (血脑屏障穿透性)

    BBBP (Blood-Brain Barrier Penetration):
    - 任务: 预测分子能否穿透血脑屏障
    - 标签: 1 = 能穿透, 0 = 不能穿透

    血脑屏障 (BBB):
    -------------
    大脑血管内壁的一层细胞,只允许某些分子通过。
    药物设计时,需要考虑是否能穿透 BBB 到达大脑。

    返回:
    -----
    Tuple[List[str], List[int]]: (SMILES列表, 穿透性标签)
    """
    # BBBP 数据集样本
    data = [
        ('O=C(C)Oc1ccccc1C(=O)O', 1),  # 阿司匹林
        ('CC(=O)Oc1ccccc1C(=O)O', 1),  # 阿司匹林
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 1),  # 咖啡因
        ('CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13', 1),  # 地西泮
        ('Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)C', 1),  # 西替利嗪
        ('CC(C)(C)Oc1ccc(cc1)C(C)C(=O)O', 1),  # 布洛芬
        ('CC(C)CC(C)C', 0),  # 异丁烷 (不穿透)
        ('C=C', 0),  # 乙烯 (不穿透)
        ('CC(C)=O', 1),  # 丙酮
        ('c1ccccc1', 1),  # 苯
        ('c1ccc(cc1)C', 1),  # 甲苯
        ('c1ccc(N)cc1', 1),  # 苯胺
        ('c1ccc(O)cc1', 1),  # 苯酚
        ('c1ccc(C)cc1', 1),  # 二甲苯
        ('c1ccc(Cl)cc1', 1),  # 氯苯
        ('c1ccc(F)cc1', 1),  # 氟苯
        ('c1ccc(Br)cc1', 1),  # 溴苯
        ('CC(C)Cc1ccc(C(C)C)cc1', 1),  # 布洛芬类
        ('CC(C)CCOC(=O)C(C)CO', 1),  # 萘普生类
        ('O=C1c2ccccc2CCc3ccccc13', 1),  # 芴酮
        ('c1ccc2c(c1)Cc3ccccc3C2', 1),  # 芴
        ('c1ccncc1', 1),  # 吡啶
        ('c1cnc[nH]1', 1),  # 咪唑
        ('N#Cc1ccc(N)cc1', 1),  # 4-氨基苯甲腈
        ('Nc1ccc(N)cc1', 1),  # 对苯二胺
        ('O=C(N)N', 1),  # 尿素
        ('CN(C)C(=O)N', 1),  # 二甲基脲
        ('CC(C)N', 1),  # 异丙胺
        ('CCN', 1),  # 乙胺
        ('CN', 1),  # 甲胺
        ('C1CCNC1', 1),  # 吡咯烷
        ('C1CCNCC1', 1),  # 哌嗪
        ('C1COCCC1', 1),  # 四氢呋喃
        ('C1CCOCC1', 1),  # 四氢吡喃
        ('ClCCCl', 0),  # 1,2-二氯乙烷 (不穿透)
        ('ClCCl', 0),  # 二氯甲烷 (不穿透)
        ('CC(=O)O', 1),  # 乙酸
        ('CC(C)O', 1),  # 异丙醇
        ('OCCO', 1),  # 乙二醇
        ('OCCOCCO', 1),  # 三乙二醇
        ('COCCO', 1),  # 2-甲氧基乙醇
        ('CCOC(=O)CO', 1),  # 乳酸乙酯
        ('c1ccc2[nH]ccc2c1', 1),  # 吲哚
        ('c1ccc2[nH]cc2c1', 1),  # 吲哚
        ('c1ccc2c(c1)[nH]cc2', 1),  # 吲哚
        ('c1ccc2c(c1)C(=O)N2', 1),  # 羟吲哚
        ('c1ccc2c(c1)Cc3ccccc3C2', 1),  # 四氢萘
        ('c1ccc2c(c1)C=CC2', 1),  # 茚
    ]

    smiles_list = [d[0] for d in data]
    labels = [d[1] for d in data]

    return smiles_list, labels


def load_clintox() -> Tuple[List[str], List[int]]:
    """
    加载 ClinTox 数据集 (临床毒性)

    ClinTox:
    - 任务: 预测化合物的临床毒性
    - 标签: 1 = 有毒, 0 = 无毒

    数据来源:
    ---------
    FDA 批准药物 vs 临床失败药物的毒性数据。

    返回:
    -----
    Tuple[List[str], List[int]]: (SMILES列表, 毒性标签)
    """
    # ClinTox 数据集样本
    data = [
        # FDA 批准药物 (标签 0 - 无毒)
        ('CC(=O)Oc1ccccc1C(=O)O', 0),  # 阿司匹林
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 0),  # 咖啡因
        ('CC(C)(C)Oc1ccc(cc1)C(C)C(=O)O', 0),  # 布洛芬
        ('CC(C)N', 0),  # 异丙胺
        ('CN', 0),  # 甲胺
        ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 0),  # 布洛芬
        ('CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13', 0),  # 地西泮
        ('Clc1ccc(cc1)C(c2ccccc2)N3CCN(CC3)C', 0),  # 西替利嗪
        ('CC(C)Cc1ccc(C(C)C)cc1', 0),  # 联苯
        ('c1ccccc1', 0),  # 苯
        ('c1ccc(N)cc1', 0),  # 苯胺
        ('c1ccc(O)cc1', 0),  # 苯酚
        ('CC(=O)N', 0),  # 乙酰胺
        ('CCOC(=O)C', 0),  # 乙酸乙酯
        ('CCO', 0),  # 乙醇
        ('CCC', 0),  # 丙烷
        ('CCCC', 0),  # 丁烷
        ('CCCCC', 0),  # 戊烷
        ('c1ccc(cc1)C', 0),  # 甲苯
        ('c1ccc(C)cc1', 0),  # 二甲苯
        ('CC(C)O', 0),  # 异丙醇

        # 有毒化合物 (标签 1)
        ('CC(C)(C)C(=O)O', 1),  # 特戊酸
        ('ClCCCl', 1),  # 1,2-二氯乙烷
        ('ClCCl', 1),  # 二氯甲烷
        ('ClCCBr', 1),  # 溴氯乙烷
        ('CC(=O)Cl', 1),  # 乙酰氯
        ('C(=O)Cl', 1),  # 光气
        ('CC(=O)OC(=O)C', 1),  # 乙酸酐
        ('O=C=O', 1),  # 二氧化碳 (高浓度有毒)
        ('C#N', 1),  # 氰化氢
        ('ClC#N', 1),  # 氯氰
        ('CC(=O)OCCOC(=O)C', 1),  # 三乙酸甘油酯
        ('c1ccc2c(c1)Cl', 1),  # 氯萘
        ('Clc1ccc(cc1)C(=O)O', 1),  # 4-氯苯甲酸
        ('Clc1ccccc1Cl', 1),  # 二氯苯
        ('Clc1ccc(Cl)cc1', 1),  # 1,4-二氯苯
        ('Clc1ccc(Cl)c(Cl)c1', 1),  # 1,2,4-三氯苯
        ('CC(C)C', 1),  # 异丁烷 (高浓度有毒)
        ('c1ccncc1', 1),  # 吡啶 (有毒)
        ('c1cnc[nH]1', 1),  # 咪唑 (有毒)
        ('Clc1ccc(N)cc1', 1),  # 4-氯苯胺 (有毒)
    ]

    smiles_list = [d[0] for d in data]
    labels = [d[1] for d in data]

    return smiles_list, labels


def get_dataset(
    name: str,
    data_dir: str = './data',
    max_length: int = 128,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset, AtomTokenizer]:
    """
    获取数据集

    这个函数是主要入口,一次性返回:
    - 训练集
    - 验证集
    - 测试集
    - 分词器

    参数:
    -----
    name : str
        数据集名称: 'ESOL', 'BBBP', 'CLINTOX'

    data_dir : str
        数据目录 (暂未使用)

    max_length : int
        最大序列长度

    split_ratio : Tuple[float, float, float]
        划分比例 (训练, 验证, 测试)

    seed : int
        随机种子,保证划分可复现

    返回:
    -----
    Tuple[...]: (训练集, 验证集, 测试集, 分词器)
    """
    np.random.seed(seed)

    # ====== 1. 加载对应数据集 ======
    if name.upper() == 'ESOL':
        smiles_list, labels = load_esol()
    elif name.upper() == 'BBBP':
        smiles_list, labels = load_bbbp()
    elif name.upper() == 'CLINTOX':
        smiles_list, labels = load_clintox()
    else:
        raise ValueError(f"未知数据集: {name}")

    # ====== 2. 创建分词器 ======
    tokenizer = AtomTokenizer()

    # ====== 3. 创建完整数据集 ======
    dataset = MoleculeDataset(
        smiles_list=smiles_list,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # ====== 4. 随机划分 ======
    indices = np.random.permutation(len(dataset))

    # 计算划分点
    n = len(dataset)
    n_train = int(n * split_ratio[0])      # 训练集大小
    n_val = int(n * split_ratio[1])        # 验证集大小

    # 划分索引
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # ====== 5. 创建子数据集 ======
    train_smiles = [dataset.valid_smiles[i] for i in train_indices]
    train_labels = [dataset.valid_labels[i] for i in train_indices]
    val_smiles = [dataset.valid_smiles[i] for i in val_indices]
    val_labels = [dataset.valid_labels[i] for i in val_indices]
    test_smiles = [dataset.valid_smiles[i] for i in test_indices]
    test_labels = [dataset.valid_labels[i] for i in test_indices]

    train_dataset = MoleculeDataset(train_smiles, train_labels, tokenizer, max_length)
    val_dataset = MoleculeDataset(val_smiles, val_labels, tokenizer, max_length)
    test_dataset = MoleculeDataset(test_smiles, test_labels, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset, tokenizer


def get_task_type(dataset_name: str) -> str:
    """
    获取数据集对应的任务类型

    参数:
    -----
    dataset_name : str
        数据集名称

    返回:
    -----
    str: 'regression' 或 'classification'
    """
    if dataset_name.upper() == 'ESOL':
        return 'regression'
    elif dataset_name.upper() in ['BBBP', 'CLINTOX']:
        return 'classification'
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


if __name__ == "__main__":
    """测试数据加载"""
    # 测试 ESOL
    print("测试 ESOL 数据集...")
    train, val, test, tokenizer = get_dataset('ESOL')
    print(f"训练集大小: {len(train)}, 验证集: {len(val)}, 测试集: {len(test)}")
    print(f"词表大小: {len(tokenizer)}")

    sample = train[0]
    print(f"样本输入形状: {sample[0].shape}")
    print(f"样本标签: {sample[1]}")

    # 测试 BBBP
    print("\n测试 BBBP 数据集...")
    train, val, test, tokenizer = get_dataset('BBBP')
    print(f"训练集大小: {len(train)}, 验证集: {len(val)}, 测试集: {len(test)}")

    # 测试 ClinTox
    print("\n测试 ClinTox 数据集...")
    train, val, test, tokenizer = get_dataset('CLINTOX')
    print(f"训练集大小: {len(train)}, 验证集: {len(val)}, 测试集: {len(test)}")
