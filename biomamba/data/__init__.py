"""
数据模块 - 包含分子数据处理和SMILES分词工具

本模块导出的组件:

【分词器】
- AtomTokenizer: 原子级分词器,将SMILES字符串转换为token序列
  支持: 元素符号(Cl, Br)、环闭合(#, %)、括号原子([N], [O]等)

【数据集】
- MoleculeDataset: PyTorch数据集类,封装分子数据和标签

【数据加载函数】
- load_esol: 加载ESOL溶解度数据集
- load_bbbp: 加载BBBP血脑屏障数据集
- load_clintox: 加载ClinTox毒性数据集
- get_dataset: 统一的数据集加载接口,自动划分训练/验证/测试集
- get_task_type: 获取数据集的任务类型(回归/分类)
"""

from .tokenizer import AtomTokenizer, build_vocab_from_dataset
from .dataset import (
    MoleculeDataset,
    load_esol,
    load_bbbp,
    load_clintox,
    get_dataset,
    get_task_type,
)

__all__ = [
    'AtomTokenizer',
    'build_vocab_from_dataset',
    'MoleculeDataset',
    'load_esol',
    'load_bbbp',
    'load_clintox',
    'get_dataset',
    'get_task_type',
]
