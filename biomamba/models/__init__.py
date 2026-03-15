"""
模型模块 - 包含Bi-Mamba-Chem的所有神经网络组件

本模块导出的组件:

【核心SSM组件】
- SSMCore: 状态空间模型的核心实现,包含选择性和并行扫描算法
- SSMBlock: SSM块,包含投影、卷积和SSM操作
- BidirectionalSSM: 双向SSM,同时处理前向和后向序列

【Mamba组件】
- MambaBlock: Mamba块,SSM + 门控前馈网络的组合
- MambaLayer: 多层Mamba块的堆叠

【双向Mamba组件】
- BiMambaBlock: 双向Mamba块,包含前向和后向两个分支
- BiMambaEncoder: 双向Mamba编码器,多层BiMambaBlock的堆叠
- BiMambaModel: 完整的Bi-Mamba模型,包含embedding、encoder和pooling

【预测头组件】
- PredictionHead: 预测头,将特征映射到任务输出
- BiMambaForPrediction: 通用预测模型,支持回归和分类
- BiMambaForSequenceClassification: 序列分类模型
- BiMambaForRegression: 回归模型
"""

from .ssm_core import SSMCore, SSMBlock, BidirectionalSSM
from .mamba_block import MambaBlock, MambaLayer
from .bi_mamba import BiMambaBlock, BiMambaEncoder, BiMambaModel
from .predictor import (
    PredictionHead,
    BiMambaForPrediction,
    BiMambaForSequenceClassification,
    BiMambaForRegression,
)

__all__ = [
    'SSMCore',
    'SSMBlock',
    'BidirectionalSSM',
    'MambaBlock',
    'MambaLayer',
    'BiMambaBlock',
    'BiMambaEncoder',
    'BiMambaModel',
    'PredictionHead',
    'BiMambaForPrediction',
    'BiMambaForSequenceClassification',
    'BiMambaForRegression',
]
