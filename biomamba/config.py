"""
配置模块 - 存储模型和训练的超参数

本文件包含项目的所有默认配置参数:
- MODEL_CONFIG: 模型结构相关的参数 (层数、维度等)
- TRAIN_CONFIG: 训练过程相关的参数 (学习率、批次大小等)
- DATA_CONFIG: 数据处理相关的参数 (序列长度、数据集划分比例)
- DATASETS: 支持的数据集信息

这些参数可以在命令行中覆盖,详见train.py和eval.py的使用说明。

设计思路:
- 使用字典存储配置,结构清晰,易于修改
- 将训练、模型、数据配置分离,便于管理
- 提供get_xxx_config()函数获取配置副本,避免修改原始配置
"""

# 模型配置 - 控制模型的结构和规模
# 这些参数决定了模型的复杂度和参数量
MODEL_CONFIG = {
    'd_model': 256,           # 模型隐藏层维度: 每个token的向量表示维度,越大表示模型越"聪明"但也越慢
    'n_layers': 4,            # 模型层数: Mamba块的堆叠层数,越多能学习越复杂的模式,但也更容易过拟合
    'd_state': 128,          # SSM状态维度: 状态空间模型的"内存"大小,类似于RNN的hidden state
    'd_conv': 4,             # 卷积核大小: 用于局部特征提取的卷积窗口大小
    'expand': 2,              # 扩展因子: 前馈网络中间层维度 = d_model * expand
    'dropout': 0.1,          # Dropout比例: 随机丢弃神经元的比例,用于防止过拟合
    'norm_eps': 1e-5,        # LayerNorm epsilon: 防止除零的小常数
    'max_len': 512,          # 最大序列长度: 支持的最长SMILES字符串长度
    'fusion': 'gate',        # 双向融合方式: 'concat'(拼接), 'add'(相加), 'gate'(门控,默认推荐)
    'pool_type': 'mean',     # 池化类型: 'mean'(平均池化), 'max'(最大池化), 'cls'(CLS标记)
}

# 训练配置 - 控制训练过程的各种超参数
# 这些参数影响训练的速度、稳定性和最终效果
TRAIN_CONFIG = {
    'epochs': 100,           # 训练轮数: 完整遍历训练集的次数,越多模型拟合越好但可能过拟合
    'batch_size': 32,       # 批次大小: 每次参数更新使用的样本数,越大训练越快但越消耗显存
    'lr': 1e-4,             # 学习率: 每次参数更新的步长,越大训练越快但可能不稳定
    'weight_decay': 1e-5,   # 权重衰减: L2正则化系数,用于防止过拟合
    'warmup_steps': 100,    # 预热步数: 学习率从0逐渐增加到设定值的步数,有助于训练稳定
    'grad_clip': 1.0,       # 梯度裁剪: 限制梯度的最大范数,防止梯度爆炸
}

# 数据配置 - 控制数据处理和划分
DATA_CONFIG = {
    'max_length': 128,      # 最大序列长度: 超过这个长度的SMILES会被截断
    'train_ratio': 0.8,     # 训练集比例: 80%的数据用于训练
    'val_ratio': 0.1,      # 验证集比例: 10%的数据用于调参和早停
    'test_ratio': 0.1,     # 测试集比例: 10%的数据用于最终评估
}

# 数据集信息 - 支持的数据集及其属性
# 每个数据集包含: 任务类型、描述、评估指标
DATASETS = {
    'ESOL': {
        'task_type': 'regression',      # 回归任务: 预测连续值
        'description': '水溶解度预测',   # 预测化合物在水中的溶解度
        'metric': 'RMSE',                # 使用RMSE作为评估指标
    },
    'BBBP': {
        'task_type': 'classification',  # 分类任务: 预测离散类别
        'description': '血脑屏障穿透性', # 预测化合物能否穿透血脑屏障
        'metric': 'ROC-AUC',              # 使用ROC-AUC作为评估指标
    },
    'CLINTOX': {
        'task_type': 'classification',   # 分类任务: 预测离散类别
        'description': '临床毒性预测',  # 预测化合物的临床毒性
        'metric': 'ROC-AUC',              # 使用ROC-AUC作为评估指标
    },
}


def get_model_config(model_name: str = 'bi_mamba'):
    """
    获取模型配置

    返回模型配置的副本,避免修改原始配置。

    Args:
        model_name: 模型名称,目前支持'bi_mamba'

    Returns:
        模型配置字典的副本
    """
    return MODEL_CONFIG.copy()


def get_train_config():
    """
    获取训练配置

    返回训练配置的副本,避免修改原始配置。

    Returns:
        训练配置字典的副本
    """
    return TRAIN_CONFIG.copy()


def get_dataset_info(dataset_name: str):
    """
    获取数据集信息

    根据数据集名称返回对应的任务类型、描述和评估指标。

    Args:
        dataset_name: 数据集名称,支持'ESOL', 'BBBP', 'CLINTOX' (大小写不敏感)

    Returns:
        包含任务类型、描述、评估指标的字典

    Raises:
        ValueError: 如果数据集名称不支持
    """
    if dataset_name.upper() not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASETS[dataset_name.upper()]


if __name__ == "__main__":
    # 测试代码: 直接运行config.py打印所有配置
    print("模型配置 (Model Configuration):")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n训练配置 (Training Configuration):")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n数据集信息 (Dataset Information):")
    for name, info in DATASETS.items():
        print(f"  {name}: {info}")
