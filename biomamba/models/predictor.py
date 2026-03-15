"""
预测头 (Prediction Head) 模块

本文件定义了模型的"预测头",也就是模型最后用于输出预测结果的部分。

整体架构回顾:
============
输入 SMILES -> 分词 -> 嵌入 -> Bi-Mamba 编码 -> 池化 -> 预测头 -> 输出

                        ┌─────────────────┐
  编码器输出              │   预测头 (MLP)   │  <- 本文件
  (batch, d_model) ---> │                 │ ---> (batch, 1)
                        └─────────────────┘

什么是预测头?
------------
预测头是模型的"最后一层",负责把编码器的输出转换成最终的预测结果。

根据任务不同:
- 回归任务: 输出一个连续值 (如溶解度)
- 分类任务: 输出每个类别的概率

常见预测头:
-----------
1. 简单线性层: y = Wx + b
2. 多层感知机 (MLP): 多个线性层 + 激活函数
3. Transformer 的 [CLS] 头
"""

import torch
import torch.nn as nn
from typing import Optional


class PredictionHead(nn.Module):
    """
    预测头 (Prediction Head)

    这是一个灵活的多层感知机 (MLP),用于:
    1. 把编码器的输出转换为最终预测
    2. 根据任务类型选择不同的输出

    结构:
    -----
    输入 (d_model) -> Linear + GELU + Dropout -> ... -> Linear -> 输出

    为什么需要多层?
    ----------------
    - 单层线性变换的表达能力有限
    - 多层可以学习更复杂的非线性关系
    - 中间的 GELU 激活函数增加非线性
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        task_type: str = 'regression',  # 'regression' or 'classification'
        dropout: float = 0.1,
        n_layers: int = 2,
    ):
        """
        初始化预测头

        参数:
        -----
        input_dim : int
            输入维度,通常是模型的隐藏维度 d_model

        hidden_dim : Optional[int]
            隐藏层维度。如果为 None,使用 input_dim。

        output_dim : int
            输出维度:
            - 回归任务: 1 (预测一个值)
            - 分类任务: 类别数 (通常是 2)

        task_type : str
            任务类型:
            - 'regression': 回归 (预测连续值)
            - 'classification': 分类 (预测类别)

        dropout : float
            Dropout 概率,防止过拟合

        n_layers : int
            隐藏层数量
        """
        super().__init__()

        # 如果没有指定隐藏维度,就使用输入维度
        if hidden_dim is None:
            hidden_dim = input_dim

        self.task_type = task_type
        self.output_dim = output_dim

        # ====== 构建 MLP ======
        # 动态创建多层神经网络
        layers = []
        in_dim = input_dim

        for i in range(n_layers):
            # 最后一层的输出维度是 output_dim,其他层是 hidden_dim
            out_dim = hidden_dim if i < n_layers - 1 else output_dim

            # 添加线性层
            layers.append(nn.Linear(in_dim, out_dim))

            # 如果不是最后一层,添加激活函数和 Dropout
            if i < n_layers - 1:
                # GELU: Gaussian Error Linear Unit
                # 是 ReLU 的平滑版本,效果通常更好
                layers.append(nn.GELU())
                # Dropout: 训练时随机丢弃一些神经元
                layers.append(nn.Dropout(dropout))

            # 更新输入维度为输出维度
            in_dim = out_dim

        # 用 nn.Sequential 把所有层打包
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
        -----
        x : torch.Tensor
            输入张量,形状 (batch_size, input_dim)

        返回:
        -----
        torch.Tensor
            输出张量,形状 (batch_size, output_dim)
        """
        return self.mlp(x)


class BiMambaForPrediction(nn.Module):
    """
    完整的 Bi-Mamba 预测模型

    这是整个模型的完整版本,包含:
    1. Bi-Mamba 编码器 (BiMambaModel)
    2. 预测头 (PredictionHead)

    数据流:
    -------
    input_ids (batch, seq_len)
          ↓
    BiMambaModel 编码器
          ↓
    pooled output (batch, d_model)
          ↓
    PredictionHead 预测头
          ↓
    predictions (batch, 1)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        use_mamba: bool = True,
        fusion: str = 'gate',
        max_len: int = 512,
        padding_idx: int = 0,
        pool_type: str = 'mean',
        # 预测头参数
        pred_hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        task_type: str = 'regression',
        pred_dropout: float = 0.1,
    ):
        """
        初始化完整模型
        """
        super().__init__()

        self.task_type = task_type

        # ====== 1. 编码器 ======
        # 导入 BiMambaModel (避免循环导入)
        from .bi_mamba import BiMambaModel
        self.encoder = BiMambaModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
            fusion=fusion,
            max_len=max_len,
            padding_idx=padding_idx,
            pool_type=pool_type,
        )

        # ====== 2. 预测头 ======
        self.predictor = PredictionHead(
            input_dim=d_model,
            hidden_dim=pred_hidden_dim,
            output_dim=output_dim,
            task_type=task_type,
            dropout=pred_dropout,
            n_layers=2,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        参数:
        -----
        input_ids : torch.Tensor
            输入的 token ID,形状 (batch_size, seq_len)

        attention_mask : Optional[torch.Tensor]
            注意力掩码 (可选)

        返回:
        -----
        torch.Tensor
            预测结果,形状 (batch_size, output_dim)
        """
        # 步骤 1: 编码
        pooled = self.encoder(input_ids, attention_mask)

        # 步骤 2: 预测
        predictions = self.predictor(pooled)

        return predictions


class BiMambaForSequenceClassification(nn.Module):
    """
    Bi-Mamba 序列分类模型

    这是一个便捷的封装类,专门用于分类任务。

    与 BiMambaForPrediction 的区别:
    - 自动设置任务类型为 'classification'
    - 支持多类别分类
    - 支持计算损失函数
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 2,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
        fusion: str = 'gate',
        pool_type: str = 'mean',
    ):
        super().__init__()

        self.num_labels = num_labels

        # 使用 BiMambaForPrediction,设置分类模式
        self.model = BiMambaForPrediction(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_mamba=use_mamba,
            fusion=fusion,
            pool_type=pool_type,
            output_dim=num_labels,
            task_type='classification',
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        前向传播,可选计算损失

        参数:
        -----
        input_ids: 输入 token ID
        attention_mask: 注意力掩码
        labels: 真实标签 (可选,用于训练时计算损失)

        返回:
        -----
        包含 loss 和 logits 的字典
        """
        # 前向计算
        logits = self.model(input_ids, attention_mask)

        # 计算损失 (如果提供了标签)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
        }


class BiMambaForRegression(nn.Module):
    """
    Bi-Mamba 回归模型

    这是一个便捷的封装类,专门用于回归任务。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
        fusion: str = 'gate',
        pool_type: str = 'mean',
    ):
        super().__init__()

        # 使用 BiMambaForPrediction,设置回归模式
        self.model = BiMambaForPrediction(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_mamba=use_mamba,
            fusion=fusion,
            pool_type=pool_type,
            output_dim=1,
            task_type='regression',
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        前向传播,可选计算损失

        参数:
        -----
        input_ids: 输入 token ID
        attention_mask: 注意力掩码
        labels: 真实值 (可选,用于训练时计算损失)

        返回:
        -----
        包含 loss 和 predictions 的字典
        """
        # 前向计算
        predictions = self.model(input_ids, attention_mask)
        predictions = predictions.squeeze(-1)

        # 计算损失 (如果提供了标签)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions,
        }


def test_prediction_head():
    """测试预测头的实现"""
    batch = 2
    input_dim = 128

    print("Testing Prediction Head...")

    # 测试回归头
    pred_head = PredictionHead(
        input_dim=input_dim,
        output_dim=1,
        task_type='regression',
    )
    x = torch.randn(batch, input_dim)
    out = pred_head(x)
    print(f"Regression head: input {x.shape} -> output {out.shape}")

    # 测试分类头
    pred_head = PredictionHead(
        input_dim=input_dim,
        output_dim=2,
        task_type='classification',
    )
    out = pred_head(x)
    print(f"Classification head: input {x.shape} -> output {out.shape}")

    # 测试完整模型
    model = BiMambaForPrediction(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        use_mamba=False,
        task_type='regression',
    )

    input_ids = torch.randint(0, 100, (batch, 32))
    out = model(input_ids)
    print(f"Full model: input {input_ids.shape} -> output {out.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_prediction_head()
