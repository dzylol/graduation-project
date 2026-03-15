"""
Bi-Mamba Encoder: Bidirectional Mamba for molecular property prediction.

本文件实现了 Bi-Mamba (双向Mamba) 编码器,用于分子属性预测。

核心思想:
---------
1. 单向 Mamba 只能从左到右(或从右到左)处理分子序列
2. 分子(特别是SMILES字符串)的化学键是双向的,需要从两个方向捕捉信息
3. 双向处理可以让模型同时学习原子的左右两侧化学环境

比喻:
-----
想象你读一段文字:
- 从左往右读: 你知道这个词之前是什么
- 从右往左读: 你知道这个词之后是什么
- 双向都读: 你知道这个词的完整上下文

Bi-Mamba 就是这个道理,让模型从两个方向理解分子结构!
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# 导入 Mamba 块
from .mamba_block import MambaBlock, MambaLayer


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block (双向 Mamba 块)

    这个块同时运行:
    1. Forward 分支: 从左到右处理序列
    2. Backward 分支: 从右到左处理序列
    3. 融合层: 把两个方向的结果合并

    为什么这样做?
    - 分子 SMILES 字符串中,化学键是双向的
    - 例如 "CC=O" 中,C 原子同时连接着另一个 C 和 O
    - 双向处理能让模型同时看到每个原子的左右邻居
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        use_mamba: bool = True,
        fusion: str = 'gate',  # 'concat', 'add', 'gate'
    ):
        """
        初始化双向 Mamba 块

        参数说明 (这些是深度学习模型常见的超参数):
        -----------
        d_model : int
            模型的隐藏维度,类似于"每个token的向量表示有多长"
            越大: 模型越强大,但越慢,越容易过拟合
            常见值: 128, 256, 512, 768

        d_state : int
            SSM 状态维度,可以理解为"模型记住多少过去的信息"
            越大: 记忆越长,但计算越慢
            常见值: 16, 32, 64, 128

        d_conv : int
            卷积核大小,用于捕捉局部上下文
            类似于 N-gram 中的 N
            常见值: 3, 4, 5

        expand : int
            扩展因子,内部隐藏层是 d_model 的多少倍
            常见值: 2

        dropout : float
            dropout 概率,防止过拟合
            0.1 表示随机 10% 的神经元不工作

        fusion : str
            融合策略,如何合并两个方向的结果
            - 'concat': 直接拼接,信息量最大
            - 'add': 相加,简单但可能丢失信息
            - 'gate': 门控融合(推荐),让模型学习哪些信息重要
        """
        super().__init__()

        self.d_model = d_model
        self.fusion = fusion

        # ====== 分支 1: Forward Mamba (前向) ======
        # 从左到右处理序列
        self.forward_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
        )

        # ====== 分支 2: Backward Mamba (后向) ======
        # 从右到左处理序列
        self.backward_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
        )

        # ====== 融合层 ======
        # 根据 fusion 参数选择不同的融合方式
        if fusion == 'concat':
            # 拼接后投影: [forward, backward] -> d_model
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),  # GELU 是 ReLU 的平滑版本
                nn.Linear(d_model, d_model),
            )
        elif fusion == 'gate':
            # 门控融合: 让模型学习哪些方向的信息重要
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            self.value_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
        -----
        x : torch.Tensor
            输入张量,形状为 (batch_size, sequence_length, d_model)
            例如: (32, 100, 256) 表示 32 个分子,每个分子 100 个 token,每个 token 256 维

        返回:
        -----
        torch.Tensor
            输出张量,形状与输入相同 (batch_size, sequence_length, d_model)
        """
        batch, seq_len, _ = x.shape

        # ====== 步骤 1: 前向处理 ======
        # 正常顺序: C-C-O-N -> 从左读到右
        forward_out = self.forward_block(x)

        # ====== 步骤 2: 后向处理 ======
        # 翻转顺序: C-C-O-N -> N-O-C-C (从右读到左)
        # torch.flip(x, dims=[1]) 沿序列维度翻转
        x_rev = torch.flip(x, dims=[1])
        backward_out = self.backward_block(x_rev)
        # 再次翻转回来,保持与 forward_out 对齐
        backward_out = torch.flip(backward_out, dims=[1])

        # ====== 步骤 3: 融合两个方向 ======
        if self.fusion == 'concat':
            # 方法 1: 拼接
            # forward_out: (batch, seq, d_model)
            # backward_out: (batch, seq, d_model)
            # 拼接后: (batch, seq, d_model * 2)
            combined = torch.cat([forward_out, backward_out], dim=-1)
            # 投影回原始维度: (batch, seq, d_model)
            output = self.fusion_proj(combined)

        elif self.fusion == 'add':
            # 方法 2: 直接相加
            # 每个位置的两个方向表示相加
            output = forward_out + backward_out

        elif self.fusion == 'gate':
            # 方法 3: 门控融合 (推荐!)
            # 思想: 让模型学习每个方向的重要程度
            combined = torch.cat([forward_out, backward_out], dim=-1)
            # gate: 0-1 之间的值,表示"应该相信哪个方向"
            gate = torch.sigmoid(self.gate_proj(combined))
            # value: 融合后的值
            value = self.value_proj(combined)
            # 输出 = gate * value + (1-gate) * forward + (1-gate) * backward
            # 这样即使 gate 很小,也不会完全丢失某个方向的信息
            output = gate * value + (1 - gate) * forward_out + (1 - gate) * backward_out
            output = output / 2  # 归一化,防止数值太大

        else:
            # 默认: 相加
            output = forward_out + backward_out

        return output


class BiMambaEncoder(nn.Module):
    """
    Bi-Mamba 编码器

    包含:
    1. 词嵌入层: 把 token ID 转换为向量
    2. 位置嵌入: 给模型位置信息 (SMILES 序列的位置也很重要!)
    3. 多个 BiMambaBlock: 层层堆叠,增加模型的表达能力
    4. LayerNorm: 稳定训练

    流程:
    -----
    Input: [CLS] C C O [SEP] [PAD] [PAD]
           ↓ 词嵌入 + 位置嵌入
    Tokens: [e1, e2, e3, e4, e5, e6, e7]
           ↓ 通过 N 层 BiMambaBlock
    Output: [h1, h2, h3, h4, h5, h6, h7]
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
    ):
        """
        初始化编码器

        参数:
        -----
        vocab_size : int
            词表大小,你的 tokenizer 有多少个不同的 token

        n_layers : int
            有多少层 BiMambaBlock
            层数越多,模型越深,能学习越复杂的模式
            但也更容易梯度消失/爆炸
            常见值: 2, 4, 6, 8
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # ====== 词嵌入层 ======
        # 把 token ID (数字) 转换为向量
        # 类似于查表: token_id=5 -> embedding[5]
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx,
        )

        # ====== 位置嵌入 ======
        # 告诉模型每个 token 在序列中的位置
        # 为什么需要? 因为序列顺序很重要!
        # 例如 "CCO" (乙醇) 和 "COC" (甲醚) 是不同的分子
        self.position_embedding = nn.Embedding(max_len, d_model)

        # ====== Dropout ======
        # 训练时随机丢弃一些连接,防止过拟合
        self.embedding_dropout = nn.Dropout(dropout)

        # ====== 堆叠多个 BiMambaBlock ======
        # 创建 n_layers 个双向 Mamba 块
        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                norm_eps=norm_eps,
                use_mamba=use_mamba,
                fusion=fusion,
            )
            for _ in range(n_layers)  # 循环创建 n_layers 层
        ])

        # ====== 最终的 LayerNorm ======
        # 让每层输出更稳定
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)

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
            例如: [[1, 5, 3, 2, 0, 0], [1, 8, 9, 2, 3, 0]]

        attention_mask : Optional[torch.Tensor]
            注意力掩码,标识哪些是真实 token,哪些是 padding
            1 = 真实 token, 0 = padding
            (本实现中暂时未使用,因为 SSM 不需要 attention)

        返回:
        -----
        torch.Tensor
            编码后的向量,形状 (batch_size, seq_len, d_model)
        """
        batch, seq_len = input_ids.shape

        # ====== 步骤 1: 词嵌入 ======
        # token ID -> 向量表示
        # input_ids: (batch, seq) -> x: (batch, seq, d_model)
        x = self.token_embedding(input_ids)

        # ====== 步骤 2: 位置嵌入 ======
        # 生成位置编码: [0, 1, 2, 3, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(position_ids)
        # 词嵌入 + 位置嵌入 = 最终的 token 表示
        x = x + pos_emb

        # ====== 步骤 3: Dropout ======
        # 训练时随机丢弃一些信息,防止过拟合
        x = self.embedding_dropout(x)

        # ====== 步骤 4: 通过每一层 BiMambaBlock ======
        for layer in self.layers:
            x = layer(x)

        # ====== 步骤 5: 最终归一化 ======
        x = self.norm(x)

        return x

    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.d_model


class BiMambaModel(nn.Module):
    """
    完整的 Bi-Mamba 模型

    在 BiMambaEncoder 基础上添加了 Pooling (池化) 层,
    把变长的序列变成固定长度的向量,用于下游任务

    Pooling 方法:
    -------------
    1. Mean Pooling (平均池化): 对所有 token 求平均
       - 最常用,能综合整个序列的信息

    2. Max Pooling (最大池化): 对每个维度取最大值
       - 保留最显著的特征

    3. CLS Pooling: 使用特殊 [CLS] token 的表示
       - 类似于 BERT 的做法
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
        pool_type: str = 'mean',  # 'mean', 'max', 'cls'
    ):
        """
        初始化完整模型
        """
        super().__init__()

        # 编码器
        self.encoder = BiMambaEncoder(
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
        )

        self.pool_type = pool_type
        self.d_model = d_model

        # [CLS] token - 用于 CLS pooling
        # 这是一个可学习的特殊 token,其表示会代表整个序列
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

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
            输入的 token ID

        attention_mask : Optional[torch.Tensor]
            注意力掩码 (可选)

        返回:
        -----
        torch.Tensor
            池化后的向量,形状 (batch_size, d_model)
        """
        # 1. 编码: (batch, seq, d_model)
        hidden = self.encoder(input_ids, attention_mask)

        # 2. 池化: (batch, seq, d_model) -> (batch, d_model)
        if self.pool_type == 'cls':
            # ====== CLS Pooling ======
            # 在序列前面添加 [CLS] token
            cls_tokens = self.cls_token.expand(input_ids.size(0), -1, -1)
            # 把 [CLS] 的表示加到序列第一个位置
            pooled = cls_tokens + hidden[:, 0:1, :]
            # (batch, 1, d_model) -> (batch, d_model)
            pooled = pooled.squeeze(1)

        elif self.pool_type == 'mean':
            # ====== Mean Pooling ======
            # 对所有 token 求平均
            if attention_mask is not None:
                # 有 mask 时,只对非 padding 位置求平均
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
                # 只对有效位置求和,然后除以有效数量
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                # 没有 mask,直接对所有位置求平均
                pooled = hidden.mean(dim=1)

        elif self.pool_type == 'max':
            # ====== Max Pooling ======
            # 对每个维度取最大值
            pooled = hidden.max(dim=1)[0]
        else:
            # 默认: mean pooling
            pooled = hidden.mean(dim=1)

        return pooled


def test_bi_mamba():
    """测试 Bi-Mamba 实现"""
    batch = 2
    seq_len = 32
    vocab_size = 100
    d_model = 64

    print("Testing Bi-Mamba Encoder...")

    # 测试编码器
    encoder = BiMambaEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        use_mamba=False,  # 使用手动实现
        fusion='gate',
    )

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    out = encoder(input_ids)
    print(f"Encoder input: {input_ids.shape} -> output: {out.shape}")

    # 测试完整模型
    model = BiMambaModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        use_mamba=False,
        pool_type='mean',
    )

    pooled = model(input_ids)
    print(f"Model input: {input_ids.shape} -> pooled: {pooled.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_bi_mamba()
