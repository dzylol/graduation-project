"""
Mamba Block 实现 - 使用 mamba-ssm 包

本文件提供了 Mamba 块的封装,可以自动选择:
1. 官方 mamba-ssm 实现 (如果有安装)
2. 手动实现的 SSM (如果没有安装)

这样设计的好处是:代码可以在任何环境下运行,
即使没有安装 mamba-ssm 包也能工作!
"""

import torch
import torch.nn as nn
from typing import Optional
import math

# ====== 尝试导入 mamba-ssm ======
# 如果导入成功,说明用户安装了官方 mamba-ssm
# 如果导入失败,我们使用手动实现的 SSM
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True  # 标记: mamba-ssm 可用
except ImportError:
    print("Warning: mamba-ssm not found. Using manual SSM implementation.")
    MAMBA_AVAILABLE = False  # 标记: mamba-ssm 不可用


class MambaBlock(nn.Module):
    """
    Mamba 块的封装类

    什么是 Mamba Block?
    -------------------
    Mamba 是一种新型的深度学习架构,叫做"状态空间模型"(State Space Model, SSM)。

    想象你正在读一本书:
    - Transformer 就像同时记住书中的每一个字 (注意力机制)
    - Mamba 就像有一个"工作记忆",只记住关键信息

    Mamba 的优势:
    1. 处理长序列时更快 (线性复杂度 O(N) vs Transformer 的 O(N²))
    2. 训练时显存占用更少
    3. 推理速度更快

    本类做了以下事情:
    1. 对输入进行 LayerNorm (层归一化,稳定训练)
    2. 通过 Mamba/SSM 处理
    3. Dropout (防止过拟合)
    4. 残差连接 (把输入加到输出,帮助梯度流动)
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
    ):
        """
        初始化 Mamba Block

        参数:
        -----
        d_model : int
            模型的隐藏维度,每个 token 向量表示的长度

        d_state : int
            SSM 状态维度,可以理解为"记忆单元的数量"
            越大能记住更多信息,但计算越慢

        d_conv : int
            一维卷积的核大小,用于捕捉局部上下文
            类似滑动窗口的大小

        expand : int
            扩展因子,内部维度 = d_model * expand
            常用值: 2

        dropout : float
            Dropout 概率,0.1 表示 10% 的神经元随机关闭
            用于防止过拟合

        norm_eps : float
            LayerNorm 的 epsilon,防止除零
            默认 1e-5 足够

        use_mamba : bool
            是否使用官方 mamba-ssm
            False 会使用手动实现的 SSM
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        # 内部维度,通常比输入维度大 expand 倍
        self.d_inner = int(expand * d_model)
        self.use_mamba = use_mamba and MAMBA_AVAILABLE

        # ====== LayerNorm (层归一化) ======
        # 作用: 把每层的输出值归一化到稳定范围
        # 就像把分数标准化,让训练更稳定
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)

        # ====== 选择使用哪种 SSM 实现 ======
        if self.use_mamba:
            # 情况 1: 使用官方 mamba-ssm
            # 这是 CUDA 优化过的,运行更快(如果有 GPU)
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # 情况 2: 使用手动实现的 SSM
            # 从同目录的 ssm_core 模块导入
            from .ssm_core import SSMCore
            self.mamba = SSMCore(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

        # ====== Dropout ======
        # 训练时随机丢弃一些连接,防止过拟合
        # 如果 dropout=0,则使用 Identity (什么都不做)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (Forward Pass)

        这就是模型"推理"时数据流动的过程:

        Input:   x (原始输入)
           ↓
        Norm:   norm(x) (归一化)
           ↓
        Mamba:  mamba(norm(x)) (SSM 处理)
           ↓
        Dropout: drop(mamba_out) (随机丢弃)
           ↓
        Add:    output + x (残差连接)
           ↓
        Output: y

        参数:
        -----
        x : torch.Tensor
            输入张量,形状 (batch_size, seq_len, d_model)

        返回:
        -----
        torch.Tensor
            输出张量,形状与输入相同
        """
        # 步骤 1: 保存原始输入 (用于残差连接)
        residual = x

        # 步骤 2: LayerNorm (归一化)
        # 这会让数值更稳定,训练更顺利
        x = self.norm(x)

        # 步骤 3: 通过 Mamba/SSM 处理
        # 这是核心操作:用状态空间模型处理序列
        x = self.mamba(x)

        # 步骤 4: Dropout (防止过拟合)
        x = self.dropout(x)

        # 步骤 5: 残差连接 (Residual Connection)
        # 输出 = 处理后的结果 + 原始输入
        # 好处: 梯度能直接传回去,训练更深层的网络
        x = x + residual

        return x


class MambaLayer(nn.Module):
    """
    简单的单层 Mamba 封装

    这只是一个便捷类,把 MambaBlock 包装一下,
    方便堆叠多层使用。
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
    ):
        super().__init__()

        # 就是一个 MambaBlock
        self.block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_eps=norm_eps,
            use_mamba=use_mamba,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """直接调用 block 的 forward"""
        return self.block(x)


def test_mamba_block():
    """测试 Mamba Block 的功能"""
    batch = 2       # 2 个样本
    seq_len = 32    # 序列长度 32
    d_model = 64   # 隐藏维度 64

    print("Testing Mamba Block...")
    print(f"MAMBA_AVAILABLE: {MAMBA_AVAILABLE}")

    # 测试两种实现方式
    for use_mamba in [False, True]:
        if use_mamba and not MAMBA_AVAILABLE:
            print("Skipping mamba-ssm test (not available)")
            continue

        print(f"\nTesting use_mamba={use_mamba}")
        block = MambaBlock(
            d_model=d_model,
            d_state=16,      # 状态维度
            d_conv=4,        # 卷积核大小
            expand=2,        # 扩展因子
            use_mamba=use_mamba,
        )

        # 创建随机输入
        x = torch.randn(batch, seq_len, d_model)
        # 前向传播
        out = block(x)
        # 打印输入输出形状
        print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_mamba_block()
