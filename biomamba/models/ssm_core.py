"""
手动实现的 SSM (状态空间模型) 核心模块

本文件提供了 Mamba/SSM 的手动实现,可以在没有安装 mamba-ssm 包的情况下运行。

什么是状态空间模型 (SSM)?
=========================
想象你在看一部电影:
- 电影画面 = 输入
- 你的记忆 = 状态
- 你对下一帧的预期 = 输出

SSM 的核心思想:
1. 有一个"状态"来记住过去的信息
2. 根据当前输入更新状态
3. 根据状态生成输出

与 Transformer 的区别:
- Transformer: 直接计算任意两个位置之间的关系 (O(N²) 复杂度)
- SSM: 通过隐藏状态传递信息 (O(N) 复杂度)

对于长分子序列,SSM 更高效!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SSMCore(nn.Module):
    """
    SSM 核心实现 (手动版)

    这个类实现了 Mamba 的核心逻辑:
    1. 输入投影: 把原始向量投影到更高维空间
    2. 卷积: 捕捉局部上下文
    3. SSM 扫描: 选择性地处理序列
    4. 输出投影: 把高维向量投影回原始维度

    简化理解:
    ---------
    这就像一个"信息处理流水线":
    输入 -> 卷积局部信息 -> SSM全局记忆 -> 输出

    每个步骤都有可学习的参数,让模型自动学会最好的处理方式!
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        """
        初始化 SSM 核心

        参数:
        -----
        d_model : int
            输入/输出的维度

        d_state : int
            状态维度,可以理解为"记忆槽"的数量
            越大能记住更多信息

        d_conv : int
            卷积核大小,用于捕捉局部上下文

        expand : int
            内部扩展因子

        dropout : float
            Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ====== 步骤 1: 输入投影 ======
        # 把 d_model 维的输入扩展到 d_inner * 2 维
        # 为什么要 *2? 因为需要同时生成两个向量: x 和 z
        # 类似于 Gating 机制中的 gate 和 value
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # ====== 步骤 2: 一维卷积 ======
        # 在处理序列之前,先用卷积捕捉局部上下文
        # 这就像在读小说之前先看每一段话的概要
        self.conv1d = nn.Conv1d(
            in_channels=d_model,        # 输入通道数
            out_channels=self.d_inner, # 输出通道数
            kernel_size=d_conv,        # 卷积核大小
            padding=d_conv - 1,        # 填充,保持序列长度
            groups=1,
        )

        # ====== SSM 参数 (A, B, C, D) ======
        # 这些是状态空间模型的核心参数!

        # A 矩阵: 状态之间的转移矩阵
        # 决定"上一时刻的状态如何影响当前状态"
        # 我们使用对角矩阵 + 低秩近似来简化
        # 初始值是随机的,训练时会自动学习
        # 使用更安全的初始化: 从小的均匀分布开始
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)

        # D 向量: 直接跳过连接
        # 类似于残差连接,让输入直接影响输出
        # 初始化为小的正值,避免初始输出过大
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)

        # ====== 投影层 ======
        # 用于生成 B 和 C 参数 (这些参数是"动态"的,取决于输入!)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)

        # 输出投影: 把内部维度转回原始维度
        self.o_proj = nn.Linear(self.d_inner, d_model)

        # ====== 选择门控 ======
        # 这是 Mamba 的关键创新!
        # 让模型选择性地关注或忽略某些信息
        self.ssm_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner),
            nn.Sigmoid()  # 输出 0-1 之间的值
        )

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """使用 Xavier 初始化权重"""
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.conv1d.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def _discretize(self, A, B, C, delta, dt):
        """
        离散化连续时间 SSM

        概念解释:
        ---------
        想象你在开车:
        - 连续时间: 速度随时变化 (微分方程)
        - 离散时间: 每秒记录一次速度 (差分方程)

        这里我们把连续时间的 SSM 转换为离散时间版本,
        这样才能在计算机上高效计算!

        参数:
        -----
        A: 状态转移矩阵
        B: 输入到状态的映射
        C: 状态到输出的映射
        delta: 输入的嵌入表示
        dt: 时间步长

        返回:
        -----
        离散化后的 B 和 C
        """
        # A_bar = exp(A * dt)
        # 这就是离散化的关键: 使用指数函数!
        A_bar = A.unsqueeze(1).unsqueeze(1) * dt.unsqueeze(-1)
        A_bar = torch.exp(A_bar)

        # B_bar = A_bar * B * dt
        B_bar = A_bar * B.unsqueeze(1).unsqueeze(1) * dt.unsqueeze(-1)

        return A_bar, B_bar

    def forward(self, x, state=None):
        """
        SSM 核心的前向传播

        数据流动:
        ---------
        输入 x
           ↓
        卷积 (局部上下文)
           ↓
        输入投影 + 门控
           ↓
        SSM 扫描 (选择性扫描!)
           ↓
        输出投影
           ↓
        输出

        参数:
        -----
        x : torch.Tensor
            输入张量 (batch, seq_len, d_model)

        state : Optional
            之前的状态,用于连续处理多个序列 (可选)

        返回:
        -----
        torch.Tensor: 输出 (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # ====== 步骤 1: 卷积 ======
        # 先用卷积捕捉局部上下文
        # 需要改变维度顺序: (batch, seq, d_model) -> (batch, d_model, seq)
        x_conv = x.transpose(1, 2)
        # 卷积
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        # 再次改变维度: (batch, d_inner, seq) -> (batch, seq, d_inner)
        x_conv = x_conv.transpose(1, 2)

        # ====== 步骤 2: 输入投影 ======
        # 生成 x (主要内容) 和 z (门控信号)
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # ====== 步骤 3: SSM 参数 ======
        # A 矩阵 (确保为正数,使用 softplus)
        A = F.softplus(self.A_log)

        # 动态生成 B 和 C (选择性机制!)
        # 这是 Mamba 和传统 SSM 的关键区别:
        # B 和 C 不是固定的,而是根据输入动态生成的!
        bc = self.x_proj(x_conv)
        B, C = bc.chunk(2, dim=-1)

        # 时间步长 dt (可学习的)
        dt = torch.sigmoid(x_conv.mean(dim=-1, keepdim=True))
        dt = dt + 0.001  # 避免 dt 为 0

        # ====== 步骤 4: 门控 ======
        # 让模型选择性地关注或忽略某些信息
        gate = self.ssm_gate(x_conv)

        # ====== 步骤 5: 选择性扫描 ======
        # 这是核心操作!用选择后的输入进行 SSM 扫描
        output = self._selective_scan(
            x_conv * gate,  # 乘以门控值
            A,
            B,
            C,
            self.D,
            dt
        )

        # ====== 步骤 6: 输出投影 ======
        # 使用 z 作为门控
        output = output * torch.sigmoid(z)
        output = self.o_proj(output)
        output = self.dropout(output)

        return output

    def _selective_scan(
        self,
        x_conv_gate: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        选择性扫描操作

        这是 SSM 的核心!类似于循环神经网络,
        但使用了"选择"机制来增强表达能力。

        简化理解:
        ---------
        想象你在看一篇文章:
        - 传统 RNN: 把所有内容都记住
        - 选择性扫描: 只记住重要的地方

        这让模型能更有效地处理长序列!

        参数:
        -----
        x_conv_gate: 卷积+门控后的输入
        A: 状态转移矩阵
        B: 输入到状态的映射
        C: 状态到输出的映射
        D: 直接跳过连接
        dt: 时间步长

        返回:
        -----
        扫描后的输出
        """
        batch, seq_len, d_inner = x_conv_gate.shape
        d_state = B.shape[-1]

        # 离散化
        A_expanded = A.unsqueeze(0).unsqueeze(1)
        dt_expanded = dt.unsqueeze(-1)
        A_bar = torch.exp(A_expanded * dt_expanded)

        B_expanded = B.unsqueeze(2)
        dt_expanded2 = dt.unsqueeze(-1)
        B_bar = A_bar * B_expanded * dt_expanded2

        # 初始化隐藏状态为 0
        h = torch.zeros(
            batch, d_inner, d_state,
            device=x_conv_gate.device,
            dtype=x_conv_gate.dtype
        )

        outputs = []

        # ====== 扫描整个序列 ======
        for t in range(seq_len):
            # 状态更新: h = A_bar * h + B_bar * x
            # 这是 RNN 的核心公式!
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv_gate[:, t].unsqueeze(-1)

            # 输出: y = C * h + D * x
            y = torch.matmul(
                C[:, t].unsqueeze(1),
                h.transpose(1, 2)
            ).squeeze(1) + D * x_conv_gate[:, t]
            outputs.append(y)

        # 把所有时间步的输出堆叠起来
        output = torch.stack(outputs, dim=1)

        return output


class SSMBlock(nn.Module):
    """
    完整的 SSM 块

    在 SSMCore 基础上添加了:
    1. LayerNorm (层归一化)
    2. 残差连接

    这让模型更容易训练!
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # LayerNorm
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)

        # SSM 核心
        self.ssm = SSMCore(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x):
        """
        前向传播,带残差连接

        流程:
        -----
        输入 x
           ↓
        归一化 (norm)
           ↓
        SSM 处理 (ssm)
           ↓
        残差连接 (+ x)
           ↓
        输出
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = x + residual
        return x


class BidirectionalSSM(nn.Module):
    """
    双向 SSM

    类似于 Bi-LSTM,同时从两个方向处理序列,
    让模型能同时看到每个位置的前后文!

    这对于分子特别重要,因为:
    - 分子的化学键是双向的
    - SMILES 字符串从左往右和从右往左可能代表不同含义
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        fusion: str = 'concat',  # 'concat', 'add', 'gate'
    ):
        super().__init__()

        self.fusion = fusion

        # 前向 SSM
        self.forward_ssm = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # 后向 SSM
        self.backward_ssm = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # 融合层
        if fusion == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion == 'gate':
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            self.value_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        双向处理

        流程:
        -----
        输入 x
           ↓
        前向 SSM (forward_ssm)
           ↓
        翻转输入
           ↓
        后向 SSM (backward_ssm)
           ↓
        翻转回来
           ↓
        融合两个方向的结果
           ↓
        输出
        """
        # 前向处理
        forward_out = self.forward_ssm(x)

        # 后向处理: 翻转序列
        x_rev = torch.flip(x, dims=[1])
        backward_out = self.backward_ssm(x_rev)
        backward_out = torch.flip(backward_out, dims=[1])

        # 融合
        if self.fusion == 'concat':
            combined = torch.cat([forward_out, backward_out], dim=-1)
            output = self.fusion_proj(combined)
        elif self.fusion == 'add':
            output = forward_out + backward_out
        elif self.fusion == 'gate':
            combined = torch.cat([forward_out, backward_out], dim=-1)
            gate = torch.sigmoid(self.gate_proj(combined))
            value = self.value_proj(combined)
            output = gate * value
        else:
            output = forward_out + backward_out

        return output


if __name__ == "__main__":
    """测试 SSM 实现"""
    batch = 2
    seq_len = 32
    d_model = 64

    # 测试 SSM 核心
    ssm = SSMCore(d_model=d_model, d_state=16)
    x = torch.randn(batch, seq_len, d_model)
    out = ssm(x)
    print(f"SSM input: {x.shape}, output: {out.shape}")

    # 测试双向 SSM
    bi_ssm = BidirectionalSSM(d_model=d_model, fusion='gate')
    out = bi_ssm(x)
    print(f"Bidirectional SSM input: {x.shape}, output: {out.shape}")
