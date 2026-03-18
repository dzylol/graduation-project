"""
Bi-Mamba 模型实现 - 分子性质预测

本模块实现了基于双向 Mamba 架构的分子性质预测模型。
主要包含三个类：
1. BiMambaBlock - 双向 Mamba 块（核心计算单元）
2. BiMambaEncoder - 双向 Mamba 编码器（处理整个序列）
3. BiMambaForPropertyPrediction - 完整的预测模型（包含编码器和预测头）

Mamba 是一种状态空间模型（State Space Model），它可以高效地处理长序列，
计算复杂度为 O(N)，比 Transformer 的 O(N^2) 更适合处理长分子序列。

作者: Bi-Mamba-Chem Team
"""

import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
from typing import Optional, Tuple, Union  # 类型提示
import math  # 数学函数


# =============================================================================
# 第一部分：BiMambaBlock - Mamba 的核心计算单元
# =============================================================================


class BiMambaBlock(nn.Module):
    """
    双向 Mamba 块 - 这是 Mamba 模型的核心组件

    Mamba 是一种选择性状态空间模型（Selective State Space Model）。
    与传统 RNN 不同，Mamba 可以根据输入内容选择性地记住或忘记信息。

    主要包含以下几个部分：
    1. 输入投影：将输入向量投影到更高的维度
    2. 一维卷积：捕捉局部模式
    3. 状态空间模型：选择性扫描，处理序列信息
    4. 门控机制：控制信息流动
    5. 输出投影：将结果投影回原始维度

    参数说明：
    - d_model: 输入和输出的维度
    - d_state: 状态维度，控制模型的"记忆容量"
    - d_conv: 卷积核大小，用于捕捉局部特征
    - expand: 扩展因子，决定内部维度 (d_inner = d_model * expand)
    """

    def __init__(
        self,
        d_model: int,  # 输入/输出维度，如 256
        d_state: int = 16,  # 状态维度，默认为 16，控制模型记忆容量
        d_conv: int = 4,  # 卷积核大小，默认为 4
        expand: int = 2,  # 扩展因子，内部维度 = d_model * expand
        dt_rank: Union[int, str] = "auto",  # Delta 参数的秩，"auto"表示自动计算
        dt_min: float = 0.001,  # Delta 参数的最小值
        dt_max: float = 0.1,  # Delta 参数的最大值
        dt_init: str = "random",  # Delta 初始化方式
        dt_scale: float = 1.0,  # Delta 缩放因子
        dt_init_floor: float = 1e-4,  # Delta 初始化下界
        conv_bias: bool = True,  # 卷积是否使用偏置
        bias: bool = False,  # 线性层是否使用偏置
        use_fast_path: bool = True,  # 是否使用快速路径（未来可能实现）
        layer_idx: Optional[int] = None,  # 层索引，用于调试
        device: Optional[str] = None,  # 设备类型
        dtype: Optional[torch.dtype] = None,  # 数据类型
    ):
        # 将设备和数据类型放入字典，方便后续传递
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()  # 调用父类初始化

        # 保存配置参数
        self.d_model = d_model  # 输入维度
        self.d_state = d_state  # 状态维度（类似 RNN 的隐藏状态大小）
        self.d_conv = d_conv  # 卷积核大小
        self.expand = expand  # 扩展因子
        self.d_inner = int(self.expand * self.d_model)  # 内部维度（扩展后的维度）

        # 计算 Delta 的秩：如果 d_model=256，则 dt_rank=16
        # Delta 控制状态更新速度
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)

        self.use_fast_path = use_fast_path  # 快速路径标志
        self.layer_idx = layer_idx  # 层索引

        # -------------------------------------------------------------------------
        # 1. 输入投影层 (in_proj)
        # 作用：将输入向量 x (维度 d_model) 投影到更高维度的空间 (维度 d_inner * 2)
        # 为什么要 *2？因为我们要同时生成两个向量：一个是主路径 x，一个是门控信号 z
        # -------------------------------------------------------------------------
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        # -------------------------------------------------------------------------
        # 2. 一维卷积层 (conv1d)
        # 作用：在序列维度上进行局部特征提取，类似于 CNN
        # 这帮助模型捕捉 SMILES 字符串中的局部模式，如官能团
        # -------------------------------------------------------------------------
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,  # 输入通道数
            out_channels=self.d_inner,  # 输出通道数
            bias=conv_bias,  # 是否使用偏置
            kernel_size=d_conv,  # 卷积核大小
            groups=self.d_inner,  # 深度可分离卷积，每个通道独立卷积
            padding=d_conv - 1,  # 填充，使输出长度与输入相同
            **factory_kwargs,
        )

        # -------------------------------------------------------------------------
        # 3. 激活函数
        # SiLU (Sigmoid Linear Unit): x * sigmoid(x)，比 ReLU 更平滑
        # -------------------------------------------------------------------------
        self.activation = nn.SiLU()

        # -------------------------------------------------------------------------
        # 4. 状态空间参数
        # x_proj: 将输入投影到三个部分
        #   - Delta (dt_rank 维): 控制状态更新速度
        #   - B (d_state 维): 输入对状态的影响权重
        #   - C (d_state 维): 状态对输出的影响权重
        # dt_proj: 将 Delta 投影到正确的维度
        # -------------------------------------------------------------------------
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # -------------------------------------------------------------------------
        # 5. 初始化 Delta 投影层的权重
        # 使用均匀分布随机初始化
        # -------------------------------------------------------------------------
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"未知的 dt_init 类型: {dt_init}")

        # -------------------------------------------------------------------------
        # 6. 初始化 Delta 投影层的偏置
        # 确保 softplus(dt_proj.bias) 的值在 [dt_min, dt_max] 范围内
        # softplus(x) = log(1 + exp(x))，是一个平滑的 ReLU
        # -------------------------------------------------------------------------
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inv_dt = softplus^{-1}(dt)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # 防止权重重新初始化

        # -------------------------------------------------------------------------
        # 7. A 矩阵（S4D 初始化）
        # A 矩阵控制状态之间的转移强度
        # 使用负值，确保状态会随时间衰减（防止数值爆炸）
        # -------------------------------------------------------------------------
        A = torch.arange(
            1, self.d_state + 1, dtype=torch.float32
        )  # [1, 2, ..., d_state]
        if device is not None:
            A = A.to(device)
        A = A.repeat(self.d_inner, 1).contiguous()  # 形状: (d_inner, d_state)
        A_log = torch.log(A)  # 取对数，便于优化
        self.A_log = nn.Parameter(A_log)  # 注册为可学习参数
        self.A_log._no_weight_decay = True  # 权重衰减时不更新 A

        # -------------------------------------------------------------------------
        # 8. D 向量（跳跃连接参数）
        # 类似于 ResNet 的跳跃连接，允许信息直接跳过 SSM
        # -------------------------------------------------------------------------
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.D._no_weight_decay = True  # 权重衰减时不更新 D

        # -------------------------------------------------------------------------
        # 9. 输出投影层 (out_proj)
        # 作用：将内部维度 d_inner 投影回原始维度 d_model
        # -------------------------------------------------------------------------
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        数据流动过程：
        1. 输入投影 + 分割：hidden_states -> [x, z]
        2. 卷积 + 激活：x -> x_conv
        3. SSM 选择性扫描：x_conv -> y
        4. 门控：y * silu(z)
        5. 输出投影：y_gated -> output

        Args:
            hidden_states: (B, L, D) 批次数×序列长度×维度

        Returns:
            output: (B, L, D) 同样形状的输出
        """
        batch, seqlen, dim = hidden_states.shape

        # -------------------------------------------------------------------------
        # 步骤 1: 输入投影并分割
        # in_proj 将 (B, L, D) -> (B, L, 2*D_inner)
        # chunk(2) 将其分成两份：(B, L, D_inner) 和 (B, L, D_inner)
        # x 用于主路径，z 用于门控
        # -------------------------------------------------------------------------
        xz = self.in_proj(hidden_states)  # (B, L, 2 * D_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner), (B, L, D_inner)

        # -------------------------------------------------------------------------
        # 步骤 2: 一维卷积（局部特征提取）
        # 注意：PyTorch 的 Conv1d 期望 (B, C, L) 格式
        # 所以需要先 transpose，卷积后再 transpose 回来
        # -------------------------------------------------------------------------
        x = x.transpose(1, 2)  # (B, D_inner, L)
        x = self.conv1d(x)[:, :, :seqlen]  # (B, D_inner, L)
        x = x.transpose(1, 2)  # (B, L, D_inner)

        # -------------------------------------------------------------------------
        # 步骤 3: 激活函数
        # SiLU(x) = x * sigmoid(x)，比 ReLU 更平滑
        # -------------------------------------------------------------------------
        x = self.activation(x)

        # -------------------------------------------------------------------------
        # 步骤 4: 状态空间模型（核心）
        # 这是 Mamba 与传统 SSM 的关键区别：选择性扫描
        # -------------------------------------------------------------------------
        y = self.ssm(x)

        # -------------------------------------------------------------------------
        # 步骤 5: 门控机制
        # 使用 z 的激活值来门控 y
        # 这允许模型动态控制信息流动
        # silu(z) = z * sigmoid(z)
        # -------------------------------------------------------------------------
        y = y * F.silu(z)

        # -------------------------------------------------------------------------
        # 步骤 6: 输出投影
        # 将 (B, L, D_inner) -> (B, L, D_model)
        # -------------------------------------------------------------------------
        output = self.out_proj(y)

        return output

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        状态空间模型（Selective Scan）

        这个函数实现了 Mamba 的核心：选择性扫描机制。
        传统的 SSM 使用固定的 A, B, C 矩阵，但 Mamba 使这些参数
        依赖于输入数据，从而实现"选择性"记忆。

        Args:
            x: (B, L, d_inner) 输入张量

        Returns:
            y: (B, L, d_inner) 状态空间模型的输出
        """
        batch, seqlen, dim = x.shape

        # -------------------------------------------------------------------------
        # 使用 x_proj 将 x 投影到三个部分：
        # - dt: (B, L, dt_rank) -> 控制状态更新速度
        # - B: (B, L, d_state) -> 输入对状态的影响
        # - C: (B, L, d_state) -> 状态对输出的影响
        # -------------------------------------------------------------------------
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # -------------------------------------------------------------------------
        # 对 Delta 进行投影和激活
        # softplus 确保 Delta 为正
        # -------------------------------------------------------------------------
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)

        # -------------------------------------------------------------------------
        # 执行选择性扫描
        # 这是一个简化的实现，真实的高效实现在 mamba-ssm 库中
        # -------------------------------------------------------------------------
        y = self.selective_scan(x, dt, B, C)  # (B, L, d_inner)

        return y

    def selective_scan(
        self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """
        选择性扫描的核心实现

        这是 Mamba 最重要的部分！它实现了一种"选择性"的状态更新机制。

        核心思想：
        1. 传统的 SSM 使用固定的转移矩阵 A，但 Mamba 让 A 依赖输入
        2. 这样模型可以决定哪些信息要记住，哪些要忘记

        数学公式：
        - 离散化: dA = exp(dt * A), dB = dt * B
        - 状态更新: h_new = dA * h + dB * x
        - 输出: y = C * h_new

        Args:
            x: (B, L, d_inner) 输入
            dt: (B, L, d_inner) 时间步长（依赖输入）
            B: (B, L, d_state) B 矩阵（依赖输入）
            C: (B, L, d_state) C 矩阵（依赖输入）

        Returns:
            y: (B, L, d_inner) 输出
        """
        batch, seqlen, dim = x.shape
        d_state = self.d_state

        # -------------------------------------------------------------------------
        # 获取 A 矩阵和 D 向量
        # A_log 存储 log(A)，取指数得到 A
        # A 是 (d_inner, d_state) 维度的矩阵
        # -------------------------------------------------------------------------
        A = -torch.exp(self.A_log)  # (d_inner, d_state) 负值确保稳定
        D = self.D  # (d_inner,) 跳跃连接

        # -------------------------------------------------------------------------
        # 初始化隐藏状态
        # h[b,d,n] 表示第 b 个样本，第 d 个维度，第 n 个状态分量
        # -------------------------------------------------------------------------
        h = torch.zeros(batch, dim, d_state, device=x.device, dtype=x.dtype)
        outputs = []  # 存储每一步的输出

        # -------------------------------------------------------------------------
        # 循环遍历序列的每个时间步
        # 这是简化的实现，真实使用并行扫描算法
        # -------------------------------------------------------------------------
        for t in range(seqlen):
            # 获取第 t 个时间步的参数
            dt_t = dt[:, t, :]  # (B, d_inner) 时间步长
            B_t = B[:, t, :]  # (B, d_state) B 矩阵
            C_t = C[:, t, :]  # (B, d_state) C 矩阵
            x_t = x[:, t, :]  # (B, d_inner) 输入

            # -------------------------------------------------------------------------
            # 离散化：将连续系统转换为离散系统
            # dA[b,d,n] = exp(dt[b,d] * A[d,n])
            # dB[b,d,n] = dt[b,d] * B[b,n]
            # -------------------------------------------------------------------------
            # 限制 dt_t 的范围，防止数值爆炸
            dt_t_clamped = torch.clamp(dt_t, min=-10, max=10)
            # dA 形状: (B, d_inner, d_state)
            dA = torch.exp(
                torch.clamp(
                    dt_t_clamped.unsqueeze(-1) * A.unsqueeze(0), min=-50, max=50
                )
            )
            # dB 形状: (B, d_inner, d_state)
            dB = dt_t_clamped.unsqueeze(-1) * B_t.unsqueeze(1)

            # -------------------------------------------------------------------------
            # 状态更新
            # h_new[b,d,n] = sum_k(dA[b,d,k] * h[b,d,k]) + dB[b,d,n] * x[b,d]
            # -------------------------------------------------------------------------
            # 限制输入范围
            x_t_clamped = torch.clamp(x_t, min=-10, max=10)
            h_new = torch.einsum("bdn,bdk->bdk", dA, h) + dB * x_t_clamped.unsqueeze(-1)
            # 限制状态范围，防止数值爆炸
            h_new = torch.clamp(h_new, min=-100, max=100)

            # -------------------------------------------------------------------------
            # 计算输出
            # y[b,d] = sum_n(h_new[b,d,n] * C[b,n])
            # -------------------------------------------------------------------------
            y_t = torch.sum(h_new * C_t.unsqueeze(1), dim=2)  # (B, d_inner)
            outputs.append(y_t)

            # 更新状态，供下一步使用
            h = h_new

        # -------------------------------------------------------------------------
        # 将所有时间步的输出堆叠起来
        # -------------------------------------------------------------------------
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        # -------------------------------------------------------------------------
        # 跳跃连接
        # y = y + D * x
        # 这类似于 ResNet 的残差连接
        # -------------------------------------------------------------------------
        y = y + x * D.unsqueeze(0)

        return y


# =============================================================================
# 第二部分：BiMambaEncoder - 双向编码器
# =============================================================================


class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器

    这个编码器同时从左到右和从右到左处理序列，然后将两个方向的
    表示进行融合。这种设计对于分子性质预测特别重要，因为：
    1. 分子的化学环境受左右两侧原子共同影响
    2. 官能团之间可能存在长程相互作用

    主要特点：
    1. 词嵌入 + 位置嵌入
    2. 前向扫描分支
    3. 后向扫描分支
    4. 门控融合机制
    """

    def __init__(
        self,
        vocab_size: int,  # 词表大小
        d_model: int = 256,  # 模型维度
        n_layers: int = 4,  # Mamba 层数
        d_state: int = 16,  # 状态维度
        d_conv: int = 4,  # 卷积核大小
        expand: int = 2,  # 扩展因子
        max_seq_length: int = 512,  # 最大序列长度
        dropout: float = 0.1,  # Dropout 比率
        pad_token_id: int = 0,  # 填充 token 的 ID
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # 保存配置
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        # -------------------------------------------------------------------------
        # 1. 嵌入层
        # -------------------------------------------------------------------------

        # 词嵌入：将 token ID 转换为向量
        # padding_idx 确保填充 token 的嵌入为零
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id, **factory_kwargs
        )

        # 位置嵌入：为每个位置添加位置信息
        # 注意：Mamba 本身不包含位置信息，需要手动添加
        self.position_embedding = nn.Embedding(
            max_seq_length, d_model, **factory_kwargs
        )

        # -------------------------------------------------------------------------
        # 2. 前向扫描分支（Forward）
        # 从左到右处理序列
        # -------------------------------------------------------------------------
        self.forward_layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **factory_kwargs,
                )
                for _ in range(n_layers)  # 创建 n_layers 个 Mamba 块
            ]
        )

        # -------------------------------------------------------------------------
        # 3. 后向扫描分支（Backward）
        # 从右到左处理序列
        # -------------------------------------------------------------------------
        self.backward_layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **factory_kwargs,
                )
                for _ in range(n_layers)
            ]
        )

        # -------------------------------------------------------------------------
        # 4. 层归一化
        # 稳定训练，加速收敛
        # -------------------------------------------------------------------------
        self.norm = nn.LayerNorm(d_model, **factory_kwargs)

        # -------------------------------------------------------------------------
        # 5. Dropout
        # 防止过拟合
        # -------------------------------------------------------------------------
        self.dropout = nn.Dropout(dropout)

        # -------------------------------------------------------------------------
        # 6. 门控融合层
        # 融合前向和后向的表示
        # 输入: 2*d_model (前向 + 后向)
        # 输出: 2*d_model (门控值)
        # -------------------------------------------------------------------------
        self.fusion_gate = nn.Linear(d_model * 2, d_model * 2, **factory_kwargs)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        数据流动：
        1. 词嵌入 + 位置嵌入
        2. 前向扫描（n_layers 次）
        3. 后向扫描（n_layers 次）
        4. 门控融合
        5. 层归一化

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) 注意力掩码

        Returns:
            hidden_states: (B, L, D) 编码后的表示
        """
        batch_size, seq_len = input_ids.shape

        # -------------------------------------------------------------------------
        # 创建位置 ID
        # 例如序列长度为 10，位置 ID 为 [0, 1, 2, ..., 9]
        # -------------------------------------------------------------------------
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # -------------------------------------------------------------------------
        # 获取嵌入
        # -------------------------------------------------------------------------
        token_embeds = self.token_embedding(input_ids)  # (B, L, D)
        position_embeds = self.position_embedding(position_ids)  # (B, L, D)

        # 嵌入相加并应用 Dropout
        hidden_states = self.dropout(token_embeds + position_embeds)

        # -------------------------------------------------------------------------
        # 应用注意力掩码（如果提供）
        # 将填充位置的嵌入置零
        # -------------------------------------------------------------------------
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        # -------------------------------------------------------------------------
        # 前向扫描分支（从左到右）
        # -------------------------------------------------------------------------
        forward_hidden = hidden_states
        for layer in self.forward_layers:
            forward_hidden = layer(forward_hidden)

        # -------------------------------------------------------------------------
        # 后向扫描分支（从右到左）
        # -------------------------------------------------------------------------
        # 反转序列，使模型从右到左处理
        backward_hidden = torch.flip(hidden_states, [1])  # 反转 L 维度

        # 通过后向 Mamba 层
        for layer in self.backward_layers:
            backward_hidden = layer(backward_hidden)

        # 再次反转，回到原始顺序
        backward_hidden = torch.flip(backward_hidden, [1])

        # -------------------------------------------------------------------------
        # 门控融合
        # 这是关键步骤：让模型学习如何组合前向和后向信息
        # -------------------------------------------------------------------------
        # 拼接前向和后向表示
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)  # (B, L, 2*D)

        # 计算门控值（使用 sigmoid 将值压缩到 0-1 之间）
        gate = torch.sigmoid(self.fusion_gate(combined))  # (B, L, 2*D)

        # 将门控分成两部分
        gate_forward, gate_backward = gate.chunk(2, dim=-1)

        # 加权融合
        # 如果门控接近 1，则更多使用对应方向的表示
        fused_hidden = gate_forward * forward_hidden + gate_backward * backward_hidden

        # -------------------------------------------------------------------------
        # 最终归一化
        # -------------------------------------------------------------------------
        output = self.norm(fused_hidden)

        return output


# =============================================================================
# 第三部分：BiMambaForPropertyPrediction - 完整的预测模型
# =============================================================================


class BiMambaForPropertyPrediction(nn.Module):
    """
    Bi-Mamba 分子性质预测模型

    这是完整的预测模型，包含：
    1. BiMambaEncoder - 编码器
    2. 池化层 - 将序列表示聚合成单个向量
    3. 预测头 - 输出最终预测

    支持两种任务类型：
    - 回归任务：预测连续值（如溶解度）
    - 分类任务：预测类别（如是否有毒性）
    """

    def __init__(
        self,
        vocab_size: int,  # 词表大小
        d_model: int = 256,  # 模型维度
        n_layers: int = 4,  # Mamba 层数
        d_state: int = 16,  # 状态维度
        d_conv: int = 4,  # 卷积核大小
        expand: int = 2,  # 扩展因子
        max_seq_length: int = 512,  # 最大序列长度
        num_labels: int = 1,  # 输出标签数（回归=1，多标签分类>1）
        task_type: str = "regression",  # 任务类型
        pooling: str = "mean",  # 池化方法
        dropout: float = 0.1,  # Dropout 比率
        pad_token_id: int = 0,  # 填充 token ID
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # 保存配置
        self.num_labels = num_labels
        self.task_type = task_type
        self.pooling = pooling
        self.pad_token_id = pad_token_id

        # -------------------------------------------------------------------------
        # 创建编码器
        # -------------------------------------------------------------------------
        self.encoder = BiMambaEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_token_id=pad_token_id,
            **factory_kwargs,
        )

        # -------------------------------------------------------------------------
        # [CLS] token（如果使用 CLS 池化）
        # 这是 BERT 风格的做法
        # -------------------------------------------------------------------------
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs))

        # -------------------------------------------------------------------------
        # 预测头
        # -------------------------------------------------------------------------
        self.dropout = nn.Dropout(dropout)

        # 线性分类/回归层
        self.classifier = nn.Linear(d_model, num_labels, **factory_kwargs)

        # -------------------------------------------------------------------------
        # 损失函数
        # -------------------------------------------------------------------------
        if task_type == "regression":
            # 均方误差损失，用于回归任务
            self.loss_fct = nn.MSELoss()
        else:
            # 二元交叉熵损失，用于分类任务
            self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        数据流动：
        1. （可选）添加 [CLS] token
        2. 编码器处理
        3. 池化
        4. 分类头
        5. 计算损失（如果提供标签）

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) 注意力掩码
            labels: (B,) 或 (B, num_labels) 目标值

        Returns:
            logits: 预测值
            loss: 损失值（如果提供 labels）
        """
        batch_size, seq_len = input_ids.shape

        # -------------------------------------------------------------------------
        # 如果使用 CLS 池化，在序列前面添加 [CLS] token
        # -------------------------------------------------------------------------
        if self.pooling == "cls":
            # 扩展 CLS token 到当前 batch
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)

            # 在序列前面添加 CLS token
            input_ids = torch.cat(
                [
                    torch.full(
                        (batch_size, 1),
                        self.pad_token_id,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    input_ids,
                ],
                dim=1,
            )

            # 更新 attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                        attention_mask,
                    ],
                    dim=1,
                )

        # -------------------------------------------------------------------------
        # 通过编码器
        # -------------------------------------------------------------------------
        encoder_outputs = self.encoder(input_ids, attention_mask)  # (B, L, D)

        # -------------------------------------------------------------------------
        # 池化操作
        # 将序列表示聚合成单个向量
        # -------------------------------------------------------------------------
        if self.pooling == "mean":
            # 平均池化：对所有位置的表示取平均
            if attention_mask is not None:
                # 加权平均，只考虑有效位置
                sum_embeddings = torch.sum(
                    encoder_outputs * attention_mask.unsqueeze(-1), dim=1
                )
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                pooled_output = sum_embeddings / sum_mask.clamp(min=1e-9)
            else:
                pooled_output = torch.mean(encoder_outputs, dim=1)

        elif self.pooling == "max":
            # 最大池化：取每个维度的最大值
            if attention_mask is not None:
                # 将无效位置设为很小的值，这样它们不会被选中
                masked_embeddings = encoder_outputs.clone()
                masked_embeddings[attention_mask == 0] = -1e9
                pooled_output = torch.max(masked_embeddings, dim=1)[0]
            else:
                pooled_output = torch.max(encoder_outputs, dim=1)[0]

        elif self.pooling == "cls":
            # 使用第一个 token（[CLS]）的表示
            pooled_output = encoder_outputs[:, 0]
        else:
            raise ValueError(f"未知的池化方法: {self.pooling}")

        # -------------------------------------------------------------------------
        # 预测
        # -------------------------------------------------------------------------
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # -------------------------------------------------------------------------
        # 调整输出形状
        # -------------------------------------------------------------------------
        if self.num_labels == 1:
            logits = logits.squeeze(-1)  # (B,) 而不是 (B, 1)

        # -------------------------------------------------------------------------
        # 计算损失
        # -------------------------------------------------------------------------
        loss = None
        if labels is not None:
            # 确保 logits 和 labels 形状一致
            # 如果 labels 是 (B, 1) 而 logits 是 (B,)，需要 squeeze
            if labels.dim() > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)  # (B, 1) -> (B,)

            if self.task_type == "regression":
                loss = self.loss_fct(logits, labels)
            else:
                # 确保标签类型正确
                if labels.dtype != logits.dtype:
                    labels = labels.to(logits.dtype)
                loss = self.loss_fct(logits, labels)

        return logits, loss


# =============================================================================
# 辅助函数
# =============================================================================


def create_bimamba_model(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 4,
    task_type: str = "regression",
    num_labels: int = 1,
    **kwargs,
) -> BiMambaForPropertyPrediction:
    """
    工厂函数：创建 BiMamba 模型

    这是一个便捷函数，用于创建完整的 BiMamba 预测模型。

    使用示例：
    ```python
    model = create_bimamba_model(
        vocab_size=50,
        d_model=256,
        n_layers=4,
        task_type="regression",
        num_labels=1
    )
    ```

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        n_layers: Mamba 层数
        task_type: "regression" 或 "classification"
        num_labels: 输出标签数
        **kwargs: 其他参数

    Returns:
        BiMambaForPropertyPrediction 模型实例
    """
    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        task_type=task_type,
        num_labels=num_labels,
        **kwargs,
    )
