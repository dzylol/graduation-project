#!/usr/bin/env python3
"""
Bi-Mamba 模型单元测试

本脚本用于测试 Bi-Mamba 模型的各个组件是否正常工作。

测试内容：
1. BiMambaBlock - 单个 Mamba 块
2. BiMambaEncoder - 双向编码器
3. BiMambaForPropertyPrediction - 完整预测模型

使用方法：
```bash
python tests/test_model.py
```

作者: Bi-Mamba-Chem Team
"""

import torch  # PyTorch 深度学习框架
import sys  # 系统模块
import os  # 文件路径操作

# 将 src 目录添加到 Python 路径
# 这样可以直接导入 src.models.bimamba 等模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 导入待测试的模型
from models.bimamba import BiMambaForPropertyPrediction, create_bimamba_model


def test_model_components():
    """
    测试模型组件

    单独测试每个组件是否工作正常：
    1. BiMambaBlock - 单个 Mamba 块
    2. BiMambaEncoder - 双向编码器
    """
    print("\n" + "=" * 50)
    print("测试模型组件")
    print("=" * 50)

    # 导入需要测试的组件
    from models.bimamba import BiMambaEncoder, BiMambaBlock

    # -------------------------------------------------------------------------
    # 测试 1: BiMambaBlock
    # -------------------------------------------------------------------------
    print("\n[测试 1] BiMambaBlock")

    # 创建一个小型的 Mamba 块用于测试
    # d_model=64: 输入/输出维度为 64
    block = BiMambaBlock(d_model=64)

    # 创建测试输入
    # (batch_size=2, seq_len=10, d_model=64)
    x = torch.randn(2, 10, 64)

    # 前向传播
    output = block(x)

    # 验证输出形状是否正确
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")

    # 断言：输出形状应该与输入形状相同
    assert output.shape == x.shape, f"期望形状 {x.shape}，实际 {output.shape}"
    print("  ✓ BiMambaBlock 测试通过！")

    # -------------------------------------------------------------------------
    # 测试 2: BiMambaEncoder
    # -------------------------------------------------------------------------
    print("\n[测试 2] BiMambaEncoder")

    # 创建编码器
    # vocab_size=50: 词汇表大小
    # d_model=64: 模型维度
    # n_layers=2: 2 层 Mamba
    encoder = BiMambaEncoder(vocab_size=50, d_model=64, n_layers=2)

    # 创建测试输入：batch_size=2, seq_len=10
    input_ids = torch.randint(0, 50, (2, 10))

    # 前向传播
    output = encoder(input_ids)

    # 验证输出形状
    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {output.shape}")

    # 断言：输出应该是 (batch, seq_len, d_model)
    assert output.shape == (2, 10, 64), f"期望形状 (2, 10, 64)，实际 {output.shape}"
    print("  ✓ BiMambaEncoder 测试通过！")

    print("\n" + "-" * 50)
    print("所有组件测试通过！")


def test_model_creation():
    """
    测试完整模型的创建和前向传播

    测试两种任务类型：
    1. 回归任务 - 预测连续值（如溶解度）
    2. 分类任务 - 预测类别（如是否有毒）
    """
    print("\n" + "=" * 50)
    print("测试完整模型")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 测试 1: 回归模型
    # -------------------------------------------------------------------------
    print("\n[测试 1] 回归模型")

    # 创建回归模型
    model_reg = create_bimamba_model(
        vocab_size=50,  # 词汇表大小
        d_model=64,  # 模型维度
        n_layers=2,  # 层数
        task_type="regression",  # 回归任务
        num_labels=1,  # 1 个输出（单个回归值）
    )

    # 统计参数量
    num_params = sum(p.numel() for p in model_reg.parameters())
    print(f"  模型参数数量: {num_params:,}")

    # -------------------------------------------------------------------------
    # 测试 2: 分类模型
    # -------------------------------------------------------------------------
    print("\n[测试 2] 分类模型")

    # 创建分类模型
    model_cls = create_bimamba_model(
        vocab_size=50,
        d_model=64,
        n_layers=2,
        task_type="classification",  # 分类任务
        num_labels=1,  # 1 个输出（二分类）
    )

    num_params = sum(p.numel() for p in model_cls.parameters())
    print(f"  模型参数数量: {num_params:,}")

    # -------------------------------------------------------------------------
    # 测试 3: 前向传播（回归）
    # -------------------------------------------------------------------------
    print("\n[测试 3] 回归模型前向传播")

    # 创建测试数据
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, 50, (batch_size, seq_len))

    # 设置为评估模式（禁用 dropout 等）
    model_reg.eval()

    # 前向传播
    # 不计算梯度，加快速度
    with torch.no_grad():
        # 传入输入和标签，返回预测值和损失
        logits, loss = model_reg(
            input_ids=input_ids,
            labels=torch.randn(batch_size),  # 随机标签
        )

    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {logits.shape}")
    print(f"  损失值: {loss.item():.4f}")

    # 断言：输出应该是 (batch_size,)
    assert logits.shape == (batch_size,), (
        f"期望形状 ({batch_size},)，实际 {logits.shape}"
    )
    print("  ✓ 回归模型前向传播测试通过！")

    # -------------------------------------------------------------------------
    # 测试 4: 前向传播（分类）
    # -------------------------------------------------------------------------
    print("\n[测试 4] 分类模型前向传播")

    model_cls.eval()

    with torch.no_grad():
        # 分类标签需要是 0 或 1
        labels = torch.randint(0, 2, (batch_size,)).float()
        logits, loss = model_cls(input_ids=input_ids, labels=labels)

    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {logits.shape}")
    print(f"  损失值: {loss.item():.4f}")

    assert logits.shape == (batch_size,), (
        f"期望形状 ({batch_size},)，实际 {logits.shape}"
    )
    print("  ✓ 分类模型前向传播测试通过！")

    print("\n" + "-" * 50)
    print("所有模型测试通过！")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    """
    运行所有测试
    
    测试执行顺序：
    1. 组件测试（BiMambaBlock, BiMambaEncoder）
    2. 完整模型测试
    """
    print("\n" + "#" * 50)
    print("# Bi-Mamba 模型单元测试")
    print("#" * 50)

    try:
        # 测试组件
        test_model_components()

        # 测试完整模型
        test_model_creation()

        print("\n" + "=" * 50)
        print("✓ 所有测试完成！")
        print("=" * 50 + "\n")

    except AssertionError as e:
        # 测试失败
        print(f"\n✗ 测试失败: {e}")
        raise
    except Exception as e:
        # 其他错误
        print(f"\n✗ 发生错误: {e}")
        raise
