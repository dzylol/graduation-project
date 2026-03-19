#!/usr/bin/env python3
"""
数据处理模块单元测试

本脚本用于测试分子数据处理相关功能。

测试内容：
1. MoleculeTokenizer - SMILES 分词器
2. MoleculeDataset - 分子数据集
3. create_data_loaders - 数据加载器创建

使用方法：
```bash
python tests/test_data.py
```

作者: Bi-Mamba-Chem Team
"""

import torch  # PyTorch 深度学习框架
import sys  # 系统模块
import os  # 文件路径操作
import tempfile  # 临时文件
import csv  # CSV 文件处理

# 将 src 目录添加到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 导入待测试的模块
from data.molecule_dataset import (
    MoleculeDataset,
    MoleculeTokenizer,
    create_data_loaders,
)


def test_tokenizer():
    """
    测试 SMILES 分词器

    测试分词器的编码和解码功能
    """
    print("\n" + "=" * 50)
    print("测试 SMILES 分词器")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 创建分词器
    # -------------------------------------------------------------------------
    print("\n[步骤 1] 创建分词器")
    tokenizer = MoleculeTokenizer()
    print(f"  词汇表大小: {tokenizer.vocab_size}")
    print(f"  特殊 token: <pad>={tokenizer.vocab['<pad>']}")

    # -------------------------------------------------------------------------
    # 测试编码
    # -------------------------------------------------------------------------
    print("\n[步骤 2] 测试编码功能")

    smiles = "CCO"  # 乙醇的 SMILES
    tokens = tokenizer.encode(smiles, max_length=10)
    print(f"  原始 SMILES: {smiles}")
    print(f"  编码结果: {tokens}")

    # -------------------------------------------------------------------------
    # 测试解码
    # -------------------------------------------------------------------------
    print("\n[步骤 3] 测试解码功能")

    decoded = tokenizer.decode(tokens)
    print(f"  解码结果: {decoded}")

    # 验证编码-解码是否一致
    assert decoded == smiles, f"编码-解码不一致: {smiles} vs {decoded}"
    print("  ✓ 编码-解码一致性验证通过！")

    # -------------------------------------------------------------------------
    # 测试特殊 token
    # -------------------------------------------------------------------------
    print("\n[步骤 4] 验证特殊 token")

    assert tokenizer.vocab["<pad>"] == 0, "<pad> 应该是 ID=0"
    assert tokenizer.vocab["<pad>"] < tokenizer.vocab_size, "<pad> 应该在词汇表范围内"
    assert "C" in tokenizer.vocab, "元素符号 C 应该在词汇表中"
    print("  ✓ 特殊 token 验证通过！")

    print("\n" + "-" * 50)
    print("✓ 分词器测试通过！")


def test_dataset():
    """
    测试分子数据集类

    测试数据集的创建和索引功能
    """
    print("\n" + "=" * 50)
    print("测试分子数据集")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 创建临时测试数据
    # -------------------------------------------------------------------------
    print("\n[步骤 1] 创建临时测试数据")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "label"])
        writer.writerow(["CCO", 1.0])
        writer.writerow(["CC(C)O", 2.0])
        writer.writerow(["c1ccccc1O", 3.0])
        temp_file = f.name

    print(f"  创建临时文件: {temp_file}")

    try:
        # -------------------------------------------------------------------------
        # 创建数据集
        # -------------------------------------------------------------------------
        print("\n[步骤 2] 创建数据集")

        dataset = MoleculeDataset(
            data_path=temp_file, task_type="regression", max_length=50
        )

        print(f"  数据集大小: {len(dataset)}")
        assert len(dataset) == 3

        # -------------------------------------------------------------------------
        # 测试获取单个样本
        # -------------------------------------------------------------------------
        print("\n[步骤 3] 获取单个样本")

        input_ids, labels = dataset[0]
        print(f"  样本索引: 0")
        print(f"  输入形状: {input_ids.shape}")
        print(f"  标签: {labels}")

        assert input_ids.shape[0] == 50
        assert labels.shape[0] == 1
        print("  ✓ 样本形状验证通过！")

        # -------------------------------------------------------------------------
        # 验证词汇表
        # -------------------------------------------------------------------------
        print("\n[步骤 4] 验证词汇表")

        vocab_size = dataset.get_vocab_size()
        pad_id = dataset.get_pad_token_id()
        print(f"  词汇表大小: {vocab_size}")
        print(f"  填充 token ID: {pad_id}")

        assert vocab_size > 0
        print("  ✓ 词汇表验证通过！")

    finally:
        # -------------------------------------------------------------------------
        # 清理临时文件
        # -------------------------------------------------------------------------
        os.unlink(temp_file)
        print(f"\n  清理临时文件: {temp_file}")

    print("\n" + "-" * 50)
    print("✓ 数据集测试通过！")


def test_data_loaders():
    """
    测试数据加载器

    测试 create_data_loaders 函数创建的数据加载器
    """
    print("\n" + "=" * 50)
    print("测试数据加载器")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 创建临时测试文件
    # -------------------------------------------------------------------------
    print("\n[步骤 1] 创建临时测试文件")

    temp_dir = tempfile.mkdtemp()
    train_file = os.path.join(temp_dir, "train.csv")
    val_file = os.path.join(temp_dir, "val.csv")
    test_file = os.path.join(temp_dir, "test.csv")

    # 训练数据
    with open(train_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "label"])
        for i in range(10):
            writer.writerow(["CCO", float(i)])
    print(f"  创建训练文件: {train_file}")

    # 验证数据
    with open(val_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "label"])
        for i in range(5):
            writer.writerow(["CC(C)O", float(i + 10)])
    print(f"  创建验证文件: {val_file}")

    # 测试数据
    with open(test_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "label"])
        for i in range(3):
            writer.writerow(["c1ccccc1O", float(i + 20)])
    print(f"  创建测试文件: {test_file}")

    try:
        # -------------------------------------------------------------------------
        # 创建数据加载器
        # -------------------------------------------------------------------------
        print("\n[步骤 2] 创建数据加载器")

        train_loader, val_loader, test_loader = create_data_loaders(
            train_path=train_file,
            val_path=val_file,
            test_path=test_file,
            batch_size=4,
            task_type="regression",
            max_length=50,
            num_workers=0,
        )

        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print(f"  测试批次数: {len(test_loader)}")

        # -------------------------------------------------------------------------
        # 测试遍历训练数据
        # -------------------------------------------------------------------------
        print("\n[步骤 3] 遍历训练数据")

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            print(f"  Batch {batch_idx}:")
            print(f"    输入形状: {input_ids.shape}")
            print(f"    标签形状: {labels.shape}")

            assert input_ids.shape[0] == 4
            assert input_ids.shape[1] == 50
            assert labels.shape[0] == 4
            break

        print("  ✓ 数据加载器验证通过！")

    finally:
        # -------------------------------------------------------------------------
        # 清理临时文件
        # -------------------------------------------------------------------------
        os.unlink(train_file)
        os.unlink(val_file)
        os.unlink(test_file)
        os.rmdir(temp_dir)
        print(f"\n  清理临时文件")

    print("\n" + "-" * 50)
    print("✓ 数据加载器测试通过！")


if __name__ == "__main__":
    """
    运行所有测试
    
    测试执行顺序：
    1. 分词器测试
    2. 数据集测试
    3. 数据加载器测试
    """
    print("\n" + "#" * 50)
    print("# 数据处理模块单元测试")
    print("#" * 50)

    try:
        test_tokenizer()
        test_dataset()
        test_data_loaders()

        print("\n" + "=" * 50)
        print("✓ 所有测试完成！")
        print("=" * 50 + "\n")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        raise
