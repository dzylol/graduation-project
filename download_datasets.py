#!/usr/bin/env python3
"""
创建示例 MoleculeNet 数据集

基于已知的 MoleculeNet 基准数据集结构创建示例数据
用于测试 Bi-Mamba 模型
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ESOL 数据集 - 水溶解度预测（回归任务）
# 数据来源：Delaney's ESOL dataset
ESOL_DATA = """smiles,label
CC(C)C,0.49
CC(C)(C)c1ccc(cc1)C(O)=O,-0.76
c1ccc(cc1)S(=O)(=O)O,-3.18
CC(C)CC(C)C,2.69
c1ccc2c(c1)cccc2O,-0.12
CC(C)Oc1ccc(cc1)S(=O)(=O)N,-2.35
CC(C)Cc1ccc(cc1)C(C)C,1.85
c1ccc(cc1)C(=O)O,-0.57
c1ccc(cc1)C#N,0.14
CC(C)Cc1ccc(cc1)O,-0.14
c1ccc(cc1)CC#N,0.16
c1ccc(cc1)CCC#N,1.03
CC(C)CC(C)CC(C)C,2.87
c1ccc2c(c1)cc(cc2)S(=O)(=O)O,-2.58
c1ccc(cc1)CC(=O)O,0.34
CC(C)CCC(C)C,2.32
c1ccc2c(c1)cccc2C(=O)O,0.17
c1ccc(cc1)COC(=O)C,0.31
c1ccc(cc1)C(C)S(=O)(=O)C,0.08
c1ccc(cc1)OC,0.71
c1ccc(cc1)Cc2ccccc2,1.64
CC(C)Cc1ccc(cc1)CC(C)C,2.12
c1ccc(cc1)C(=O)OC,0.23
c1ccc(cc1)CCS(=O)(=O)C,-0.45
c1ccc(cc1)CCCC,1.83
c1ccc2c(c1)cccc2C(C)C,1.68
c1ccc(cc1)C(O)=O,-1.05
CC(C)CCc1ccc(cc1)C(C)C,1.96
c1ccc(cc1)OC(=O)C,0.43
c1ccc(cc1)Cc1ccccc1Cl,-0.53
c1ccc(cc1)CCOC,1.06
c1ccc(cc1)OCc2ccccc2,1.23
c1ccc(cc1)OCCO,0.08
CC(C)CC(C)CC(C)C,2.54
c1ccc(cc1)C(C)Oc2ccccc2,0.98
c1ccc(cc1)CC(=O)N,-0.54
c1ccc2c(c1)cccc2C#N,0.35
c1ccc(cc1)C(=O)c2ccccc2,0.86
CC(C)CCC(C)CC(C)C,2.61
c1ccc(cc1)Cc2cccc(C)c2,1.12
c1ccc(cc1)CCCCCCC,2.59
c1ccc(cc1)CCCc2ccccc2,1.34
c1ccc(cc1)CCCCc2ccccc2,1.67
c1ccc(cc1)CCCCCC,2.38
c1ccc(cc1)CCCCCCCc2ccccc2,2.23
CC(C)Cc1ccc(cc1)C(C)CC,1.86
c1ccc(cc1)Cc2ccc(cc2)C,1.54
c1ccc(cc1)CC(C)CC(C)C,2.18
c1ccc(cc1)CCCCCCCC,2.89
c1ccc(cc1)CCCCCCCCc2ccccc2,2.67
"""

# BBBP 数据集 - 血脑屏障渗透（分类任务）
# 1 = 可渗透, 0 = 不可渗透
BBBP_DATA = """smiles,label
CC(C)NCC(C)Oc1ccc2ccccc2c1,1
CC(C)Nc1ccc2ccccc2c1,1
CCNC(C)Cc1ccc2ccccc2c1,1
CC(C)NCCc1ccc2ccccc2c1,1
c1ccc2c(c1)cccc2CCNCC,1
CC(C)NCCOc1ccc2ccccc2c1,1
CC(C)Nc1ccc(cc1)C(C)C,0
CC(C)NCc1ccc(cc1)C(C)C,0
c1ccc(cc1)CNCC,0
c1ccc2c(c1)cccc2CNCC,0
CC(C)NCc1ccc(cc1)OC,0
c1ccc(cc1)NCC(C)C,0
c1ccc2c(c1)cccc2Nc3ccc4ccccc4c3,1
CC(C)NCCc1ccc2ccccc2c1,1
CC(C)Nc1ccc(cc1)CNCC,1
c1ccc(cc1)CN(C)CC,0
c1ccc(cc1)CCN(C)C,0
c1ccc2c(c1)cccc2CCNC,1
CC(C)NCCOc1ccc2ccccc2c1,1
CC(C)NCc1ccc2ccccc2c1,1
c1ccc(cc1)NC(C)C,0
c1ccc(cc1)NCC,0
CC(C)NCCc1ccc2ccccc2c1,1
CC(C)Nc1ccc2ccccc2c1,1
c1ccc(cc1)CNCC,0
c1ccc2c(c1)cccc2CCNCC,1
CC(C)NCc1ccc(cc1)CNCC,1
c1ccc(cc1)NCC(C)C,0
CC(C)Nc1ccc(cc1)OC,0
c1ccc2c(c1)cccc2NC,1
CC(C)NCCOc1ccc2ccccc2c1,1
c1ccc2c(c1)cccc2CCNC,1
c1ccc(cc1)Nc2ccc3ccccc3c2,1
CC(C)NCCc1ccc2ccccc2c1,1
c1ccc(cc1)NC(C)C,0
c1ccc(cc1)NCC,0
c1ccc2c(c1)cccc2Nc3ccccc3,1
CC(C)Nc1ccc(cc1)C(C)C,0
CC(C)NCc1ccc(cc1)C(C)C,0
c1ccc(cc1)CNCC,0
c1ccc2c(c1)cccc2CCNCC,1
CC(C)NCc1ccc(cc1)OC,0
CC(C)NCCc1ccc2ccccc2c1,1
c1ccc(cc1)NCC(C)C,0
c1ccc(cc1)CN(C)CC,0
c1ccc2c(c1)cccc2CCNC,1
CC(C)NCCOc1ccc2ccccc2c1,1
c1ccc(cc1)CN(C)CC,0
c1ccc(cc1)NC(C)C,0
c1ccc(cc1)NCC,0
c1ccc2c(c1)cccc2Nc3ccc4ccccc4c3,1
c1ccc(cc1)CNCC,0
"""

# ClinTox 数据集 - 临床毒性预测（分类任务）
# CT_TOX: 1 = 临床毒性, 0 = 无毒性
CLINTOX_DATA = """smiles,label
CC(C)Cc1ccc(cc1)C(C)C,0
c1ccc(cc1)C(C)C,0
c1ccc2c(c1)cccc2C(C)C,0
CC(C)CC(C)C,0
c1ccc(cc1)CC(C)C,0
c1ccc2c(c1)cccc2O,0
c1ccc(cc1)O,0
CC(C)Oc1ccc(cc1)C(C)C,0
c1ccc(cc1)CC,0
CC(C)Cc1ccc(cc1)O,0
c1ccc(cc1)CCC,0
c1ccc2c(c1)cccc2CC,0
c1ccc(cc1)CCCC,0
c1ccc(cc1)Cc1ccccc1,0
CC(C)CCc1ccc(cc1)CC,0
c1ccc(cc1)COC,0
CC(C)CC,0
c1ccc(cc1)C#N,0
c1ccc(cc1)C=O,0
c1ccc(cc1)C(=O)O,0
CC(C)NCc1ccc(cc1)CC,1
CC(C)NCc1ccc(cc1)C(C)C,1
c1ccc(cc1)NC(C)C,1
CC(C)Nc1ccc(cc1)CC,1
c1ccc(cc1)NC(C)CC,1
c1ccc(cc1)NC(C)C(C)C,1
CC(C)NCc1ccc(cc1)OC,1
c1ccc(cc1)NCC,1
c1ccc2c(c1)cccc2NC,1
c1ccc(cc1)Nc1ccccc1,1
c1ccc(cc1)CNc2ccccc2,1
c1ccc(cc1)NCc2ccccc2,1
CC(C)Nc1ccc(cc1)Cc2ccccc2,1
c1ccc(cc1)Nc1ccc2ccccc2c1,1
c1ccc(cc1)NCc1ccccc1C,1
c1ccc(cc1)NCCc1ccccc1,1
c1ccc(cc1)NCC(C)C,1
CC(C)NCc1ccc2ccccc2c1,1
c1ccc(cc1)NC(C)OC,1
c1ccc2c(c1)cccc2NCc1ccccc1,1
c1ccc(cc1)Nc1ccc2ccccc2c1,1
c1ccc(cc1)NCc1ccc2ccccc2c1,1
c1ccc(cc1)NC(C)Cc1ccccc1,1
c1ccc(cc1)NCCc1ccccc1C,1
c1ccc(cc1)Nc1ccccc1C(C)C,1
c1ccc(cc1)NC(C)Cc1ccccc1C,1
c1ccc(cc1)NCC(C)C(C)C,1
c1ccc(cc1)NC(C)CC(C)C,1
c1ccc(cc1)Nc1ccccc1CC,1
c1ccc(cc1)NCc1ccccc1CC,1
"""


def create_dataset(dataset_name: str, data_str: str, data_dir: str):
    """创建并分割数据集"""
    from io import StringIO

    print(f"\n{'=' * 50}")
    print(f"创建数据集: {dataset_name}")
    print(f"{'=' * 50}")

    # 解析数据
    df = pd.read_csv(StringIO(data_str))
    print(f"样本数: {len(df)}")

    # 分割数据
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 创建目录
    save_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # 保存
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    # 保存元信息
    import json

    task_type = "regression" if dataset_name == "ESOL" else "classification"
    meta = {
        "name": dataset_name,
        "task_type": task_type,
        "description": "水溶解度预测" if dataset_name == "ESOL" else "临床性质预测",
        "num_train": len(train_df),
        "num_val": len(val_df),
        "num_test": len(test_df),
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✓ 保存完成!")
    print(f"  训练集: {len(train_df)} 样本 -> {save_dir}/train.csv")
    print(f"  验证集: {len(val_df)} 样本 -> {save_dir}/val.csv")
    print(f"  测试集: {len(test_df)} 样本 -> {save_dir}/test.csv")


def main():
    data_dir = "./data"

    print("\n" + "=" * 60)
    print("创建 MoleculeNet 示例数据集")
    print("=" * 60)

    # 创建目录
    os.makedirs(data_dir, exist_ok=True)

    # 创建数据集
    create_dataset("ESOL", ESOL_DATA, data_dir)
    create_dataset("BBBP", BBBP_DATA, data_dir)
    create_dataset("ClinTox", CLINTOX_DATA, data_dir)

    print("\n" + "=" * 60)
    print("✓ 所有数据集创建完成!")
    print("=" * 60)
    print(f"\n数据集位置: {os.path.abspath(data_dir)}")
    print("\n使用方法:")
    print("  python train.py --dataset ESOL --epochs 10 --batch_size 32")
    print("  python train.py --dataset BBBP --task_type classification --epochs 10")


if __name__ == "__main__":
    main()
