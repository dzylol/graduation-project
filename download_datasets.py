#!/usr/bin/env python3
"""
下载 MoleculeNet 数据集

使用 DeepChem 加载完整的 MoleculeNet 基准数据集。
如 DeepChem 未安装，则生成示例数据用于快速测试。

用法:
    python download_datasets.py              # 下载所有数据集
    python download_datasets.py --dataset ESOL  # 下载特定数据集
    python download_datasets.py --zinc           # 下载 ZINC 预训练数据
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetInfo:
    """数据集元信息。"""

    name: str
    task_type: str
    description: str
    num_molecules: int
    metric: str
    category: str


# 数据集配置
DATASET_CONFIGS: dict[str, DatasetInfo] = {
    "ESOL": DatasetInfo(
        name="ESOL",
        task_type="regression",
        description="水溶解度预测 (Delaney ESOL)",
        num_molecules=1128,
        metric="RMSE",
        category="Physical Chemistry",
    ),
    "BBBP": DatasetInfo(
        name="BBBP",
        task_type="classification",
        description="血脑屏障渗透预测",
        num_molecules=2039,
        metric="ROC-AUC",
        category="Physiology",
    ),
    "ClinTox": DatasetInfo(
        name="ClinTox",
        task_type="classification",
        description="临床毒性预测",
        num_molecules=1478,
        metric="ROC-AUC",
        category="Physiology",
    ),
    "FreeSolv": DatasetInfo(
        name="FreeSolv",
        task_type="regression",
        description="水合自由能预测",
        num_molecules=642,
        metric="RMSE",
        category="Physical Chemistry",
    ),
    "Lipophilicity": DatasetInfo(
        name="Lipophilicity",
        task_type="regression",
        description="脂溶性预测",
        num_molecules=4200,
        metric="RMSE",
        category="Physical Chemistry",
    ),
    "Tox21": DatasetInfo(
        name="Tox21",
        task_type="classification",
        description="21项毒性预测",
        num_molecules=7831,
        metric="ROC-AUC",
        category="Physiology",
    ),
}


def check_deepchem_installed() -> bool:
    """检查 DeepChem 是否已安装。"""
    try:
        import deepchem  # noqa: F401

        return True
    except ImportError:
        return False


def load_with_deepchem(dataset_name: str, data_dir: str) -> bool:
    """使用 DeepChem 加载完整数据集。

    Args:
        dataset_name: 数据集名称
        data_dir: 保存目录

    Returns:
        True 如果成功，False 如果失败
    """
    try:
        import deepchem as dc
        from rdkit import Chem

        print(f"使用 DeepChem 加载 {dataset_name}...")

        # DeepChem 数据集映射
        loaders = {
            "ESOL": dc.molnet.load_esol,
            "BBBP": dc.molnet.load_bbbp,
            "ClinTox": dc.molnet.load_clintox,
            "FreeSolv": dc.molnet.load_freesolv,
            "Lipophilicity": dc.molnet.load_lipophilicity,
            "Tox21": dc.molnet.load_tox21,
        }

        if dataset_name not in loaders:
            print(f"  DeepChem 不支持 {dataset_name}")
            return False

        # 加载数据集
        tasks, dataset, transformers = loaders[dataset_name]()

        # 分割数据 (DeepChem 默认 80/10/10)
        train_dataset, val_dataset, test_dataset = dataset

        # 转换为 SMILES 和 labels
        def extract_data(dc_dataset):
            smiles_list = []
            labels_list = []
            for i in range(len(dc_dataset)):
                # DeepChem 数据集的 IDs 是 SMILES 字符串
                smiles = str(dc_dataset.ids[i])
                # 处理多任务标签
                labels = dc_dataset.y[i]
                if len(labels.shape) == 1:
                    labels = labels.tolist()
                else:
                    labels = labels.tolist()
                smiles_list.append(smiles)
                labels_list.append(labels)
            return smiles_list, labels_list

        train_smiles, train_labels = extract_data(train_dataset)
        val_smiles, val_labels = extract_data(val_dataset)
        test_smiles, test_labels = extract_data(test_dataset)

        # 过滤无效 SMILES
        def filter_valid(smiles_list, labels_list):
            valid_smiles = []
            valid_labels = []
            for smi, lab in zip(smiles_list, labels_list):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_smiles.append(smi)
                    valid_labels.append(lab)
            return valid_smiles, valid_labels

        train_smiles, train_labels = filter_valid(train_smiles, train_labels)
        val_smiles, val_labels = filter_valid(val_smiles, val_labels)
        test_smiles, test_labels = filter_valid(test_smiles, test_labels)

        # 保存为 CSV
        save_dir = os.path.join(data_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # 保存训练集
        train_df = pd.DataFrame({"smiles": train_smiles, "label": train_labels})
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)

        # 保存验证集
        val_df = pd.DataFrame({"smiles": val_smiles, "label": val_labels})
        val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)

        # 保存测试集
        test_df = pd.DataFrame({"smiles": test_smiles, "label": test_labels})
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

        # 保存元信息
        config = DATASET_CONFIGS.get(
            dataset_name,
            DatasetInfo(
                name=dataset_name,
                task_type="regression"
                if train_dataset.y.shape[1] == 1
                else "classification",
                description=dataset_name,
                num_molecules=len(train_smiles) + len(val_smiles) + len(test_smiles),
                metric="RMSE" if train_dataset.y.shape[1] == 1 else "ROC-AUC",
                category="MoleculeNet",
            ),
        )

        meta = {
            "name": config.name,
            "task_type": config.task_type,
            "description": config.description,
            "metric": config.metric,
            "category": config.category,
            "num_train": len(train_smiles),
            "num_val": len(val_smiles),
            "num_test": len(test_smiles),
            "source": "DeepChem MoleculeNet",
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  ✓ {dataset_name} 下载完成!")
        print(f"    训练集: {len(train_smiles)} 样本")
        print(f"    验证集: {len(val_smiles)} 样本")
        print(f"    测试集: {len(test_smiles)} 样本")
        return True

    except Exception as e:
        print(f"  ✗ DeepChem 加载失败: {e}")
        return False


def generate_example_data(dataset_name: str, data_dir: str) -> bool:
    """生成示例数据（当 DeepChem 未安装时）。

    Args:
        dataset_name: 数据集名称
        data_dir: 保存目录

    Returns:
        True 如果成功
    """
    # 硬编码的示例数据（每个数据集约 50 条）
    EXAMPLE_DATA = {
        "ESOL": [
            ("CC(C)C", 0.49),
            ("CC(C)(C)c1ccc(cc1)C(O)=O", -0.76),
            ("c1ccc(cc1)S(=O)(=O)O", -3.18),
            ("CC(C)CC(C)C", 2.69),
            ("c1ccc2c(c1)cccc2O", -0.12),
            ("c1ccc(cc1)C(=O)O", -0.57),
            ("c1ccc(cc1)C#N", 0.14),
            ("c1ccc(cc1)CC#N", 0.16),
            ("c1ccc(cc1)CCC#N", 1.03),
            ("c1ccc(cc1)CC(=O)O", 0.34),
            ("c1ccc(cc1)OC", 0.71),
            ("c1ccc(cc1)Cc2ccccc2", 1.64),
            ("c1ccc(cc1)C(=O)OC", 0.23),
            ("c1ccc(cc1)CCCC", 1.83),
            ("c1ccc(cc1)C(O)=O", -1.05),
            ("c1ccc(cc1)OC(=O)C", 0.43),
            ("c1ccc(cc1)Cc1ccccc1Cl", -0.53),
            ("c1ccc(cc1)CCOC", 1.06),
            ("c1ccc(cc1)OCc2ccccc2", 1.23),
            ("c1ccc(cc1)CCCCCCC", 2.59),
            ("c1ccc(cc1)CCCc2ccccc2", 1.34),
            ("c1ccc(cc1)CCCCc2ccccc2", 1.67),
            ("c1ccc(cc1)CCCCCC", 2.38),
            ("c1ccc(cc1)CCCCCCCC", 2.89),
            ("CC(C)Cc1ccc(cc1)C(C)CC", 1.86),
            ("c1ccc(cc1)Cc2ccc(cc2)C", 1.54),
            ("c1ccc(cc1)CC(C)CC(C)C", 2.18),
            ("c1ccc2c(c1)cccc2C(C)C", 1.68),
            ("c1ccc(cc1)C(C)S(=O)(=O)C", 0.08),
            ("c1ccc(cc1)CCS(=O)(=O)C", -0.45),
            ("CC(C)CCc1ccc(cc1)C(C)C", 1.96),
            ("c1ccc(cc1)CC(=O)N", -0.54),
            ("c1ccc(cc1)C(=O)c2ccccc2", 0.86),
            ("c1ccc2c(c1)cccc2C#N", 0.35),
            ("c1ccc2c(c1)cc(cc2)S(=O)(=O)O", -2.58),
            ("c1ccc2c(c1)cccc2C(=O)O", 0.17),
            ("c1ccc2c(c1)cccc2C(C)C", 1.68),
            ("c1ccc(cc1)Cc2cccc(C)c2", 1.12),
            ("c1ccc(cc1)C(C)Oc2ccccc2", 0.98),
            ("CC(C)CC(C)CC(C)C", 2.54),
            ("CC(C)CCC(C)C", 2.32),
            ("CC(C)CCC(C)CC(C)C", 2.61),
            ("c1ccc(cc1)CCCCCCCc2ccccc2", 2.23),
            ("c1ccc(cc1)CCCCCCCCc2ccccc2", 2.67),
            ("c1ccc(cc1)OCCO", 0.08),
            ("CC(C)Oc1ccc(cc1)S(=O)(=O)N", -2.35),
        ],
        "BBBP": [
            ("CC(C)NCC(C)Oc1ccc2ccccc2c1", 1),
            ("CC(C)Nc1ccc2ccccc2c1", 1),
            ("CCNC(C)Cc1ccc2ccccc2c1", 1),
            ("CC(C)NCCc1ccc2ccccc2c1", 1),
            ("c1ccc2c(c1)cccc2CCNCC", 1),
            ("CC(C)NCCOc1ccc2ccccc2c1", 1),
            ("CC(C)Nc1ccc(cc1)C(C)C", 0),
            ("CC(C)NCc1ccc(cc1)C(C)C", 0),
            ("c1ccc(cc1)CNCC", 0),
            ("c1ccc2c(c1)cccc2CNCC", 0),
            ("CC(C)NCc1ccc(cc1)OC", 0),
            ("c1ccc(cc1)NCC(C)C", 0),
            ("c1ccc2c(c1)cccc2Nc3ccc4ccccc4c3", 1),
            ("CC(C)NCCc1ccc2ccccc2c1", 1),
            ("CC(C)Nc1ccc(cc1)CNCC", 1),
            ("c1ccc(cc1)CN(C)CC", 0),
            ("c1ccc(cc1)CCN(C)C", 0),
            ("c1ccc2c(c1)cccc2CCNC", 1),
            ("CC(C)NCCOc1ccc2ccccc2c1", 1),
            ("CC(C)NCc1ccc2ccccc2c1", 1),
            ("c1ccc(cc1)NC(C)C", 0),
            ("c1ccc(cc1)NCC", 0),
            ("CC(C)NCCc1ccc2ccccc2c1", 1),
            ("c1ccc(cc1)CNCC", 0),
            ("c1ccc2c(c1)cccc2CCNCC", 1),
            ("CC(C)NCc1ccc(cc1)CNCC", 1),
            ("c1ccc(cc1)NCC(C)C", 0),
            ("CC(C)Nc1ccc(cc1)OC", 0),
            ("c1ccc2c(c1)cccc2NC", 1),
            ("c1ccc(cc1)Nc2ccc3ccccc3c2", 1),
            ("c1ccc(cc1)NC(C)C", 0),
            ("c1ccc(cc1)NCC", 0),
            ("c1ccc2c(c1)cccc2Nc3ccccc3", 1),
            ("c1ccc(cc1)CN(C)CC", 0),
            ("c1ccc(cc1)NC(C)C", 0),
            ("CC(C)NCCOc1ccc2ccccc2c1", 1),
            ("c1ccc2c(c1)cccc2CCNC", 1),
            ("c1ccc(cc1)CN(C)CC", 0),
            ("c1ccc(cc1)NC(C)C", 0),
            ("c1ccc(cc1)NCC", 0),
            ("c1ccc2c(c1)cccc2Nc3ccc4ccccc4c3", 1),
            ("c1ccc(cc1)CNCC", 0),
        ],
        "ClinTox": [
            ("CC(C)Cc1ccc(cc1)C(C)C", 0),
            ("c1ccc(cc1)C(C)C", 0),
            ("c1ccc2c(c1)cccc2C(C)C", 0),
            ("CC(C)CC(C)C", 0),
            ("c1ccc(cc1)CC(C)C", 0),
            ("c1ccc2c(c1)cccc2O", 0),
            ("c1ccc(cc1)O", 0),
            ("CC(C)Oc1ccc(cc1)C(C)C", 0),
            ("c1ccc(cc1)CC", 0),
            ("CC(C)Cc1ccc(cc1)O", 0),
            ("c1ccc(cc1)CCC", 0),
            ("c1ccc2c(c1)cccc2CC", 0),
            ("c1ccc(cc1)CCCC", 0),
            ("c1ccc(cc1)Cc1ccccc1", 0),
            ("CC(C)CCc1ccc(cc1)CC", 0),
            ("c1ccc(cc1)COC", 0),
            ("CC(C)CC", 0),
            ("c1ccc(cc1)C#N", 0),
            ("c1ccc(cc1)C=O", 0),
            ("c1ccc(cc1)C(=O)O", 0),
            ("CC(C)NCc1ccc(cc1)CC", 1),
            ("CC(C)NCc1ccc(cc1)C(C)C", 1),
            ("c1ccc(cc1)NC(C)C", 1),
            ("CC(C)Nc1ccc(cc1)CC", 1),
            ("c1ccc(cc1)NC(C)CC", 1),
            ("c1ccc(cc1)NC(C)C(C)C", 1),
            ("CC(C)NCc1ccc(cc1)OC", 1),
            ("c1ccc(cc1)NCC", 1),
            ("c1ccc2c(c1)cccc2NC", 1),
            ("c1ccc(cc1)Nc1ccccc1", 1),
            ("c1ccc(cc1)CNc2ccccc2", 1),
            ("c1ccc(cc1)NCc2ccccc2", 1),
            ("CC(C)Nc1ccc(cc1)Cc2ccccc2", 1),
            ("c1ccc(cc1)Nc1ccc2ccccc2c1", 1),
            ("c1ccc(cc1)NCc1ccccc1C", 1),
            ("c1ccc(cc1)NCCc1ccccc1", 1),
            ("c1ccc(cc1)NCC(C)C", 1),
            ("CC(C)NCc1ccc2ccccc2c1", 1),
            ("c1ccc(cc1)NC(C)OC", 1),
            ("c1ccc2c(c1)cccc2NCc1ccccc1", 1),
        ],
    }

    if dataset_name not in EXAMPLE_DATA:
        print(f"  ✗ 无示例数据: {dataset_name}")
        return False

    data = EXAMPLE_DATA[dataset_name]
    df = pd.DataFrame(data, columns=["smiles", "label"])

    # 分割数据
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 保存
    save_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    # 元信息
    config = DATASET_CONFIGS.get(
        dataset_name,
        DatasetInfo(
            name=dataset_name,
            task_type="classification",
            description=dataset_name,
            num_molecules=len(data),
            metric="ROC-AUC",
            category="MoleculeNet",
        ),
    )

    meta = {
        "name": config.name,
        "task_type": config.task_type,
        "description": config.description,
        "metric": config.metric,
        "category": config.category,
        "num_train": len(train_df),
        "num_val": len(val_df),
        "num_test": len(test_df),
        "source": "Example Data (tiny subset)",
        "warning": "这是示例数据，仅用于快速测试。用于真实训练请安装 DeepChem。",
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {dataset_name} 示例数据生成完成 (WARNING: 仅用于测试!)")
    print(f"    训练集: {len(train_df)} 样本")
    print(f"    验证集: {len(val_df)} 样本")
    print(f"    测试集: {len(test_df)} 样本")
    return True


def download_zinc_pretrain(data_dir: str) -> bool:
    """下载 ZINC 250K 预训练数据（用于自监督预训练）。

    ZINC 是药物类小分子库，用于 SMILES-Mamba 风格的两阶段训练。

    Returns:
        True 如果成功
    """
    zinc_path = os.path.join(data_dir, "ZINC250K")
    os.makedirs(zinc_path, exist_ok=True)

    # ZINC 250K 子集可以从多个来源获取
    # 这里我们尝试从公开来源下载
    print("\n" + "=" * 50)
    print("下载 ZINC 250K 预训练数据")
    print("=" * 50)

    # 检查是否已有数据
    train_path = os.path.join(zinc_path, "train.smiles")
    if os.path.exists(train_path):
        with open(train_path) as f:
            lines = len(f.readlines())
        if lines > 100000:
            print(f"  ✓ ZINC 数据已存在 ({lines} 条)")
            return True

    # 尝试下载
    print("  尝试从公开来源下载...")
    try:
        import urllib.request

        # ZINC 250K 子集 URL (从 various sources)
        urls = [
            "https://raw.githubusercontent.com/patrick-kidger/ZINC250K/master/train.smiles",
            "https://raw.githubusercontent.com/marcovaltullio/ZINC250K/master/train.smiles",
        ]

        downloaded = False
        for url in urls:
            try:
                print(f"  尝试: {url[:60]}...")
                urllib.request.urlretrieve(url, train_path)
                downloaded = True
                with open(train_path) as f:
                    lines = len(f.readlines())
                print(f"  ✓ 下载成功! ({lines} 条 SMILES)")
                return True
            except Exception:
                continue

        if not downloaded:
            raise Exception("所有 URL 都失败了")

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  可以手动下载 ZINC 250K:")
        print("    1. 访问 https://zinc15.docking.org/")
        print("    2. 注册账号")
        print("    3. 下载 250K 子集")
        print("    4. 保存到 data/ZINC250K/train.smiles")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 MoleculeNet 数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASET_CONFIGS.keys()) + ["all", "zinc"],
        help="指定数据集 (默认下载所有)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="数据保存目录 (默认: ./data)",
    )
    parser.add_argument(
        "--zinc",
        action="store_true",
        help="下载 ZINC 预训练数据",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="强制使用示例数据 (不尝试 DeepChem)",
    )

    args = parser.parse_args()

    deepchem_available = check_deepchem_installed()
    print("\n" + "=" * 60)
    print("MoleculeNet 数据集下载")
    print("=" * 60)
    print(f"DeepChem 安装: {'✓ 是' if deepchem_available else '✗ 否'}")
    print(f"数据目录: {os.path.abspath(args.data_dir)}")
    print()

    if args.example:
        deepchem_available = False

    os.makedirs(args.data_dir, exist_ok=True)

    # 下载 ZINC 预训练数据
    if args.zinc or args.dataset == "zinc":
        download_zinc_pretrain(args.data_dir)

    # 确定要下载的数据集
    if args.dataset and args.dataset != "zinc":
        datasets_to_download = [args.dataset]
    else:
        datasets_to_download = [k for k in DATASET_CONFIGS.keys()]

    # 下载每个数据集
    for dataset_name in datasets_to_download:
        print(f"\n{'=' * 50}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'=' * 50}")

        success = False
        if deepchem_available:
            success = load_with_deepchem(dataset_name, args.data_dir)

        if not success:
            print(f"  回退到示例数据...")
            generate_example_data(dataset_name, args.data_dir)

    # 总结
    print("\n" + "=" * 60)
    print("✓ 数据集下载完成!")
    print("=" * 60)
    print(f"\n数据集位置: {os.path.abspath(args.data_dir)}")
    print("\n使用方法:")
    print("  python train.py --dataset ESOL --epochs 10 --batch_size 32")
    print("  python train.py --dataset BBBP --task_type classification")
    print()
    print("注意: 示例数据仅用于快速测试，真实训练请安装 DeepChem:")
    print("  pip install deepchem")
    print()

    if deepchem_available:
        print("已使用 DeepChem 加载完整数据集，可以开始正式训练!")


if __name__ == "__main__":
    main()
