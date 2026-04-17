"""
长序列分组分析脚本

按 SMILES 长度将分子分为三组：Short（< 50 tokens）、Medium（50–100 tokens）、Long（> 100 tokens），
分别计算每组的 RMSE。
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.bimamba_with_mamba_ssm import BiMambaForPropertyPrediction
from src.data.tokenizer import MoleculeTokenizer


def analyze_length_groups(
    checkpoint_path: str,
    test_csv: str,
    smiles_col: str,
    label_col: str,
    output_path: str,
    device: str = "cuda",
) -> dict:
    """
    分析不同长度序列的预测性能。

    Args:
        checkpoint_path: 模型检查点路径
        test_csv: 测试集 CSV 路径
        smiles_col: SMILES 列名
        label_col: 标签列名
        output_path: 输出 JSON 路径
        device: 设备

    Returns:
        分析结果字典
    """
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict", checkpoint)

    model = BiMambaForPropertyPrediction(
        vocab_size=45,
        d_model=256,
        d_mamba=256,
        n_layers=4,
        pooling="mean",
        num_labels=1,
        dropout=0.1,
    )
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    print(f"加载测试数据: {test_csv}")
    df = pd.read_csv(test_csv)
    smiles_list = df[smiles_col].tolist()
    labels = df[label_col].values

    tokenizer = MoleculeTokenizer()
    tokens_list = []
    lengths = []
    for s in smiles_list:
        tokens = tokenizer.encode(s, max_length=512)
        tokens_list.append(tokens)
        length = sum(1 for t in tokens if t != 0)
        lengths.append(length)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), 32):
            batch_tokens = tokens_list[i : i + 32]
            batch_len = len(batch_tokens)
            max_len = max(len(t) for t in batch_tokens)

            input_ids = torch.zeros(batch_len, max_len, dtype=torch.long, device=device)
            attention_mask = torch.zeros(batch_len, max_len, dtype=torch.bool, device=device)

            for j, tokens in enumerate(batch_tokens):
                input_ids[j, : len(tokens)] = torch.tensor(tokens, device=device)
                attention_mask[j, : len(tokens)] = True

            outputs = model(input_ids, attention_mask)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            preds = logits.squeeze(-1).cpu().numpy()
            predictions.extend(preds if isinstance(preds, list) else preds.tolist())

    predictions = np.array(predictions)

    short_mask = np.array(lengths) < 50
    medium_mask = (np.array(lengths) >= 50) & (np.array(lengths) <= 100)
    long_mask = np.array(lengths) > 100

    def calc_rmse(y_true, y_pred):
        if len(y_true) == 0:
            return None
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def get_group_lengths(lengths_arr, mask):
        indices = [i for i in range(len(lengths_arr)) if mask[i]]
        if not indices:
            return None, None
        group_lens = [lengths_arr[i] for i in indices]
        return int(min(group_lens)), int(max(group_lens))

    results = {
        "checkpoint": checkpoint_path,
        "test_csv": test_csv,
        "total_samples": len(df),
        "groups": {
            "short": {
                "name": "Short (< 50 tokens)",
                "count": int(short_mask.sum()),
                "rmse": calc_rmse(labels[short_mask], predictions[short_mask]),
                "lengths": dict(zip(["min", "max"], get_group_lengths(lengths, short_mask)))
                if short_mask.sum() > 0
                else None,
            },
            "medium": {
                "name": "Medium (50-100 tokens)",
                "count": int(medium_mask.sum()),
                "rmse": calc_rmse(labels[medium_mask], predictions[medium_mask]),
                "lengths": dict(zip(["min", "max"], get_group_lengths(lengths, medium_mask)))
                if medium_mask.sum() > 0
                else None,
            },
            "long": {
                "name": "Long (> 100 tokens)",
                "count": int(long_mask.sum()),
                "rmse": calc_rmse(labels[long_mask], predictions[long_mask]),
                "lengths": dict(zip(["min", "max"], get_group_lengths(lengths, long_mask)))
                if long_mask.sum() > 0
                else None,
            },
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {output_path}")
    print(f"\n=== 长序列分组分析结果 ===")
    print(f"总样本数: {results['total_samples']}")
    for group_name, group_data in results["groups"].items():
        rmse_str = f"{group_data['rmse']:.4f}" if group_data["rmse"] is not None else "N/A"
        if group_data["lengths"]:
            len_str = f"长度范围=[{group_data['lengths']['min']}, {group_data['lengths']['max']}]"
        else:
            len_str = "N/A"
        print(f"  {group_data['name']}: n={group_data['count']}, RMSE={rmse_str}, {len_str}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="长序列分组分析")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_csv", type=str, required=True, help="测试集 CSV 路径")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="SMILES 列名")
    parser.add_argument("--label_col", type=str, required=True, help="标签列名")
    parser.add_argument(
        "--output", type=str, default="logs/length_group_analysis.json", help="输出路径"
    )
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    analyze_length_groups(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        output_path=args.output,
        device=args.device,
    )
