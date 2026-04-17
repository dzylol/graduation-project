"""
绘制效率对比图 - Bi-Mamba vs Transformer

生成论文用图：时间对比和显存对比。
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_time_comparison(results_128: dict, results_512: dict, output_path: str):
    seq_lengths = [128, 512]
    bimamba_times = [results_128["bimamba"]["avg_time_ms"], results_512["bimamba"]["avg_time_ms"]]
    transformer_times = [
        results_128["transformer"]["avg_time_ms"],
        results_512["transformer"]["avg_time_ms"],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(seq_lengths))
    width = 0.35
    bars1 = ax.bar(x - width / 2, bimamba_times, width, label="Bi-Mamba", color="#2E86AB")
    bars2 = ax.bar(x + width / 2, transformer_times, width, label="Transformer", color="#A23B72")
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Inference Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_memory_comparison(results_128: dict, results_512: dict, output_path: str):
    seq_lengths = [128, 512]
    bimamba_memory = [results_128["bimamba"]["memory_mb"], results_512["bimamba"]["memory_mb"]]
    transformer_memory = [
        results_128["transformer"]["memory_mb"],
        results_512["transformer"]["memory_mb"],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(seq_lengths))
    width = 0.35
    bars1 = ax.bar(x - width / 2, bimamba_memory, width, label="Bi-Mamba", color="#2E86AB")
    bars2 = ax.bar(x + width / 2, transformer_memory, width, label="Transformer", color="#A23B72")
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax.set_title("Memory Usage Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_128",
        type=str,
        default="logs/efficiency_results.json",
        help="Results for seq_len 128",
    )
    parser.add_argument(
        "--input_512",
        type=str,
        default="logs/efficiency_results_512.json",
        help="Results for seq_len 512",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/efficiency",
        help="Output prefix",
    )
    args = parser.parse_args()

    with open(args.input_128) as f:
        results_128 = json.load(f)
    with open(args.input_512) as f:
        results_512 = json.load(f)

    plot_time_comparison(results_128, results_512, "logs/efficiency_time.png")
    plot_memory_comparison(results_128, results_512, "logs/efficiency_memory.png")


if __name__ == "__main__":
    main()
