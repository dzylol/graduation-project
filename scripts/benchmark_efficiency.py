"""
效率对比脚本 - 比较 Bi-Mamba vs Vanilla Transformer

测量训练时间和显存使用。
"""

import argparse
import json
import time
import torch
import numpy as np

from src.models.bimamba_with_mamba_ssm import BiMambaForPropertyPrediction
from src.models.vanilla_transformer import VanillaTransformerForPropertyPrediction


def benchmark_model(
    model_class,
    model_kwargs,
    input_ids,
    n_iterations: int = 100,
    warmup: int = 10,
    device: str = "cuda",
):
    model = model_class(**model_kwargs).to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for i in range(warmup):
            _ = model(input_ids.to(device))

        if device == "cuda":
            torch.cuda.synchronize()

        for i in range(n_iterations):
            start = time.perf_counter()
            _ = model(input_ids.to(device))
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    memory = 0
    if device == "cuda":
        memory = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "memory_mb": memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="logs/efficiency_results.json")
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    seq_length = args.seq_length
    n_iter = args.n_iterations

    input_ids = torch.randint(0, 45, (batch_size, seq_length))

    common_kwargs = {
        "vocab_size": 45,
        "d_model": 256,
        "n_layers": 4,
        "num_labels": 1,
        "task_type": "regression",
        "pooling": "mean",
        "dropout": 0.1,
        "pad_token_id": 0,
    }

    print("=" * 60)
    print("效率对比: Bi-Mamba vs Vanilla Transformer")
    print("=" * 60)
    print(f"Batch size: {batch_size}, Seq length: {seq_length}")
    print(f"Device: {device}, Iterations: {n_iter}")
    print()

    print("Benchmarking Bi-Mamba...")
    bimamba_results = benchmark_model(
        BiMambaForPropertyPrediction,
        {
            **common_kwargs,
            "d_mamba": 256,
        },
        input_ids,
        n_iterations=n_iter,
        device=device,
    )
    print(
        f"  Avg time: {bimamba_results['avg_time_ms']:.2f} ± {bimamba_results['std_time_ms']:.2f} ms"
    )
    print(f"  Memory: {bimamba_results['memory_mb']:.2f} MB")

    print()
    print("Benchmarking Vanilla Transformer...")
    transformer_results = benchmark_model(
        VanillaTransformerForPropertyPrediction,
        {
            **common_kwargs,
            "n_heads": 8,
            "d_ffn": 512,
            "max_seq_length": 512,
        },
        input_ids,
        n_iterations=n_iter,
        device=device,
    )
    print(
        f"  Avg time: {transformer_results['avg_time_ms']:.2f} ± {transformer_results['std_time_ms']:.2f} ms"
    )
    print(f"  Memory: {transformer_results['memory_mb']:.2f} MB")

    results = {
        "config": {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "n_iterations": n_iter,
            "device": device,
        },
        "bimamba": bimamba_results,
        "transformer": transformer_results,
        "comparison": {
            "time_speedup": transformer_results["avg_time_ms"] / bimamba_results["avg_time_ms"],
            "memory_ratio": bimamba_results["memory_mb"] / transformer_results["memory_mb"],
        },
    }

    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"结果已保存到: {args.output}")
    print(f"时间加速比: {results['comparison']['time_speedup']:.2f}x")
    print(f"显存节省比: {results['comparison']['memory_ratio']:.2f}x")


if __name__ == "__main__":
    main()
