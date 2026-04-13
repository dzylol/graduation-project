#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.transformer(x)
        return self.head(x.mean(dim=1))


def create_bimamba(
    vocab_size, d_model, n_layers, pooling, dropout, pad_token_id, max_seq_length
):
    from src.models.bimamba_with_mamba_ssm import (
        BiMambaForPropertyPrediction,
    )

    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        pooling=pooling,
        dropout=dropout,
        pad_token_id=pad_token_id,
        max_seq_length=max_seq_length,
    )

    return BiMambaForPropertyPrediction(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        pooling=pooling,
        dropout=dropout,
        pad_token_id=pad_token_id,
    )


def benchmark_model(model, tokens, warmup=10, iterations=50):
    model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids=tokens)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_ids=tokens)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def run_benchmark():
    vocab_size = 45
    d_model = 256
    n_layers = 4
    batch_size = 8
    max_seq_length = 8192
    lengths = [64, 128, 256, 512, 1024, 2048, 4096]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(".sisyphus/evidence")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("\nBi-Mamba Benchmark")
    bimamba = create_bimamba(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        pooling="mean",
        dropout=0.1,
        pad_token_id=0,
        max_seq_length=max_seq_length,
    )
    bimamba.eval().to(device)
    print(f"Params: {sum(p.numel() for p in bimamba.parameters()):,}")

    bimamba_times = []
    for length in lengths:
        tokens = torch.randint(4, vocab_size, (batch_size, length), device=device)
        elapsed = benchmark_model(bimamba, tokens)
        bimamba_times.append(elapsed)
        results.append({"model": "Bi-Mamba", "length": length, "time_ms": elapsed})
        print(f"  {length:4d}: {elapsed:.3f}ms")

    del bimamba
    torch.cuda.empty_cache()

    print("\nTransformer Benchmark")
    transformer = SmallTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=8,
        num_layers=n_layers,
    )
    transformer.eval().to(device)
    print(f"Params: {sum(p.numel() for p in transformer.parameters()):,}")

    transformer_times = []
    for length in lengths:
        tokens = torch.randint(4, vocab_size, (batch_size, length), device=device)
        elapsed = benchmark_model(transformer, tokens)
        transformer_times.append(elapsed)
        results.append({"model": "Transformer", "length": length, "time_ms": elapsed})
        print(f"  {length:4d}: {elapsed:.3f}ms")

    df = pd.DataFrame(results)
    csv_path = output_dir / "efficiency_benchmark.csv"
    df.to_csv(csv_path, index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.loglog(
        lengths,
        bimamba_times,
        "o-",
        label="Bi-Mamba (O(N))",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
    )
    plt.loglog(
        lengths,
        transformer_times,
        "s-",
        label="Transformer (O(N²))",
        linewidth=2,
        markersize=8,
        color="#E94F37",
    )

    lengths_arr = np.array(lengths)
    t_on = bimamba_times[0] * (lengths_arr / lengths_arr[0])
    plt.loglog(lengths, t_on, "--", color="gray", alpha=0.5, label="Theoretical O(N)")
    t_on2 = transformer_times[0] * (lengths_arr / lengths_arr[0]) ** 2
    plt.loglog(lengths, t_on2, ":", color="red", alpha=0.5, label="Theoretical O(N²)")

    bimamba_slope = np.polyfit(np.log(lengths), np.log(bimamba_times), 1)[0]
    transformer_slope = np.polyfit(np.log(lengths), np.log(transformer_times), 1)[0]

    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Inference Time (ms)", fontsize=12)
    plt.title(
        f"Bi-Mamba vs Transformer: O(N) vs O(N²) Complexity\nSlopes: Bi-Mamba={bimamba_slope:.2f}, Transformer={transformer_slope:.2f}",
        fontsize=14,
    )
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "efficiency-comparison.png", dpi=150)
    plt.savefig(output_dir / "efficiency-comparison.pdf")
    plt.close()

    print(f"\nCSV: {csv_path}")
    print(f"Plot: {output_dir}/efficiency-comparison.{{png,pdf}}")

    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(f"Bi-Mamba log-log slope: {bimamba_slope:.2f} (expected ~1.0 for O(N))")
    print(
        f"Transformer log-log slope: {transformer_slope:.2f} (expected ~2.0 for O(N²))"
    )
    print(f"\nBi-Mamba: O(N^{bimamba_slope:.2f})")
    print(f"Transformer: O(N^{transformer_slope:.2f})")

    if transformer_slope > 1.5 and bimamba_slope < 1.3:
        print("\n✓ SUCCESS: Bi-Mamba shows O(N) vs Transformer's O(N²)")
    else:
        print("\n⚠ WARNING: Results may not clearly show O(N) vs O(N²)")

    return bimamba_slope, transformer_slope


if __name__ == "__main__":
    run_benchmark()
