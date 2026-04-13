import torch
import torch.nn as nn
import pandas as pd


class SmallTransformer(nn.Module):
    """Standard Transformer with comparable parameters to Bi-Mamba.

    Same d_model=256, num_layers=4 for fair comparison.
    Uses global average pooling to match Bi-Mamba's mean pooling.
    """

    def __init__(self, vocab_size=45, d_model=256, nhead=8, num_layers=4, max_len=4096):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling (match Bi-Mamba)
        return self.head(x)


def benchmark_transformer():
    """Benchmark Transformer inference time vs sequence length."""
    device = "cuda"
    model = SmallTransformer(vocab_size=45, d_model=256, nhead=8, num_layers=4).to(
        device
    )
    model.eval()

    # Warm-up
    print("Warming up...")
    dummy_input = torch.randint(4, 45, (8, 64)).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids=dummy_input)
    torch.cuda.synchronize()

    # Benchmark
    lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    batch_size = 8
    results = []

    print("\nTransformer Inference Benchmark")
    print("-" * 50)

    for length in lengths:
        tokens = torch.randint(4, 45, (batch_size, length)).to(device)

        # Warm-up for this length
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids=tokens)
        torch.cuda.synchronize()

        # Timing runs
        N = 50
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(N):
            with torch.no_grad():
                _ = model(input_ids=tokens)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / N
        results.append({"length": length, "time_ms": elapsed_ms})
        print(f"Length {length:4d}: {elapsed_ms:.3f} ms")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("/workspace/.sisyphus/evidence/transformer-efficiency.csv", index=False)
    print(f"\nSaved to /workspace/.sisyphus/evidence/transformer-efficiency.csv")
    return df


if __name__ == "__main__":
    df = benchmark_transformer()
    print("\nResults:")
    print(df)
