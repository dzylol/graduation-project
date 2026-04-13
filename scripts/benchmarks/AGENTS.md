# AGENTS.md - scripts/benchmarks/

**Efficiency benchmarking + data splitting scripts.** Used for ablation experiments and O(N) vs O(N²) complexity validation.

## Structure
```
scripts/benchmarks/
├── benchmark_efficiency.py   # Bi-Mamba vs Transformer inference timing
├── benchmark_transformer.py  # Standalone transformer benchmark
├── split_all.py              # Split all datasets (ESOL/BBBP/ClinTox)
├── split_data.sh             # Shell wrapper for data splitting
├── split_esol.py            # ESOL-specific data split
├── train_evol_pooling.py     # ESOL ablation pooling comparison
└── train_esol.sh            # Shell wrapper for ESOL training
```

## Usage

### Efficiency Benchmark (Key Script)
```bash
# Run on remote GPU server via Podman
ssh qfh@19.tcp.vip.cpolar.cn -p 11668
cd ~/graduation-project
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  python scripts/benchmarks/benchmark_efficiency.py
```

### Data Splitting
```bash
python scripts/benchmarks/split_all.py  # Split all datasets
python scripts/benchmarks/split_esol.py  # ESOL only
```

## Key Results (from experiments)

| Model | Complexity | Speedup at 4096 tokens |
|-------|-----------|----------------------|
| Bi-Mamba | O(N^0.65) | 3.2x vs Transformer |
| Transformer | O(N^1.30) | baseline |

## Anti-Patterns (THIS MODULE)
- **NEVER** run benchmark without GPU warmup (first run is cold)
- **NEVER** compare different batch sizes — use same batch_size for fair comparison