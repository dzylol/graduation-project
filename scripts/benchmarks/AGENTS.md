# AGENTS.md - scripts/benchmarks/

**Efficiency benchmarking + ablation experiment scripts.** O(N) vs O(N²) complexity validation, pooling strategy comparison.

## Overview
Benchmarks validate Bi-Mamba's linear complexity advantage over Transformer. Ablation scripts test pooling strategies on ESOL dataset.

## Structure

```
scripts/benchmarks/
├── benchmark_efficiency.py      # Bi-Mamba vs Transformer inference timing (203L)
├── benchmark_transformer.py     # Standalone Transformer benchmark (92L)
├── split_all.py                 # Split all datasets (ESOL/BBBP/ClinTox)
├── split_evol.py                # ESOL-only data split (4L)
├── split_data.sh                # Podman wrapper: split all via container
├── train_evol_pooling.py        # ESOL pooling ablation (mean/max/cls)
└── train_evol.sh                # Podman wrapper: run ESOL training
```

## Usage

### Efficiency Benchmark (Primary)
```bash
# Local run
python scripts/benchmarks/benchmark_efficiency.py

# Remote GPU via Podman
ssh qfh@19.tcp.vip.cpolar.cn -p 11668
cd ~/graduation-project
podman run --rm -v "$(pwd):/workspace" --workdir /workspace \
  --device nvidia.com/gpu=all localhost/bimamba \
  python scripts/benchmarks/benchmark_efficiency.py
```

### Standalone Transformer Benchmark
```bash
python scripts/benchmarks/benchmark_transformer.py
# Saves to .sisyphus/evidence/transformer-efficiency.csv
```

### Data Splitting
```bash
python scripts/benchmarks/split_all.py           # All datasets
python scripts/benchmarks/split_evol.py          # ESOL only
bash scripts/benchmarks/split_data.sh             # Podman wrapper
```

### Pooling Ablation
```bash
python scripts/benchmarks/train_evol_pooling.py  # mean/max/cls comparison
bash scripts/benchmarks/train_evol.sh             # Podman wrapper
```

## Key Results

| Model | Complexity | Speedup at 4096 tokens |
|-------|-----------|------------------------|
| Bi-Mamba | O(N^0.65) | 3.2x vs Transformer |
| Transformer | O(N^1.30) | baseline |

**Success criteria**: log-log slope Bi-Mamba < 1.3, Transformer > 1.5

## Anti-Patterns
- **NEVER** run benchmark without GPU warmup (10 warmup + 50 timing iterations)
- **NEVER** compare different batch sizes (use batch_size=8 for fair comparison)
- **NEVER** run once and trust results (cold start ~2x slower on first run)
