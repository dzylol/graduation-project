# AGENTS.md - scripts/

**Generated:** 2026-04-20

**Operational scripts for experiment management, batch training, and analysis.**

## Structure
```
scripts/
├── manage_experiments.py      # SQLite CRUD CLI
├── batch_train_phase1.py      # Batch training orchestrator (81 experiments)
├── benchmark_efficiency.py    # Training/memory efficiency
├── plot_efficiency.py         # Efficiency plotting
├── analyze_length_groups.py   # RMSE by sequence length
└── benchmarks/                # O(N) benchmarking suite
    ├── benchmark_efficiency.py
    ├── benchmark_transformer.py
    ├── train_esol_pooling.py
    ├── split_esol.py
    └── split_all.py
```

## Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `manage_experiments.py` | Experiment CRUD | `python scripts/manage_experiments.py --list` |
| `batch_train_phase1.py` | 81-experiment batch | `python scripts/batch_train_phase1.py` |
| `benchmarks/*.py` | Ablation studies | See benchmarks/AGENTS.md |

## manage_experiments.py Commands
```bash
python scripts/manage_experiments.py --list              # List all
python scripts/manage_experiments.py --list --status completed
python scripts/manage_experiments.py -d <id>             # Detail
python scripts/manage_experiments.py -c <id1> <id2>     # Compare
python scripts/manage_experiments.py --delete <id>      # Delete
```

## batch_train_phase1.py
- Runs 81 experiments: 3 datasets × 3 poolings × 3 d_model × 3 n_layers
- Optimized for RTX 5060 Ti
- GPU via Podman: `podman run --rm --device nvidia.com/gpu=all`

## Anti-Patterns
- **NEVER** commit experiment outputs (checkpoints/, logs/)
- **NEVER** run batch_train without GPU warmup
