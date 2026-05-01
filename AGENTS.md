# AGENTS.md — Bi-Mamba-Chem

**Updated:** 2026-05-01
**Latest commit:** ca0b0b9 (main)
**Language:** Python (PyTorch + RDKit)

## Overview

Bidirectional Mamba SSM for molecular property prediction. O(N) linear complexity vs Transformer's O(N²). Targets long-sequence biomolecules; verified on small-molecule SMILES as proof-of-concept.

## Entry Points

| File | Role |
|------|------|
| `train.py` | Single-task training (ESOL/BBBP/ClinTox/FreeSolv/Lipophilicity) |
| `eval.py` | Model evaluation on test sets |
| `scripts/manage_experiments.py` | SQLite experiment CRUD |
| `scripts/batch_train_phase1.py` | Batch training (RTX 5060 Ti optimized) |

## Environment Setup

```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Required on macOS — avoid OpenMP conflicts
pip install -r requirements.txt
pip install torch torchvision     # MPS / CUDA support
conda install -c conda-forge rdkit -y
```

- **`uv.lock`** is present → `uv sync` works as an alternative to pip.
- **`environment.yml`** → full conda env (includes mamba-ssm, causal-conv1d for GPU).
- **`Dockerfile`** → CUDA 12.8 container with mamba-ssm + causal-conv1d from source tarballs.

## Key Commands

### Testing (run first after any change)

```bash
python -m pytest tests/ -v                    # all tests
python -m pytest tests/test_model.py -v       # single file
python -m pytest tests/test_data.py::test_tokenization -v  # single function
python tests/test_model.py                    # standalone (no pytest)
python -m pytest tests/ --cov=src --cov-report=term-missing  # coverage
```

Tests are dual-mode: each file runs under both `pytest` and `python file.py`. No `conftest.py`, no shared fixtures. See `tests/AGENTS.md` for full conventions.

### Linting & Type Checking

```bash
ruff check src/ tests/      # lint (line-length 100 via pyproject.toml)
ruff format src/ tests/     # format
mypy src/                   # type check
bandit -r src/              # security scan
```

### Package config

`pyproject.toml` declares setuptools build, Python ≥3.9, and optional dev deps (`pytest`, `ruff`). Run with `PYTHONPATH=.` (no editable install required).

## Code Style

### Imports (stdlib → third-party → local)

```python
import torch
from rdkit import Chem
from src.models.bimamba import BiMambaForPropertyPrediction
```

### Conventions

- Type hints **required** on all fn signatures.
- Dataclasses as DTOs (not bare dicts).
- ≤50 lines per fn.
- `validate_smiles_internal()` before RDKit processing (enabled by default in `MoleculeDataset`).
- Z-score normalization for regression via `LabelNormalizer`.
- `create_data_loaders()` returns `(train, val, test, normalizer)`.
- Device order: `cuda → mps → cpu` (see `get_device()` in `src/shared/utils.py`).

### MoleculeNet Datasets

ESOL (regression, ~1,100), BBBP (classification, ~2,000), ClinTox (classification, ~1,500), FreeSolv (regression, ~640), Lipophilicity (regression, ~4,200), bace (classification), HIV (classification), SIDER (classification), MUV (classification), ZINC250K (regression), EGFR (regression), mpro (regression), BDB2020+ (regression)

### Device Management

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### Training Best Practices

- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Linear warmup for first 5 epochs
- Save `state_dict` only; filename: `{dataset}_bi_mamba_best.pt`
- **MLP head for regression**: `nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2, num_labels))`
- **Loss types**: `--loss_type mse | smooth_l1 | huber` for regression robustness

### Regression Head Architecture

```
task_type=regression   → MLP head (d_model → d_model//2 → num_labels)
task_type=classification → Linear head (d_model → num_labels)
```

## Directory Layout

```
src/
├── models/
│   ├── __init__.py
│   ├── bimamba.py                          # Manual SSM (primary, no deps)
│   ├── bimamba_with_mamba_ssm.py           # mamba_ssm package wrapper
│   ├── bimamba_with_mamba_ssm_architecture.md
│   ├── vanilla_transformer.py              # Transformer baseline
│   └── AGENTS.md
├── data/                                   # Consolidated — see note below
│   ├── __init__.py                         # Re-exports all public symbols
│   ├── tokenizer.py                        # MoleculeTokenizer (SMILES → token indices)
│   ├── molecule_dataset.py                 # MoleculeDataset, LabelNormalizer, create_data_loaders,
│   │                                       #   ColumnMapping, scaffold/random split, DB dataset
│   ├── split.py                            # select_database, list_available_databases
│   ├── molecule_dataset_architecture.md
│   └── AGENTS.md
├── db/
│   ├── __init__.py
│   ├── database.py                         # SQLite singleton + Dataclass
│   ├── experiment_repo.py                  # ExperimentRepository CRUD
│   ├── molecule_repo.py                    # MoleculeRepository CRUD
│   └── AGENTS.md
├── visualization/
│   ├── __init__.py
│   ├── dashboard.py                        # Multi-experiment comparison
│   ├── training_plots.py                   # Training curves
│   ├── prediction_plots.py                 # Prediction scatter
│   ├── molecule_plots.py                   # RDKit molecule rendering
│   └── AGENTS.md
├── shared/
│   ├── __init__.py
│   ├── utils.py                            # parse_train_args, get_device, evaluate, set_seed
│   └── AGENTS.md
└── utils/                                  # ⚠️ EMPTY — stale directory, ignore

tests/
├── __init__.py
├── test_model.py
├── test_data.py
├── test_column_detection.py
└── AGENTS.md

scripts/
├── manage_experiments.py
├── batch_train_phase1.py
├── benchmark_efficiency.py                 # Top-level efficiency benchmark
├── plot_efficiency.py                      # Efficiency plotting
├── analyze_length_groups.py                # RMSE by sequence length
├── AGENTS.md
└── benchmarks/
    ├── benchmark_efficiency.py
    ├── benchmark_transformer.py
    ├── train_esol_pooling.py
    ├── split_esol.py
    ├── split_all.py
    ├── split_data.sh
    ├── train_esol.sh
    └── AGENTS.md

dataset/                                     # MoleculeNet datasets
├── ESOL/ BBBP/ ClinTox/ FreeSolv/ Lipophilicity/
├── bace/ HIV/ SIDER/ MUV/ ZINC250K/
└── EGFR/ mpro/ BDB2020+/

help-learning-file/                          # Educational annotated copies (not production)
├── bimamba.py, bimamba_with_mamba_ssm.py
├── molecule_dataset.py, mamba_theory.md
```

**`src/data/` consolidation note:** As of commit 0081924, `dataset.py`, `dataloader.py`, and `column_mapping.py` were merged into `molecule_dataset.py`. All public symbols (MoleculeDataset, create_data_loaders, LabelNormalizer, ColumnMapping, etc.) are re-exported from `__init__.py`. Do NOT create separate `dataset.py` / `dataloader.py` files.

Top-level files: `train.py`, `eval.py`, `pyproject.toml`, `uv.lock`, `requirements.txt`, `environment.yml`, `Dockerfile`, `README.md`, `mamba.tutorial.md`

`.claude/` — Claude agent permissions config (not used by OpenCode).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NaN loss | Reduce LR, check gradients |
| OOM | Decrease batch size / d_model / n_layers |
| RDKit failures | SMILES validated by `validate_smiles_internal()` in dataset; set `validate_smiles=False` to skip (faster load) |
| MPS unavailable | Use conda PyTorch or fall back to `--device cpu` |

## Project Status

| Aspect | Status |
|--------|--------|
| CI/CD | None — no GitHub Actions workflows |
| Package config | `pyproject.toml` + `setup.py`-style; `PYTHONPATH=.` for dev |
| Test config | pytest via `pyproject.toml` `[tool.pytest.ini_options]`; no separate config file |
| Dependency locking | `uv.lock` present; `requirements.txt` for pip fallback |

## Module-Specific AGENTS.md

- `src/models/AGENTS.md` — BiMamba architecture, fusion modes, pooling, MLP head
- `src/visualization/AGENTS.md` — Plotting conventions, RDKit molecule rendering
- `src/data/AGENTS.md` — SMILES tokenization, dataset handling, z-score norm (⚠️ file tree inside is stale — see note above)
- `src/db/AGENTS.md` — SQLite persistence, ExperimentRepository, singleton pattern
- `src/shared/AGENTS.md` — Argparse factories, device management, evaluate()
- `tests/AGENTS.md` — Test conventions, dual-mode execution, pytest patterns
- `scripts/AGENTS.md` — Batch training, experiment management
- `scripts/benchmarks/AGENTS.md` — O(N) benchmarking scripts
