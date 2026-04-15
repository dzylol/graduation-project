# AGENTS.md - Bi-Mamba-Chem

**Generated:** 2026-04-15
**Commit:** 34cbd91 (main)
**Language:** Python (PyTorch + RDKit)

When reasoning through problem, use draft-style thinking:
- Keep reasoning steps brief (≤5 words per step)
- Only expand when writing final code or answer
- No verbose explanations unless explicitly asked

## Overview
Bidirectional Mamba SSM for molecular property prediction. O(N) linear complexity vs Transformer's O(N²).

## Entry Points
| File | Role |
|------|------|
| `train.py` | Single-task training (ESOL/BBBP/ClinTox/FreeSolv/Lipophilicity) |
| `eval.py` | Model evaluation on test sets |
| `scripts/manage_experiments.py` | SQLite experiment CRUD |
| `scripts/batch_train_phase1.py` | Batch training script (RTX 5060 Ti optimized) |

## Key Commands

### Environment Setup
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Required on Mac to avoid OpenMP conflicts
pip install -r requirements.txt
pip install torch torchvision  # MPS/CUDA support
conda install -c conda-forge rdkit -y  # RDKit (strongly recommended)
```

### Testing (Run First)
```bash
# All tests with pytest
python -m pytest tests/ -v

# Single test file
python -m pytest tests/test_model.py -v

# Single test function (2 ways)
python -m pytest tests/test_data.py::test_tokenization -v
python -c "from tests.test_data import test_tokenization; test_tokenization()"

# With coverage report
pip install pytest pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Linting & Type Checking
```bash
ruff check src/ tests/          # Lint
ruff format src/ tests/         # Format
mypy src/                       # Type check
bandit -r src/                  # Security scan
```

## Code Style

### Imports (stdlib → third-party → local)
```python
import torch
from rdkit import Chem
from src.models.bimamba import BiMambaForPropertyPrediction
```

### Conventions (project-specific deviations)
- Type hints **required** on all fn signatures
- Dataclasses as DTOs (not bare dicts)
- ≤50 lines per fn
- `validate_smiles()` before RDKit processing
- Z-score normalization for regression via `LabelNormalizer`
- `create_data_loaders()` returns `(train, val, test, normalizer)`
- Device order: cuda → mps → cpu (see `get_device()`)

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
- Save `state_dict` only; filename: `{dataset}_bi_mamba_epoch{N}_valLoss{val_loss:.4f}.pt`

## Directory Layout
```
src/
├── models/
│   ├── __init__.py
│   ├── bimamba.py                  # Manual SSM (primary, no deps)
│   ├── bimamba_with_mamba_ssm.py   # mamba_ssm package wrapper
│   ├── bimamba_with_mamba_ssm_architecture.md
│   └── AGENTS.md
├── data/
│   ├── __init__.py
│   ├── tokenizer.py                 # SMILES tokenization
│   ├── dataset.py                   # Dataset base classes
│   ├── molecule_dataset.py          # MoleculeDataset + MoleculeTokenizer
│   ├── dataloader.py                # create_data_loaders + LabelNormalizer
│   ├── split.py                     # scaffold_split / random_split
│   ├── column_mapping.py             # CSV column detection
│   ├── molecule_dataset_architecture.md
│   └── AGENTS.md
├── db/
│   ├── __init__.py
│   ├── database.py                  # SQLite singleton + Dataclass
│   ├── experiment_repo.py           # ExperimentRepository CRUD
│   ├── molecule_repo.py             # MoleculeRepository CRUD
│   └── AGENTS.md
├── visualization/
│   ├── __init__.py
│   ├── dashboard.py                 # Multi-experiment comparison
│   ├── training_plots.py            # Training curves
│   ├── prediction_plots.py          # Prediction scatter
│   ├── molecule_plots.py            # RDKit molecule rendering
│   └── AGENTS.md
└── shared/
    ├── __init__.py
    └── utils.py                     # Training/eval arg parsing, helpers

tests/
├── __init__.py
├── test_model.py
├── test_data.py
└── test_column_detection.py

scripts/
├── manage_experiments.py
├── batch_train_phase1.py
└── benchmarks/
    ├── benchmark_efficiency.py
    ├── benchmark_transformer.py
    ├── train_esol_pooling.py
    ├── split_esol.py
    ├── split_all.py
    └── AGENTS.md

dataset/                           # MoleculeNet datasets
├── ESOL/
├── BBBP/
├── ClinTox/
├── FreeSolv/
├── Lipophilicity/
└── ... (more datasets)

train.py, eval.py
```

## Troubleshooting
| Issue | Solution |
|-------|----------|
| NaN loss | Reduce learning rate, check gradients |
| OOM errors | Decrease batch size, enable gradient checkpointing |
| RDKit failures | Validate SMILES with `_validate_smiles()` |

## Project Status

| Aspect | Status |
|--------|--------|
| CI/CD | **Partial** — no GitHub Actions workflows |
| Anti-patterns | **Clean** — no DO NOT/NEVER/ALWAYS/WARNING comments in source code |
| Package config | **Partial** — `__init__.py` in all subpackages; use `PYTHONPATH=.` |
| Test config | **None** — pytest runs without config file |

## Module-Specific AGENTS.md
- `src/models/AGENTS.md` — BiMamba architecture, fusion modes, pooling
- `src/visualization/AGENTS.md` — Plotting conventions, RDKit molecule rendering
- `src/data/AGENTS.md` — SMILES tokenization, dataset handling, z-score norm
- `src/db/AGENTS.md` — SQLite persistence, ExperimentRepository, singleton pattern
- `tests/AGENTS.md` — Test conventions, dual-mode execution, pytest patterns
- `scripts/benchmarks/AGENTS.md` — O(N) benchmarking scripts

## Testing Conventions

> See `tests/AGENTS.md` for full conventions (dual-mode, no fixtures, step-by-step style).

```bash
python -m pytest tests/ -v   # All tests
python tests/test_model.py    # Standalone (no pytest)
```
