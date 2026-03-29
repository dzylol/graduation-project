# AGENTS.md - Bi-Mamba-Chem

**Generated:** 2026-03-23
**Commit:** 4e2d9f2 (main)
**Language:** Python (PyTorch + RDKit)

## Overview
Bidirectional Mamba SSM for molecular property prediction. O(N) linear complexity vs Transformer's O(N²).

## Entry Points
| File | Role |
|------|------|
| `train.py` | Single-task training (ESOL/BBBP/ClinTox/FreeSolv/Lipophilicity) |
| `eval.py` | Model evaluation on test sets |
| `download_datasets.py` | Download full MoleculeNet datasets via DeepChem |
| `scripts/manage_experiments.py` | SQLite experiment CRUD |

**Note:** `train_multitask.py` referenced in README does NOT exist — do not look for it.

## Key Commands

### Environment Setup
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Required on Mac to avoid OpenMP conflicts
pip install -r requirements.txt
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
- Type hints **required** on all function signatures
- Dataclasses as DTOs (not bare dicts)
- ≤50 lines per function
- `validate_smiles()` before RDKit processing
- Device order: cuda → mps → cpu (see `get_device()`)
- Z-score normalization is done automatically for regression tasks via `LabelNormalizer`

## ML-Specific Guidelines

### Data Pipeline
- Validate SMILES via `_validate_smiles()` before RDKit processing
- Cache tokenized sequences to disk
- Z-score normalization is done automatically for regression via `LabelNormalizer`
- `create_data_loaders()` returns `(train_loader, val_loader, test_loader, normalizer)`

### Downloading Datasets
```bash
python download_datasets.py                    # Download all via DeepChem
python download_datasets.py --dataset ESOL   # Specific dataset
python download_datasets.py --zinc           # ZINC 250K pretraining data
python download_datasets.py --example         # Force tiny example data
```

### MoleculeNet Datasets (via DeepChem)
| Dataset | Task | Molecules | Metric |
|---------|------|-----------|--------|
| ESOL | Regression | 1,128 | RMSE |
| BBBP | Classification | 2,039 | ROC-AUC |
| ClinTox | Classification | 1,478 | ROC-AUC |
| FreeSolv | Regression | 642 | RMSE |
| Lipophilicity | Regression | 4,200 | RMSE |

### Pretraining (Two-Stage)
For SMILES-Mamba style training: pretrain on ZINC 250K, then fine-tune on downstream tasks.

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

### Assertion Best Practices
```python
assert abs(pred - expected) < 1e-5, f"Expected ~{expected}, got {pred}"
```

## Directory Layout
```
src/
├── models/           # bimamba.py (477L), bimamba_with_mamba_ssm.py (436L)
├── data/             # molecule_dataset.py (374L)
├── db/               # database.py, experiment_repo.py, molecule_repo.py
├── visualization/    # dashboard.py, training_plots.py, prediction_plots.py, molecule_plots.py
└── (utils/)         # EMPTY — do not use
tests/
├── test_model.py
└── test_data.py
scripts/
└── manage_experiments.py
train.py, eval.py, download_datasets.py
```

## Key Files
| File | Description |
|------|-------------|
| `src/models/bimamba.py` | Core BiMamba model (BiMambaBlock, BiMambaEncoder, BiMambaForPropertyPrediction) |
| `src/data/molecule_dataset.py` | MoleculeTokenizer, MoleculeDataset, create_data_loaders |
| `train.py` | Training entry point |
| `eval.py` | Evaluation script |

## Troubleshooting
| Issue | Solution |
|-------|----------|
| NaN loss | Reduce learning rate, check gradients |
| OOM errors | Decrease batch size, enable gradient checkpointing |
| RDKit failures | Validate SMILES with `_validate_smiles()` |
| MPS errors | Use CPU for debugging: `--device cpu` |

## Project Status

| Aspect | Status |
|--------|--------|
| CI/CD | **None** — no GitHub Actions, Docker, or automated pipelines |
| Anti-patterns | **Clean** — no DO NOT/NEVER/ALWAYS/WARNING comments in source |
| Package config | **None** — not pip-installable (use `PYTHONPATH=.` or `pip install -e .`) |
| Test config | **None** — pytest runs without config file |

## Module-Specific AGENTS.md

| Module | File | Purpose |
|--------|------|---------|
| `src/models/` | AGENTS.md | BiMamba architecture, fusion modes, pooling |
| `src/visualization/` | AGENTS.md | Plotting conventions, RDKit molecule rendering |
| `src/data/` | AGENTS.md | SMILES tokenization, dataset handling, z-score norm |
| `src/db/` | AGENTS.md | SQLite persistence, ExperimentRepository, singleton pattern |
| `tests/` | AGENTS.md | Test conventions, dual-mode execution, pytest patterns |

## Testing Conventions

> See `tests/AGENTS.md` for full conventions (dual-mode, no fixtures, step-by-step style).

```bash
python -m pytest tests/ -v   # All tests
python tests/test_model.py    # Standalone (no pytest)
```
