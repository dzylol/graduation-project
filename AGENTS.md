# AGENTS.md - Bi-Mamba-Chem

**Generated:** 2026-04-14
**Commit:** 7390a0b (main)
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
| `download_datasets.py` | ⚠️ **MISSING** — documented but not present in codebase |
| `scripts/manage_experiments.py` | SQLite experiment CRUD |
| `scripts/batch_train_phase1.py` | ⚠️ **UNDOCUMENTED** — verify before using, optimized for RTX 5060 Ti |

**Note:** `train_multitask.py` referenced in README does NOT exist — do not look for it.
**Note:** `scripts/batch_train_phase1.py` exists but is undocumented — verify before using.

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
- Type hints **required** on all fn signatures
- Dataclasses as DTOs (not bare dicts)
- ≤50 lines per fn
- `validate_smiles()` before RDKit processing
- Z-score normalization for regression via `LabelNormalizer`
- `create_data_loaders()` returns `(train, val, test, normalizer)`
- Device order: cuda → mps → cpu (see `get_device()`)

### MoleculeNet Datasets (via DeepChem)
ESOL (regression, 1,128), BBBP (classification, 2,039), ClinTox (classification, 1,478), FreeSolv (regression, 642), Lipophilicity (regression, 4,200)

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
├── models/           # bimamba.py (400L), bimamba_with_mamba_ssm.py (574L)
├── data/             # tokenizer.py, dataset.py, dataloader.py, split.py, column_mapping.py
├── db/               # database.py, experiment_repo.py, molecule_repo.py
├── visualization/    # dashboard.py, training_plots.py, prediction_plots.py, molecule_plots.py
└── shared/          # shared utilities
tests/
├── test_model.py
├── test_data.py
└── test_column_detection.py
scripts/
├── manage_experiments.py
└── benchmarks/       # benchmark_efficiency.py, benchmark_transformer.py, split_*.py
train.py, eval.py, download_datasets.py (MISSING)
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
| CI/CD | **Partial** — Dockerfile exists but NOT integrated with CI; no GitHub Actions workflows |
| Anti-patterns | **Clean** — no DO NOT/NEVER/ALWAYS/WARNING comments in source code |
| Package config | **Partial** — `__init__.py` added to `src/models/` and `src/data/`; use `PYTHONPATH=.` |
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

