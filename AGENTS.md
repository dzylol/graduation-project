# AGENTS.md - Bi-Mamba-Chem

## Project Overview
Bi-Mamba-Chem implements a bidirectional Mamba architecture for molecular property prediction using PyTorch and RDKit.

## Key Commands

### Environment Setup
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Required on Mac to avoid OpenMP conflicts
pip install -r requirements.txt
```

### Testing (Run First)
```bash
# All tests
python -m pytest tests/ -v

# Single test file
python tests/test_model.py
python tests/test_data.py

# Single test function (pytest)
python -m pytest tests/test_data.py::test_tokenization -v

# Single test function (direct import)
python -c "from tests.test_data import test_tokenization; test_tokenization()"
```

### Linting & Type Checking
```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## Code Style

### Imports (stdlib → third-party → local)
```python
import os
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from rdkit import Chem

from src.models.bimamba import BiMambaForPropertyPrediction
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `BiMambaBlock` |
| Functions/methods | snake_case | `create_bimamba_model` |
| Constants | snake_case | `max_length = 512` |
| Module-level private | `_internal` suffix | `validate_smiles_internal` |
| Test functions | `test_` prefix | `test_model_forward_pass` |

### Type Hints (Required)
```python
def create_bimamba_model(
    vocab_size: int,
    d_model: int = 256,
    n_layers: int = 4,
    task_type: str = "regression",
    num_labels: int = 1,
    **kwargs,
) -> BiMambaForPropertyPrediction:
```

### Docstrings (Google Style)
```python
def predict_property(smiles: str, model: nn.Module) -> float:
    """Predict molecular property.

    Args:
        smiles: Molecule SMILES string
        model: Trained prediction model

    Returns:
        float: Predicted property value

    Raises:
        ValueError: On invalid SMILES
        RuntimeError: On model inference failure
    """
```

### Error Handling
```python
# Validate inputs early
if not isinstance(smiles, str) or not smiles:
    raise ValueError("smiles must be non-empty string")

# Specific exceptions
raise FileNotFoundError(f"Dataset not found: {data_path}")
raise RuntimeError(f"Model forward pass failed: {e}")
```

### Function Design
- ≤50 lines per function
- Single responsibility principle
- Use dataclasses for configuration objects

## ML-Specific Guidelines

### Data Pipeline
- Validate SMILES via `_validate_smiles()` before RDKit processing
- Cache tokenized sequences to disk
- Normalize regression targets (z-score)

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
├── models/           # bimamba.py, multitask.py
├── data/             # molecule_dataset.py, multitask_dataset.py
├── db/               # database.py, experiment_repo.py, molecule_repo.py
├── utils/
└── visualization/    # dashboard.py, molecule_plots.py, prediction_plots.py
tests/
├── test_model.py
└── test_data.py
train.py, eval.py, train_multitask.py, download_datasets.py
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
