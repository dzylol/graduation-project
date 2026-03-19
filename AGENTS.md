# AGENTS.md - Bi-Mamba-Chem Codebase Guidelines

## Project Overview
Bi-Mamba-Chem implements a bidirectional Mamba architecture for molecular property prediction.
Uses PyTorch with RDKit for molecular data processing.

## Directory Layout
```
├── src/
│   ├── models/           # bimamba.py, multitask.py
│   ├── data/             # molecule_dataset.py, multitask_dataset.py
│   ├── db/               # database.py, experiment_repo.py, molecule_repo.py
│   ├── utils/            # Utility modules
│   └── visualization/    # dashboard.py, molecule_plots.py, prediction_plots.py, training_plots.py
├── tests/
│   ├── test_model.py     # Model unit tests
│   └── test_data.py      # Data processing tests
├── scripts/
│   └── manage_experiments.py
├── train.py              # Training entry point
├── eval.py              # Evaluation script
├── train_multitask.py   # Multitask training
├── download_datasets.py  # Dataset downloader
├── requirements.txt     # Runtime dependencies
└── environment.yml      # Conda environment (alternative)
```

## Build, Lint, Test

### Environment Setup
```bash
# Option 1: pip (for NVIDIA GPU systems)
pip install -r requirements.txt

# Option 2: conda (for Apple Silicon or CPU-only systems)
conda env create -f environment.yml
conda activate bi-mamba-chem
```

### Testing Commands
```bash
# All tests with pytest
python -m pytest tests/ -v

# Run specific test file directly
python tests/test_model.py
python tests/test_data.py

# Single test function (pytest style)
python -m pytest tests/test_data.py::test_tokenization -v

# Single test function (direct import)
python -c "from tests.test_data import test_tokenization; test_tokenization()"

# Coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

### Linting & Formatting
```bash
# Install ruff for linting and formatting
pip install ruff

# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking
pip install mypy
mypy src/
```

## Container Testing with Podman

### Prerequisites
```bash
# Install Podman (macOS)
brew install podman

# Install Podman (Linux)
sudo apt-get install podman  # Debian/Ubuntu
sudo dnf install podman    # Fedora
```

### GPU Detection & Container Execution

**Step 1: Check for NVIDIA GPU**
```bash
# Check if NVIDIA GPU is available
nvidia-smi

# If nvidia-smi shows driver version and CUDA version, you have NVIDIA GPU
# Example output:
# +------------------------------------------------------------------+
# | NVIDIA-SMI 535.54.03    Driver Version: 535.54.03  CUDA Version: 12.2 |
# +------------------------------------------------------------------+
```

**NVIDIA GPU Requirements for Mamba:**
| Component | Minimum Version |
|-----------|-----------------|
| NVIDIA Driver | 525.60.13+ |
| CUDA Toolkit | 11.6+ (12.x recommended) |
| VRAM | 2GB (130M model) / 4GB (370M) / 8GB (790M) |

**Step 2: Run Tests in Podman**
```bash
# For NVIDIA GPU systems - use nvidia-container-toolkit
podman run --rm \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
    bash -c "pip install -r requirements.txt && python -m pytest tests/ -v"

# For Apple Silicon / AMD GPU / CPU-only systems - use conda image
podman run --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    docker.io/continuumio/miniconda3:latest \
    bash -c "
        conda install -y -c conda-forge python=3.10 pytorch torchvision torchaudio \
            pytorch-cuda=11.8 numpy pandas rdkit scikit-learn tqdm matplotlib && \
        python tests/test_model.py && \
        python tests/test_data.py
    "
```

**Fallback: Local Conda Testing (Recommended for Apple Silicon)**
```bash
# Create conda environment
conda create -n bimamba python=3.10 -y
conda activate bimamba

# Install PyTorch with MPS support (Apple Silicon)
conda install pytorch torchvision torchaudio pytorch::pytorch -c pytorch -y

# Install other dependencies
conda install -c conda-forge numpy pandas scikit-learn tqdm matplotlib -y
conda install -c conda-forge rdkit -y

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Run tests
python tests/test_model.py
python tests/test_data.py
```

## Code Style

### Imports
- Order: stdlib → third-party → local
- Blank line between groups
```python
import os
import json
import logging

import torch
import torch.nn as nn
from rdkit import Chem

from src.models.bimamba import BiMambaForPropertyPrediction
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `BiMambaBlock` |
| Functions/methods | snake_case | `create_bimamba_model` |
| Constants | UPPER_SNAKE_CASE | `MAX_LENGTH = 512` |
| Variables | snake_case | `learning_rate` |
| Private | `_prefix` | `_validate_smiles` |
| Test functions | `test_` prefix | `test_model_forward_pass` |

### Function Design
- ≤50 lines per function (excl. docstring/imports)
- Single responsibility principle
- Default arguments after non-default parameters
- Use dataclasses for configuration objects

### Docstrings
Google style with Args, Returns, Raises:
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
    ...
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

### Logging
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
```

## ML-Specific Guidelines

### Data Pipeline
- Validate SMILES via `_validate_smiles()` before RDKit processing
- Cache tokenized sequences to disk
- Log dataset statistics (size, missing) at startup
- Normalize regression targets (z-score)
- Scaffold-split data to avoid data leakage

### Model Checkpoints
- Save `state_dict` only
- Filename: `{dataset}_bi_mamba_epoch{N}_valLoss{val_loss:.4f}.pt`
- Keep best model as `{dataset}_bi_mamba_best.pt`

### Training Techniques
- Linear warmup for first 5 epochs
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Mixed precision with `torch.cuda.amp.autocast()` (NVIDIA) or `torch.mps.amp.autocast()` (Apple)
- Auto-detect device: `cuda` > `mps` > `cpu`

### Device Management
```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
logger.info(f"Using device: {device}")
```

## Testing Guidelines

### Test Structure
Mirror package structure: `tests/test_{module}.py`
```python
def test_model_forward_pass():
    """Test model forward pass with random input."""
    model = create_bimamba_model(vocab_size=50, d_model=64)
    x = torch.randint(0, 50, (2, 10))
    logits, loss = model(x, labels=torch.randn(2))
    assert logits.shape == (2,)
```

### Assertion Best Practices
```python
# Good: explicit message
assert abs(pred - expected) < 1e-5, f"Expected ~{expected}, got {pred}"

# Bad: bare assert
assert abs(pred - expected) < 1e-5
```

## Troubleshooting
| Issue | Solution |
|-------|----------|
| NaN loss | Reduce learning rate, check gradients |
| OOM errors | Decrease batch size, enable gradient checkpointing |
| RDKit failures | Validate SMILES with `_validate_smiles()` |
| MPS errors | Use CPU for debugging |
| Podman GPU access | Ensure nvidia-container-toolkit installed |
| CUDA OOM | Reduce d_model or batch_size |

## Key Files
| File | Description |
|------|-------------|
| `src/models/bimamba.py` | Core BiMamba model implementation |
| `src/data/molecule_dataset.py` | Data loading & tokenization |
| `train.py` | Training entry point |
| `eval.py` | Evaluation script |
| `requirements.txt` | pip dependencies |
| `environment.yml` | conda dependencies |
