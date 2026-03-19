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
└── requirements.txt     # Runtime dependencies
```

## Build, Lint, Test
### Dependencies
- Install: `pip install -r requirements.txt`

### Testing
- All tests: `python -m pytest tests/ -v`
- Model tests: `python tests/test_model.py`
- Data tests: `python tests/test_data.py`
- Single test function: `python -c "from tests.test_data import test_tokenization; test_tokenization()"`
- Coverage: `python -m pytest tests/ --cov=src --cov-report=html`

## Code Style

### Imports
- Order: stdlib → third-party → local
- Blank line between groups
- Example:
  ```python
  import os
  import json
  import torch
  import torch.nn as nn
  from src.models.bimamba import BiMambaForPropertyPrediction
  ```

### Naming Conventions
- Classes: `PascalCase` (e.g., `BiMambaBlock`)
- Functions/methods: `snake_case` (e.g., `create_bimamba_model`, `validate_smiles`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_LENGTH = 512`)
- Variables: `snake_case` (e.g., `learning_rate`)
- Private: `_prefix` (e.g., `_validate_smiles`)
- Test functions: `test_{feature_description}`

### Function Design
- ≤50 lines per function (excl. docstring/imports)
- Single responsibility principle
- Default arguments after non-default parameters
- Use dataclasses for configuration objects

### Docstrings
- Google style with Args, Returns, Raises
- Concise and complete
- Example:
  ```python
  def predict_property(smiles: str) -> float:
      """Predict molecular property.

      Args:
          smiles: Molecule SMILES string

      Returns:
          float: Predicted property value

      Raises:
          ValueError: On invalid SMILES
      """
      ...
  ```

### Error Handling
- Specific exceptions (`ValueError`, `RuntimeError`, `FileNotFoundError`)
- Log error context before raising
- Validate inputs early
  ```python
  if not isinstance(batch_size, int) or batch_size <= 0:
      raise ValueError(f"batch_size must be positive int, got {batch_size}")
  ```

### Logging
- Root logger configured at startup
- Levels: DEBUG, INFO, WARNING, ERROR
  ```python
  logging.basicConfig(level=logging.INFO, format="%(message)s")
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
- Mixed precision with `torch.cuda.amp.autocast()`
- Auto-detect device: `cuda` > `mps` > `cpu`

### Device Management
- Log selected device on startup
- Use CPU for CI/debugging

## Testing Guidelines
### Test Structure
- Mirror package structure: `tests/test_{module}.py`
- Each test targets one behavior
- Use fixtures for reusable objects
  ```python
  @pytest.fixture
  def model():
      return create_bimamba_model(vocab_size=50, d_model=64)

  def test_forward_pass(model):
      x = torch.randn(2, 10)
      logits = model(x)
      assert logits.shape == (2,)
  ```

### Test Naming
- Prefix `test_`
- Use underscores, not camelCase
  - `test_model_forward_pass`
  - `test_data_loading_invalid_smiles`

### Assertion Best Practices
- Explicit `assert <cond>, "msg"` over bare asserts
  ```python
  assert abs(pred - expected) < 1e-5, f"Expected ~{expected}, got {pred}"
  ```

## Common Practices
- Validate SMILES before RDKit processing
- Cache expensive computations
- Log dataset split sizes (train/val/test)
- Use context managers for temporary files
- Keep secrets out of version control

## Troubleshooting
| Issue | Solution |
|-------|----------|
| NaN loss | Reduce learning rate, check gradients |
| OOM errors | Decrease batch size, enable checkpointing |
| RDKit failures | Validate SMILES with `_validate_smiles()` |
| MPS errors | Use CPU for debugging |

## Key Files
- `src/models/bimamba.py` - Core BiMamba model implementation
- `src/data/molecule_dataset.py` - Data loading & tokenization
- `train.py` - Training entry point
- `eval.py` - Evaluation script

*End of guidelines (~150 lines)*
