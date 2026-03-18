# AGENTS.md - Bi-Mamba-Chem Codebase Guidelines

## Project Overview
Bi-Mamba-Chem implements a bidirectional Mamba architecture for molecular property prediction. 
Standard Python layout with src/, tests/, data/, checkpoints/, scripts/, configs/. 
Emphasis on reproducibility and modularity.

## Directory Layout
├── src/
│   ├── models/
││   │   └── bimamba.py          # Core model
│   └── data/
│       └── molecule_dataset.py # Data loading & tokenization
├── tests/
│   ├── test_model.py           # Model unit tests
│   └── test_data.py            # Data processing tests
├── checkpoints/                # Saved model checkpoints
├── scripts/
│   ├── train.py                # Training entry point
│   └── evaluate.py             # Evaluation script
├── configs/                    # Config files (YAML/JSON)
└── requirements.txt            # Runtime dependencies

## Build, Lint, Test
### Dependencies
- Runtime: `pip install -r requirements.txt`
- Development: `pip install -r requirements-dev.txt --upgrade`
- Pre‑commit: `pre-commit install`

### Formatting
- Check: `ruff check src/ --show-source`
- Fix: `ruff check src/ --fix`
- Format: `black src/`
- Combined: `ruff check src/ && black src/`

### Testing
- All tests: `python -m pytest tests/ -v`
- Single test: `python -m pytest tests/test_model.py::test_predict -v`
- Specific test function: `python -c "from tests.test_data import test_tokenization; test_tokenization()"`
- Coverage: `python -m pytest tests/ --cov=src --cov-report=html`

### CI Triggers
- Push to main/merge request
- Nightly full suite with coverage
- Release tag execution

### Pre‑commit Hooks
- Lint & format: `pre-commit run --all-files`
- Type check: `mypy src/`
- Verify EOL: `pre-commit run end-of-file`

## Code Style

### Imports
- Absolute only: `from src.models import BiMamba`
- Order: stdlib → third‑party → local
- Blank line between groups
- Example:
  ```python
  import os
  import json
  import torch
  import torch.nn as nn
  from src.models.bimamba import BiMambaForPropertyPrediction
  ```

### Naming
- Classes: `PascalCase` (e.g., `BiMambaBlock`)
- Functions/methods: `snake_case` (e.g., `create_bimamba_model`, `validate_smiles`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_LENGTH = 512`)
- Variables: `snake_case` (e.g., `learning_rate`)
- Private: `_prefix` (e.g., `_validate_smiles`)
- Avoid single‑letter names except loops
- Test functions: `test_{feature}`

### Function Design
- ≤30 lines per function (excl. docstring/imports)
- Single responsibility principle
- Default arguments after non‑default parameters
- Limit arguments; use configs/dataclasses
- Favor pure functions when possible

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
- Structured JSON in production, readable in dev
- Levels: DEBUG, INFO, WARNING, ERROR
  ```python
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  logger = logging.getLogger(__name__)
  ```

## ML‑Specific
### Data Pipeline
- Validate SMILES via `_validate_smiles()` before tokenization
- Cache tokenized sequences to disk with SHA256 filenames
- Log dataset statistics (size, missing) at startup
- Normalize regression targets (z‑score)
- Scaffold‑split data to avoid leakage

### Model Checkpoints
- Save `state_dict` only
- Filename: `{dataset}_bi_mamba_epoch{N}_valLoss{val_loss:.4f}.pt`
- Keep best model as `{dataset}_bi_mamba_best.pt`
- Upload validated checkpoints to cloud storage

### Training Techniques
- Linear warmup for first 5 epochs
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Mixed precision with `torch.cuda.amp.autocast()`
- Distributed training via `torchrun`
- Mixed Stochastic Weighting (MSW) for stability

### Device Management
- Auto‑detect device: `cuda` > `mps` > `cpu`
- Log selected device on start
- Use CPU for CI/debugging

## Testing Guidelines
### Test Structure
- Mirror package structure: `tests/test_{module}.py`
- Each test targets one behavior
- Mock external services with `unittest.mock` or `pytest-mock`
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
- Describe behavior clearly
- Use underscores, not camelCase
  - `test_model_forward_pass`
  - `test_data_loading_invalid_smiles`

### Assertion Best Practices
- Prefer explicit `assert <cond>, "msg"` over bare asserts
- Include context in failure messages
  ```python
  assert abs(pred - expected) < 1e-5, f"Expected ~{expected}, got {pred}"
  ```

## Common Practices
- Validate SMILES before RDKit processing
- Cache expensive computations
- Log dataset split sizes (train/val/test)
- Use context managers for temporary files
- Keep secrets out of version control
- Write idempotent data scripts

## CI/CD Integration
### Pre‑commit Example
- Lint: `ruff`
- Format: `black`
- Type check: `mypy src/`
- Test with coverage >80%

### Deployment Checklist
- Verify full test suite passes
- Measure inference memory footprint
- Confirm model size constraints
- Validate preprocessing pipeline
- Ensure reproducibility of results

## Troubleshooting & FAQ
| Issue | Solution |
|-------|----------|
| NaN loss | Reduce learning rate, check gradients |
| OOM errors | Decrease batch size, enable checkpointing |
| RDKit failures | Validate SMILES with `_validate_smiles()` |
| MPS errors | Use CPU for debugging |
| Flaky tests | Increase timeout, fix nondeterminism |

## Appendix
- VS Code shortcuts:
  - Run test: `Ctrl+Alt+R`
  - Debug: `F5`
  - Open terminal: ``Ctrl+` ``
  - Aliases:
    ```bash
    alias tf='python -m pytest tests/ -v'
    alias tfs='python -m pytest tests/ --cov=src -v'
    aliasfix='black src/'
    aliascheck='ruff check src/'
    ```
- Recommended extensions: Python, Jupyter, Pylance, Rainbow CSV
- Key config files:
  - `.pre-commit-config.yaml`
  - `pyproject.toml`
  - `.github/workflows/ci.yml`
  - `Dockerfile`
  - `.readthedocs.yml`

*End of guidelines (~150 lines)*