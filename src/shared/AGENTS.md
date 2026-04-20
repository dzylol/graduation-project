# AGENTS.md - src/shared/

**Generated:** 2026-04-20

**Training/eval argument parsing + utilities.** Argparse factories, device management, evaluation helpers.

## Structure
```
src/shared/
├── __init__.py
└── utils.py          # parse_train_args, parse_eval_args, set_seed, get_device, evaluate
```

## Key Functions

| Symbol | Type | Signature | Purpose |
|--------|------|-----------|---------|
| `parse_train_args` | factory | `() → Namespace` | CLI args for training |
| `parse_eval_args` | factory | `() → Namespace` | CLI args for evaluation |
| `set_seed` | fn | `(seed: int) → None` | Reproducibility |
| `get_device` | fn | `() → str` | `cuda > mps > cpu` |
| `evaluate` | fn | `(model, loader, device, args, normalizer?) → Dict` | Metrics computation |

## CLI Arguments (Training)

| Argument | Default | Purpose |
|----------|---------|---------|
| `--dataset` | required | Dataset name |
| `--model_type` | `manual` | `manual` or `mamba_ssm` |
| `--task_type` | `regression` | `regression` or `classification` |
| `--loss_type` | `mse` | `mse`, `smooth_l1`, `huber` |
| `--pooling` | `mean` | `mean`, `max`, `cls` |
| `--d_model` | 256 | Model dimension |
| `--n_layers` | 4 | BiMamba layers |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--device` | `auto` | `cuda`, `mps`, `cpu`, `auto` |
| `--max_length` | 512 | Max SMILES length |

## Device Order
```python
def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"
```

## Anti-Patterns
- **NEVER** hardcode device — use `get_device()` or `--device` arg
- **NEVER** use `torch.random` — use `set_seed()` for reproducibility
