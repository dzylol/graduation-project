# Tests AGENTS.md - BiMamba-Chem

**Generated:** 2026-03-26

## OVERVIEW

Test suite for BiMamba molecular property prediction models.

## Structure

| File | Lines | Purpose |
|------|-------|---------|
| `test_model.py` | 238 | BiMambaForPropertyPrediction forward pass, gradient, state dict |
| `test_data.py` | 289 | SMILES tokenization, dataset collation, data loader creation |

## Key Test Functions

| Function | File | Description |
|----------|------|-------------|
| `test_tokenization` | test_data.py | SMILES to token ID conversion |
| `test_dataset_andataloader` | test_data.py | Dataset and DataLoader creation |
| `test_collate_fn` | test_data.py | Batch padding and truncation |
| `test_forward_pass` | test_model.py | Model forward pass with random input |
| `test_backward_pass` | test_model.py | Gradient computation |
| `test_state_dict` | test_model.py | Model save/load roundtrip |

## Conventions

**Dual-Mode Execution:** Each test runs under two modes:
1. `pytest tests/test_model.py -v` — pytest runner
2. `python tests/test_model.py` — standalone execution

**No conftest.py:** No shared fixtures or pytest configuration. Each test is self-contained.

**No pytest in requirements.txt:** Install separately with `pip install pytest pytest-cov`

**Step-by-Step Comments:** Tests use sequential Chinese comments:
```
[测试 1] 验证 tokenization 返回正确形状
[步骤 1] 输入 "C" 生成 token IDs
[步骤 2] 验证长度等于预期
```

**Manual Cleanup:** Tests use try/finally for cleanup, not pytest fixtures.

**Real Data via tempfile:** Tests create real molecule data in temporary files instead of mocks.

## Anti-Patterns

- DO NOT add conftest.py with shared fixtures
- DO NOT use unittest.TestCase classes
- DO NOT mock RDKit or torch tensors
- DO NOT run tests without verifying both execution modes

## Commands

```bash
# All tests with pytest
python -m pytest tests/ -v

# Single test file
python -m pytest tests/test_model.py -v

# Single test function
python -m pytest tests/test_data.py::test_tokenization -v

# Standalone execution (no pytest)
python tests/test_model.py

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```
