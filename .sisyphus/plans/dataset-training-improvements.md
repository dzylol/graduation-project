# Plan: Bi-Mamba-Chem Dataset & Training Improvements

## TL;DR

> Download full MoleculeNet datasets via DeepChem, implement z-score normalization, fix zombie process risk, and update documentation.

**Deliverables**:
- `download_datasets.py` - DeepChem integration for full MoleculeNet datasets
- `src/data/molecule_dataset.py` - LabelNormalizer class for z-score normalization
- `train.py` - Signal handling to prevent zombie processes
- `AGENTS.md` (root + src/data/) - Updated documentation

**Estimated Effort**: Short
**Parallel Execution**: YES
**Critical Path**: download_datasets.py → molecule_dataset.py → train.py → AGENTS.md

---

## Context

### Original Problem
- `download_datasets.py` only generated tiny example data (50-60 samples per dataset)
- Z-score normalization documented in AGENTS.md but NOT implemented in code
- Training script had no signal handling for DataLoader worker cleanup

### Research Findings
- MoleculeNet: ESOL (1,128), BBBP (2,039), ClinTox (1,478), FreeSolv (642), Lipophilicity (4,200)
- DeepChem provides `dc.molnet.load_*` for all MoleculeNet datasets
- SMILES-Mamba (arXiv:2408.05696) uses two-stage training: ZINC 250K pretrain → downstream fine-tune
- `Tangshengku/Bi-Mamba` on GitHub is NOT molecular model - it's LLM 1-bit quantization

---

## Work Objectives

### Core Objective
Enhance dataset handling and training infrastructure for Bi-Mamba-Chem.

### Concrete Deliverables
- [x] `download_datasets.py` with DeepChem integration
- [x] `LabelNormalizer` class in `molecule_dataset.py`
- [x] `create_data_loaders()` returns 4-tuple `(train, val, test, normalizer)`
- [x] Signal handlers in `train.py` for zombie process prevention
- [x] Updated `AGENTS.md` (root and src/data/)

### Definition of Done
- [ ] All files pass Python syntax check (`python3 -m py_compile`)
- [ ] AGENTS.md accurately describes implemented features

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (pytest not available in environment)
- **Automated tests**: None
- **Verification**: Manual syntax check + import test

### QA Policy
No agent-executed QA scenarios (trivial changes, syntax verification only).

---

## TODOs

- [ ] 1. `download_datasets.py` - DeepChem Integration

  **What to do**:
  - Replace hardcoded example data with DeepChem API calls
  - Add `--dataset` flag for selecting specific datasets
  - Add `--zinc` flag for ZINC 250K pretraining data
  - Add `--example` flag to force example data fallback
  - Add `DatasetInfo` dataclass for metadata
  - Implement `check_deepchem_installed()`, `load_with_deepchem()`, `generate_example_data()`, `download_zinc_pretrain()`
  - Save metadata to `meta.json` per dataset

  **Must NOT do**:
  - Don't break existing CLI interface (`python download_datasets.py` still works)
  - Don't remove example data fallback

  **References**:
  - `pip install deepchem` for installation
  - `dc.molnet.load_esol()`, `dc.molnet.load_bbbp()`, etc. for dataset loading

  **Acceptance Criteria**:
  - [ ] `python3 -m py_compile download_datasets.py` → PASS
  - [ ] `python3 download_datasets.py --help` shows all new flags

- [ ] 2. `src/data/molecule_dataset.py` - Z-Score Normalization

  **What to do**:
  - Add `LabelNormalizer` class with `fit()`, `transform()`, `inverse_transform()`
  - Update `create_data_loaders()` to accept `normalize` parameter
  - Update `create_data_loaders()` return signature to 4-tuple

  **Must NOT do**:
  - Don't break existing code that expects 3-tuple return

  **References**:
  - `numpy.mean()`, `numpy.std()` for z-score computation
  - sklearn `StandardScaler` pattern (fit on train only)

  **Acceptance Criteria**:
  - [ ] `python3 -m py_compile src/data/molecule_dataset.py` → PASS
  - [ ] `LabelNormalizer` documented in src/data/AGENTS.md

- [ ] 3. `train.py` - Signal Handling for Zombie Process Prevention

  **What to do**:
  - Add imports: `signal`, `atexit`
  - Add `_cleanup()` function, `_signal_handler()`, `_interrupted` flag
  - Register signal handlers for SIGINT and SIGTERM
  - Register `atexit.register(_cleanup)`
  - Update `create_data_loaders()` call to handle 4-tuple return

  **Must NOT do**:
  - Don't break existing training workflow

  **References**:
  - `signal.signal(signal.SIGINT, handler)`
  - `atexit.register(cleanup_func)`

  **Acceptance Criteria**:
  - [ ] `python3 -m py_compile train.py` → PASS
  - [ ] Training script responds to Ctrl+C gracefully

- [ ] 4. `AGENTS.md` (root) - Documentation Update

  **What to do**:
  - Update Entry Points table: `download_datasets.py` description
  - Add dataset download commands to "Key Commands" section
  - Add MoleculeNet dataset table (ESOL, BBBP, ClinTox, FreeSolv, Lipophilicity)
  - Add "Pretraining (Two-Stage)" section for ZINC 250K

  **Acceptance Criteria**:
  - [ ] All commands in AGENTS.md are accurate
  - [ ] Dataset table matches available datasets

- [ ] 5. `src/data/AGENTS.md` - Documentation Update

  **What to do**:
  - Add `LabelNormalizer` to Key Classes table
  - Update Conventions to reflect `LabelNormalizer` usage
  - Update return signature for `create_data_loaders()`
  - Fix "Disk caching" mention (was pickle, now functools.lru_cache)

  **Acceptance Criteria**:
  - [ ] `LabelNormalizer` is documented
  - [ ] Return 4-tuple is documented

---

## Final Verification Wave

- [ ] F1. **Syntax Check** — `python3 -m py_compile` on all modified files
- [ ] F2. **Import Check** — `python3 -c "from src.data.molecule_dataset import..."` (requires numpy)
- [ ] F3. **Documentation Review** — AGENTS.md matches actual implementation

---

## Commit Strategy

- **1**: `feat(data): add DeepChem dataset download and z-score normalization`

---

## Success Criteria

```bash
python3 -m py_compile download_datasets.py   # PASS
python3 -m py_compile src/data/molecule_dataset.py  # PASS
python3 -m py_compile train.py              # PASS
```
