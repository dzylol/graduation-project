# AGENTS.md - src/data/

**SMILES tokenization + molecular dataset handling.** CSV/JSON/TXT loading, RDKit validation, z-score normalization, SQLite database support.

## Structure
```
src/data/
└── molecule_dataset.py    # MoleculeTokenizer, MoleculeDataset, DatabaseMoleculeDataset, LabelNormalizer, create_data_loaders
```

## Key Classes

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `MoleculeTokenizer` | class | molecule_dataset.py | SMILES → token indices |
| `MoleculeDataset` | class | molecule_dataset.py | Torch Dataset from file (CSV/JSON/TXT) |
| `DatabaseMoleculeDataset` | class | molecule_dataset.py | Torch Dataset from SQLite database |
| `LabelNormalizer` | class | molecule_dataset.py | Z-score normalization for regression |
| `NormalizedDataset` | class | molecule_dataset.py | Dataset wrapper that applies z-score to labels |
| `create_data_loaders` | factory | molecule_dataset.py | Train/val/test DataLoaders from file or database |

## Token Vocabulary

**Atom tokens** (56): `( ) [ ] = # % 0-9 + - / . : < > @ B Br C Cl F H I N O P S Si Te Se At`
**Special tokens** (4): `<pad>`, `<unk>`, `<bos>`, `<eos>`

## Conventions (THIS MODULE)

- **ALWAYS validate SMILES** via `_validate_smiles()` before RDKit processing
- **Z-score normalization** via `LabelNormalizer` class (fit on train, apply to val/test)
- **Disk caching** of tokenized sequences via `functools.lru_cache`
- **max_length default**: 512 (configurable via `--max_length`)
- Dataclasses as DTOs (not bare dicts)
- Type hints required on all function signatures
- `create_data_loaders()` returns 4-tuple: `(train, val, test, normalizer)`

## Data Loading (Dual Mode)

### File Mode
```python
train_loader, val_loader, test_loader, normalizer = create_data_loaders(
    train_path="data/ESOL/train.csv",
    val_path="data/ESOL/val.csv",
)
```

### Database Mode
```python
from src.db.molecule_repo import MoleculeRepository
repo = MoleculeRepository()
repo.import_from_csv("dataset/ESOL/delaney.csv", dataset_name="ESOL")
train_loader, val_loader, test_loader, normalizer = create_data_loaders(
    train_dataset_name="ESOL",
    db_path="bi_mamba_chem.db",
    property_name="measured log(solubility:mol/L)",
)
```

## Data Format (CSV)

```csv
smiles,label
CCO,-2.5
CC(=O)OC,-1.8
c1ccccc1,3.2
```

## Multi-task Format

```csv
smiles,solubility,toxicity,logp
CCO,-2.5,0,1.3
```

## Anti-Patterns (THIS MODULE)

- **NEVER** pass raw SMILES to RDKit without `_validate_smiles()` check
- **NEVER** use bare dicts — use dataclasses
- **NEVER** skip z-score normalization for regression
