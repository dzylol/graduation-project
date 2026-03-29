# AGENTS.md - src/data/

**SMILES tokenization + molecular dataset handling.** CSV/JSON/TXT loading, RDKit validation, z-score normalization.

## Structure
```
src/data/
├── molecule_dataset.py    # MoleculeTokenizer, MoleculeDataset, LabelNormalizer, create_data_loaders
└── database/           # (empty - do not use)
```

## Key Classes

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `MoleculeTokenizer` | class | molecule_dataset.py | SMILES → token indices |
| `MoleculeDataset` | class | molecule_dataset.py | Torch Dataset wrapper |
| `LabelNormalizer` | class | molecule_dataset.py | Z-score normalization for regression |
| `NormalizedDataset` | class | molecule_dataset.py | Dataset wrapper that applies z-score to labels |
| `create_data_loaders` | factory | molecule_dataset.py | Train/val/test DataLoaders + normalizer |

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
