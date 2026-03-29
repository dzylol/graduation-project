# Plan: Fix Z-Score Normalization in Training Loop

## Problem

The `LabelNormalizer` class exists but is NOT actually applied to labels during training.

**Current state:**
- `create_data_loaders()` returns `(train_loader, val_loader, test_loader, normalizer)`
- `normalizer.fit()` is called on training labels
- But labels are NOT transformed — they go directly to the model unchanged

**Impact:**
- Z-score normalization is documented but not actually used
- Regression tasks don't benefit from normalized targets

---

## Root Cause

The `MoleculeDataset.__getitem__` returns raw labels. There's no normalization applied.

---

## Solution: NormalizedDataset Wrapper

Add a `NormalizedDataset` class that wraps the base dataset and applies z-score transform to labels.

### Step 1: Update `src/data/molecule_dataset.py`

Add wrapper class:

```python
class NormalizedDataset(Dataset):
    """Wraps MoleculeDataset to apply z-score normalization to labels."""

    def __init__(
        self,
        base_dataset: MoleculeDataset,
        normalizer: LabelNormalizer,
    ) -> None:
        self.base_dataset = base_dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        input_ids, labels = self.base_dataset[idx]
        normalized_labels = self.normalizer.transform(labels.numpy())
        return input_ids, torch.tensor(normalized_labels, dtype=torch.float)
```

### Step 2: Update `create_data_loaders()`

When `normalize=True` and `task_type="regression"`:
- Wrap train dataset with `NormalizedDataset`
- Apply normalizer fit before wrapping
- Val/test datasets also get wrapped with same normalizer

### Step 3: Update `train.py`

- Pass normalizer through training functions (or use closure)
- For evaluation metrics in original scale, use `normalizer.inverse_transform()`

---

## Files to Modify

| File | Change |
|------|--------|
| `src/data/molecule_dataset.py` | Add `NormalizedDataset` class, update `create_data_loaders()` |
| `train.py` | Update `evaluate()` to accept optional normalizer for denormalized metrics |

---

## Verification

```bash
python3 -m py_compile src/data/molecule_dataset.py
python3 -m py_compile train.py
```

---

## TODO

- [ ] 1. Add `NormalizedDataset` wrapper class to `molecule_dataset.py`
- [ ] 2. Update `create_data_loaders()` to wrap datasets when normalizing
- [ ] 3. Update `train.py` `evaluate()` to support denormalized metrics
- [ ] 4. Syntax check
