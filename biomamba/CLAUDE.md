# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bi-Mamba-Chem is a PyTorch implementation of Bidirectional Mamba for molecular property prediction on SMILES sequences. It leverages State Space Models (SSM) with O(N) linear complexity compared to Transformer's O(N²).

**Research Goal:** Build a molecular property prediction model using Mamba architecture with bidirectional scanning to capture chemical environment from both directions.

## Common Commands

```bash
# Install dependencies
cd biomamba
pip install -r requirements.txt

# Run training (manual SSM - default, no mamba-ssm required)
python3 train.py --dataset ESOL --epochs 100 --batch_size 32

# Run training with mamba-ssm (if installed)
python3 train.py --dataset ESOL --epochs 100 --use_ssm

# Run evaluation
python3 eval.py --checkpoint checkpoints/ESOL_bi_mamba_best.pt --dataset ESOL
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset (ESOL, BBBP, CLINTOX) | ESOL |
| `--use_ssm` | Use manual SSM instead of mamba-ssm | False |
| `--d_model` | Model dimension | 256 |
| `--n_layers` | Number of layers | 4 |
| `--fusion` | Bidirectional fusion (concat/add/gate) | gate |
| `--pool_type` | Pooling (mean/max/cls) | mean |

## Architecture

The project implements a bidirectional Mamba encoder for SMILES sequences:

1. **Tokenization** (`data/tokenizer.py`): Atom-level SMILES tokenization handling Cl, Br, ring closures (#, %), and bracket atoms ([N], [O], etc.)

2. **SSM Core** (`models/ssm_core.py`): Manual SSM implementation with selective scan - used by default. Falls back to mamba-ssm if available.

3. **Bidirectional Mamba** (`models/bi_mamba.py`): Forward + backward branches with gated fusion to capture chemical environment from both directions

4. **Prediction Head** (`models/predictor.py`): Global pooling (mean/max/cls) + MLP for regression (MSE) or classification (BCE)

## Testing

```bash
# Test tokenizer
python3 -c "from data.tokenizer import AtomTokenizer; t = AtomTokenizer(); print(t.tokenize('CCO'))"

# Test dataset
python3 -c "from data.dataset import get_dataset; train,_,_,_ = get_dataset('ESOL'); print(len(train))"

# Test model
python3 -c "from models.ssm_core import BidirectionalSSM; import torch; m = BidirectionalSSM(64); print(m(torch.randn(2,32,64)).shape)"
```

Note: Some environments require `KMP_DUPLICATE_LIB_OK=TRUE` to avoid OpenMP conflicts.

## Data

- **ESOL**: Aqueous solubility (regression, RMSE metric)
- **BBBP**: Blood-brain barrier penetration (classification, ROC-AUC)
- **ClinTox**: Clinical trial toxicity (classification, ROC-AUC)

The datasets are embedded in `data/dataset.py` with small sample sets for demonstration.
