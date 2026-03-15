# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bi-Mamba-Chem is a PyTorch implementation of Bidirectional Mamba for molecular property prediction on SMILES sequences. It leverages State Space Models (SSM) with O(N) linear complexity compared to Transformer's O(N²).

**Research Goal:** Build a molecular property prediction model using Mamba architecture with bidirectional scanning to capture chemical environment from both directions.

## Common Commands

```bash
# Required environment variable for Mac (avoids OpenMP conflicts)
export KMP_DUPLICATE_LIB_OK=TRUE

# Training with interactive device selection (auto-detects MPS/CUDA/CPU)
python train.py --dataset ESOL --epochs 100

# Training with explicit device
python train.py --dataset ESOL --device mps --epochs 100
python train.py --dataset ESOL --device cuda --epochs 100
python train.py --dataset ESOL --device cpu --epochs 100

# Training with performance options
python train.py --dataset ESOL --batch_size 64 --num_workers 4 --gradient_accumulation_steps 2

# Run evaluation
python eval.py --checkpoint checkpoints/ESOL_bi_mamba_best.pt --dataset ESOL
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset (ESOL, BBBP, CLINTOX) | ESOL |
| `--device` | Device (auto/mps/cuda/cpu) | auto |
| `--batch_size` | Batch size | 64 |
| `--num_workers` | DataLoader workers | 4 |
| `--gradient_accumulation_steps` | Gradient accumulation | 1 |
| `--d_model` | Model dimension | 256 |
| `--n_layers` | Number of layers | 4 |
| `--fusion` | Bidirectional fusion (concat/add/gate) | gate |
| `--pool_type` | Pooling (mean/max/cls) | mean |
| `--lr` | Learning rate | 1e-3 |

## Architecture

The project implements a bidirectional Mamba encoder for SMILES sequences:

1. **Tokenization** (`data/tokenizer.py`): Atom-level SMILES tokenization handling Cl, Br, ring closures (#, %), and bracket atoms ([N], [O], etc.)

2. **SSM Core** (`models/ssm_core.py`): Manual SSM implementation with selective scan - used by default. Falls back to mamba-ssm if available.

3. **Bidirectional Mamba** (`models/bi_mamba.py`): Forward + backward branches with gated fusion to capture chemical environment from both directions

4. **Prediction Head** (`models/predictor.py`): Global pooling (mean/max/cls) + MLP for regression (MSE) or classification (BCE)

## Device Support

- **Apple Silicon (M1/M2/M3)**: Uses MPS backend - auto-detected
- **NVIDIA GPU**: Uses CUDA - auto-detected
- **CPU**: Fallback option

When running `train.py` or `eval.py` with `--device auto`, the script will detect available devices and prompt user to select.

## Data

- **ESOL**: Aqueous solubility (regression, RMSE metric)
- **BBBP**: Blood-brain barrier penetration (classification, ROC-AUC)
- **ClinTox**: Clinical trial toxicity (classification, ROC-AUC)

The datasets are embedded in `data/dataset.py` with small sample sets for demonstration.
