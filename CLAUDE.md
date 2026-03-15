# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a graduation project implementing **Bi-Mamba-Chem** (Bidirectional Mamba for molecular property prediction). The project explores using State Space Models (SSM) for molecular property prediction, with a focus on efficiency advantages over Transformer models.

**Research Goal:** Build a molecular property prediction model using Mamba architecture with bidirectional scanning to capture chemical environment information from both directions.

**Key Innovation:** Bi-Mamba architecture with O(N) linear complexity vs Transformer's O(N²) for long molecular sequences.

---

## Tech Stack

- **Framework:** PyTorch
- **Environment:** Python with conda/pip
- **Mamba:** Manual SSM implementation (default) or `mamba-ssm` package
- **Data:** MoleculeNet datasets (ESOL, BBBP, ClinTox)
- **Evaluation:** RDKit for molecular fingerprint computation

---

## Common Commands

```bash
# Required environment variable for Mac (avoids OpenMP conflicts)
export KMP_DUPLICATE_LIB_OK=TRUE

# Navigate to project directory
cd biomamba

# Run training with interactive device selection
python train.py --dataset ESOL --epochs 100

# Run training with explicit device (mps/cuda/cpu)
python train.py --dataset ESOL --device mps --epochs 100

# Training with performance options
python train.py --dataset ESOL --batch_size 64 --num_workers 4 --gradient_accumulation_steps 2

# Run evaluation
python eval.py --checkpoint checkpoints/ESOL_bi_mamba_best.pt --dataset ESOL
```

---

## Architecture

The project implements:

1. **Data Preprocessing:** SMILES tokenization (atom-level for elements like Cl, Br)
2. **Embedding Layer:** Token embeddings + positional embeddings
3. **Bi-Mamba Encoder:** Forward + backward branches with fusion (concat/add/gated)
4. **Prediction Head:** Global pooling (mean/max/[CLS]) + task-specific output layer
5. **Loss Functions:** MSE for regression, BCE for classification

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

## Device Support

- **Apple Silicon (M1/M2/M3)**: Uses MPS backend - auto-detected
- **NVIDIA GPU**: Uses CUDA - auto-detected
- **CPU**: Fallback option

---

## Datasets

- **ESOL:** Solubility (regression)
- **BBBP:** Blood-brain barrier penetration (binary classification)
- **ClinTox:** Toxicity prediction (binary classification)
- **Long sequences:** For demonstrating Mamba's efficiency advantage over Transformers
