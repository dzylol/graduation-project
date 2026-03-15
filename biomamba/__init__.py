"""
Bi-Mamba-Chem: Bidirectional Mamba for molecular property prediction.

A PyTorch implementation of Bidirectional Mamba (Bi-Mamba) for
molecular property prediction on SMILES sequences.

Key Features:
- O(N) linear complexity vs Transformer's O(N²)
- Bidirectional scanning to capture chemical environment
- Support for both mamba-ssm and manual SSM implementations
- MoleculeNet datasets (ESOL, BBBP, ClinTox)
"""

__version__ = '0.1.0'

from .models import (
    BiMambaBlock,
    BiMambaEncoder,
    BiMambaModel,
    BiMambaForPrediction,
    BiMambaForSequenceClassification,
    BiMambaForRegression,
    SSMCore,
    SSMBlock,
    BidirectionalSSM,
    MambaBlock,
    MambaLayer,
)

from .data import (
    AtomTokenizer,
    MoleculeDataset,
    load_esol,
    load_bbbp,
    load_clintox,
    get_dataset,
    get_task_type,
)

from .utils import (
    compute_metrics,
    compute_regression_metrics,
    compute_classification_metrics,
    format_metrics,
    Logger,
    get_logger,
)

__all__ = [
    # Version
    '__version__',
    # Models
    'BiMambaBlock',
    'BiMambaEncoder',
    'BiMambaModel',
    'BiMambaForPrediction',
    'BiMambaForSequenceClassification',
    'BiMambaForRegression',
    'SSMCore',
    'SSMBlock',
    'BidirectionalSSM',
    'MambaBlock',
    'MambaLayer',
    # Data
    'AtomTokenizer',
    'MoleculeDataset',
    'load_esol',
    'load_bbbp',
    'load_clintox',
    'get_dataset',
    'get_task_type',
    # Utils
    'compute_metrics',
    'compute_regression_metrics',
    'compute_classification_metrics',
    'format_metrics',
    'Logger',
    'get_logger',
]
