"""
Configuration file for Bi-Mamba-Chem.
Contains default hyperparameters and settings.
"""

# Model configuration
MODEL_CONFIG = {
    'd_model': 256,           # Model dimension
    'n_layers': 4,             # Number of layers
    'd_state': 128,            # SSM state dimension
    'd_conv': 4,               # Convolution kernel size
    'expand': 2,               # Expansion factor
    'dropout': 0.1,           # Dropout rate
    'norm_eps': 1e-5,         # LayerNorm epsilon
    'max_len': 512,           # Maximum sequence length
    'fusion': 'gate',          # Fusion strategy: 'concat', 'add', 'gate'
    'pool_type': 'mean',       # Pooling type: 'mean', 'max', 'cls'
}

# Training configuration
TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 100,
    'grad_clip': 1.0,
}

# Data configuration
DATA_CONFIG = {
    'max_length': 128,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
}

# Dataset information
DATASETS = {
    'ESOL': {
        'task_type': 'regression',
        'description': 'Aqueous solubility',
        'metric': 'RMSE',
    },
    'BBBP': {
        'task_type': 'classification',
        'description': 'Blood-brain barrier penetration',
        'metric': 'ROC-AUC',
    },
    'CLINTOX': {
        'task_type': 'classification',
        'description': 'Clinical trial toxicity',
        'metric': 'ROC-AUC',
    },
}


def get_model_config(model_name: str = 'bi_mamba'):
    """
    Get model configuration.

    Args:
        model_name: Model name

    Returns:
        Model configuration dictionary
    """
    return MODEL_CONFIG.copy()


def get_train_config():
    """
    Get training configuration.

    Returns:
        Training configuration dictionary
    """
    return TRAIN_CONFIG.copy()


def get_dataset_info(dataset_name: str):
    """
    Get dataset information.

    Args:
        dataset_name: Dataset name

    Returns:
        Dataset information dictionary
    """
    if dataset_name.upper() not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASETS[dataset_name.upper()]


if __name__ == "__main__":
    # Print configuration
    print("Model Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")

    print("\nTraining Configuration:")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")

    print("\nDataset Information:")
    for name, info in DATASETS.items():
        print(f"  {name}: {info}")
