"""
Molecular dataset handling module.

This module provides functionality for:
1. Loading molecular data (CSV, JSON, TXT formats)
2. Tokenizing SMILES strings
3. Validating molecular structures with RDKit
4. Creating PyTorch data loaders
"""

# ============================================================================
# Imports
# ============================================================================
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import os
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# RDKit for chemical structure validation
from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================================
# MoleculeDataset class
# ============================================================================
class MoleculeDataset(Dataset):
    """
    Dataset for loading and tokenizing molecular data.

    Provides functionality for:
    - Loading CSV/JSON/TXT files with SMILES strings and labels
    - Validating SMILES strings using RDKit
    - Tokenizing SMILES into fixed-length token IDs
    - Returning PyTorch tensors for model input
    """

    def __init__(
        self,
        data_path: str,
        task_type: str = "regression",
        max_length: int = 512,
        tokenizer: Optional[object] = None,
        cache_smiles: bool = True,
        validate_smiles: bool = True,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_path: Path to data file (CSV, JSON, or TXT)
            task_type: Type of task ("regression" or "classification")
            max_length: Maximum sequence length (longer sequences are truncated)
            tokenizer: Custom tokenizer (defaults to internal tokenizer)
            cache_smiles: Cache processed SMILES for faster access
            validate_smiles: Validate SMILES strings using RDKit
        """
        self.task_type = task_type
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache_smiles = cache_smiles
        self.validate_smiles = validate_smiles

        # Initialize vocabulary
        self.vocab = self._init_vocab()
        self.vocab_size = len(self.vocab)

        # Load data
        self.data = self._load_data(data_path)

        # Initialize cache for processed SMILES (always a dict)
        self.smiles_cache: Dict[str, List[int]] = {} if cache_smiles else {}

    def _init_vocab(self) -> Dict[str, int]:
        """
        Initialize vocabulary mapping characters to IDs.

        Returns:
            Vocabulary dictionary {token: id}
        """
        # Common SMILES characters
        chars: List[str] = [
            "(",  # Left parenthesis
            ")",  # Right parenthesis
            "[",  # Left bracket
            "]",  # Right bracket
            "=",  # Double bond
            "#",  # Triple bond
            "%",  # Ring opening
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",  # Digits
            "+",  # Positive charge
            "-",  # Negative charge/single bond
            "/",  # Steric information
            ".",  # Bond breakage
            ":",  # Aromatic bond
            ";",  # Stereo information
            "<",  # Start of special ring
            ">",  # End of special ring
            "@",  # Stereo notation
            # Element symbols
            "B",
            "Br",
            "C",
            "Cl",
            "F",
            "H",
            "I",
            "N",
            "O",
            "P",
            "S",
            "Si",
            "Te",
            "Se",
            "At",
        ]

        # Special tokens
        special_tokens: List[str] = ["<pad>", ">", "<bos>", "<eos>"]

        # Build vocabulary
        vocab: Dict[str, int] = {token: idx for idx, token in enumerate(special_tokens)}
        vocab.update(
            {char: idx + len(special_tokens) for idx, char in enumerate(chars)}
        )

        return vocab

    def _load_data(self, data_path: str) -> List[dict]:
        """
        Load data from CSV, JSON, or TXT files.

        Args:
            data_path: Path to data file

        Returns:
            List of dictionaries with "smiles" and "labels" keys
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data: List[dict] = []

        # Load based on file extension
        if data_path.endswith(".csv"):
            # CSV format: first column is SMILES, remaining columns are labels
            df = pd.read_csv(data_path)
            smiles_col = df.columns[0]
            label_cols = df.columns[1:] if len(df.columns) > 1 else []

            for _, row in df.iterrows():
                smiles = str(row[smiles_col])
                if len(label_cols) > 0:
                    labels = [float(row[col]) for col in label_cols]
                else:
                    labels = [0.0]
                data.append({"smiles": smiles, "labels": labels})

        elif data_path.endswith(".json"):
            # JSON format: list of {"smiles": "...", "labels": [...]} objects
            with open(data_path, "r") as f:
                data = json.load(f)

        elif data_path.endswith(".txt"):
            # TXT format: each line "SMILES,label1,label2,..."
            with open(data_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        smiles = parts[0]
                        labels = [float(x) for x in parts[1:]]
                        data.append({"smiles": smiles, "labels": labels})
                    else:
                        data.append({"smiles": parts[0], "labels": [0.0]})

        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Validate SMILES strings if enabled
        if self.validate_smiles:
            original_len = len(data)
            data = [item for item in data if self._validate_smiles(item["smiles"])]
            if len(data) < original_len:
                print(f"Filtered out {original_len - len(data)} invalid SMILES strings")

        return data

    def _validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string using RDKit.

        Args:
            smiles: SMILES string

        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def _tokenize_smiles(self, smiles: str) -> List[int]:
        """
        Convert a SMILES string to a list of token IDs.

        Args:
            smiles: SMILES string

        Returns:
            List of token IDs
        """
        # Check cache first
        if self.cache_smiles and smiles in self.smiles_cache:
            return self.smiles_cache[smiles]

        tokens: List[int] = []
        i: int = 0

        # Tokenize SMILES string
        while i < len(smiles):
            # Try matching double-character tokens first (e.g., "Br", "Cl")
            if i + 1 < len(smiles) and smiles[i : i + 2] in self.vocab:
                tokens.append(self.vocab[smiles[i : i + 2]])
                i += 2
            # Then match single-character tokens
            elif smiles[i] in self.vocab:
                tokens.append(self.vocab[smiles[i]])
                i += 1
            # Handle unknown characters
            else:
                tokens.append(self.vocab["<pad>"])
                i += 1

        # Handle sequence length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            pad_token: int = self.vocab["<pad>"]
            tokens = tokens + [pad_token] * (self.max_length - len(tokens))

        # Update cache if enabled
        if self.cache_smiles:
            self.smiles_cache[smiles] = tokens

        return tokens

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieve a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_ids, labels) tensors
        """
        item = self.data[idx]
        smiles = item["smiles"]
        labels = item["labels"]

        # Tokenize SMILES
        token_ids = self._tokenize_smiles(smiles)

        # Convert to PyTorch tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        if self.task_type == "regression":
            labels_tensor = torch.tensor(labels, dtype=torch.float)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels_tensor

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def get_pad_token_id(self) -> int:
        """Return the ID of the padding token."""
        return self.vocab["<pad>"]


# ============================================================================
# Data loader creation function
# ============================================================================
def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    task_type: str = "regression",
    max_length: int = 512,
    num_workers: int = 4,
    tokenizer: Optional[object] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        batch_size: Number of samples per batch
        task_type: Task type ("regression" or "classification")
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading
        tokenizer: Custom tokenizer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create training dataset and loader
    train_dataset = MoleculeDataset(
        data_path=train_path,
        task_type=task_type,
        max_length=max_length,
        tokenizer=tokenizer,
    )

    # Validation dataset (optional)
    val_dataset: Optional[MoleculeDataset] = None
    if val_path and os.path.exists(val_path):
        val_dataset = MoleculeDataset(
            data_path=val_path,
            task_type=task_type,
            max_length=max_length,
            tokenizer=tokenizer,
        )

    # Test dataset (optional)
    test_dataset: Optional[MoleculeDataset] = None
    if test_path and os.path.exists(test_path):
        test_dataset = MoleculeDataset(
            data_path=test_path,
            task_type=task_type,
            max_length=max_length,
            tokenizer=tokenizer,
        )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader: Optional[torch.utils.data.DataLoader] = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader: Optional[torch.utils.data.DataLoader] = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


# ============================================================================
# Collate function for dynamic padding
# ============================================================================
def collate_fn(batch):
    """
    Custom collate function for handling variable-length sequences.

    Args:
        batch: List of (input_ids, labels) tuples

    Returns:
        Tuple of (input_ids_batch, labels_batch)
    """
    input_ids, labels = zip(*batch)

    # Pad sequences to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )

    # Stack labels
    labels = torch.stack(labels, dim=0)

    return input_ids, labels


# ============================================================================
# Independent tokenizer class
# ============================================================================
class MoleculeTokenizer:
    """
    Independent SMILES tokenizer with encode/decode methods.
    """

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize the tokenizer.

        Args:
            vocab_dict: Optional vocabulary dictionary (token string -> ID)
        """
        if vocab_dict is None:
            # Default vocabulary
            chars: List[str] = [
                "(",
                ")",
                "[",
                "]",
                "=",
                "#",
                "%",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "+",
                "-",
                "/",
                ".",
                ":",
                ";",
                "<",
                ">",
                "@",
                "B",
                "Br",
                "C",
                "Cl",
                "F",
                "H",
                "I",
                "N",
                "O",
                "P",
                "S",
                "Si",
                "Te",
                "Se",
                "At",
            ]
            special_tokens: List[str] = ["<pad>", ">", "<bos>", "<eos>"]

            vocab_dict: Dict[str, int] = {
                token: idx for idx, token in enumerate(special_tokens)
            }
            vocab_dict.update(
                {char: idx + len(special_tokens) for idx, char in enumerate(chars)}
            )

        # Vocabulary maps token string -> integer ID
        self.vocab: Dict[str, int] = vocab_dict
        # Inverse vocabulary maps integer ID -> token string
        self.inverse_vocab: Dict[int, str] = {
            idx: token for token, idx in vocab_dict.items()
        }
        self.vocab_size: int = len(vocab_dict)

    def encode(self, smiles: str, max_length: int = 512) -> List[int]:
        """
        Encode a SMILES string into token IDs.

        Args:
            smiles: SMILES string
            max_length: Maximum length of output sequence

        Returns:
            List of token IDs
        """
        tokens: List[int] = []
        i: int = 0
        while i < len(smiles):
            # Try matching double-character tokens first (e.g., "Br", "Cl")
            if i + 1 < len(smiles) and smiles[i : i + 2] in self.vocab:
                tokens.append(self.vocab[smiles[i : i + 2]])
                i += 2
            # Then match single-character tokens
            elif smiles[i] in self.vocab:
                tokens.append(self.vocab[smiles[i]])
                i += 1
            # Handle unknown characters
            else:
                tokens.append(self.vocab["<pad>"])
                i += 1

        # Handle sequence length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            pad_token_id: int = self.vocab["<pad>"]
            tokens = tokens + [pad_token_id] * (max_length - len(tokens))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to a SMILES string.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded SMILES string
        """
        tokens: List[str] = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token: str = self.inverse_vocab[token_id]
                # Skip special tokens during decoding
                if token not in ["<pad>", ">", "<bos>", "<eos>"]:
                    tokens.append(token)
            else:
                tokens.append("")
        return "".join(tokens)
