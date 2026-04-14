"""Dataset splitting functions (scaffold + random splits)."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem


_SPLIT_SEED_FILE = ".split_seed"


def get_next_split_seed() -> int:
    """Get and increment next split seed."""
    try:
        with open(_SPLIT_SEED_FILE, "r") as f:
            seed = int(f.read().strip())
    except FileNotFoundError:
        seed = 42

    with open(_SPLIT_SEED_FILE, "w") as f:
        f.write(str(seed + 1))

    return seed


def get_current_split_seed() -> int:
    """Get current split seed without incrementing."""
    try:
        with open(_SPLIT_SEED_FILE, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 42


def random_split_dataset(
    input_csv: str,
    output_dir: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split CSV dataset into train/val/test.

    Args:
        input_csv: Input CSV file path
        output_dir: Output directory (optional, saves files if specified)
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.1)
        seed: Random seed (default None uses numpy default)
        n_jobs: Number of threads for CSV reading.
                Default None uses os.cpu_count() or 4.

    Returns:
        (train_df, val_df, test_df) tuple of DataFrames

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    if n_jobs is None:
        n_jobs = os.cpu_count() or 4

    from concurrent.futures import ThreadPoolExecutor

    def read_chunk(args):
        start, end = args
        return pd.read_csv(
            input_csv,
            skiprows=range(1, start + 1) if start > 0 else None,
            nrows=end - start,
        )

    total_lines = sum(1 for _ in open(input_csv)) - 1
    chunk_size = max(1, total_lines // n_jobs)
    chunks = []
    for i in range(n_jobs):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_lines) if i < n_jobs - 1 else total_lines
        if start < total_lines:
            chunks.append((start, end))

    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(read_chunk, chunks))
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.read_csv(input_csv)

    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    relative_val_ratio = val_ratio / val_test_ratio if val_test_ratio > 0 else 0.5
    val_df, test_df = train_test_split(
        val_test_df,
        train_size=relative_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.csv")
        val_path = os.path.join(output_dir, "val.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df


def scaffold_split_dataset(
    input_csv: str,
    output_dir: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
    smiles_col: str = "smiles",
    label_col: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scaffold split CSV dataset into train/val/test based on molecular scaffolds.

    Uses MurckoScaffold to group molecules by their scaffold structure.
    This provides a more realistic evaluation by ensuring molecules with
    similar scaffolds are in the same split.

    Args:
        input_csv: Input CSV file path
        output_dir: Output directory (optional, saves files if specified)
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.1)
        seed: Random seed (default None uses numpy default)
        smiles_col: Column name for SMILES (default "smiles")
        label_col: Column name for label (optional, for multi-task)
        n_jobs: Number of threads for CSV reading.

    Returns:
        (train_df, val_df, test_df) tuple of DataFrames

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    from collections import defaultdict

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    df = pd.read_csv(input_csv)

    # Find SMILES column (handle different column names)
    if smiles_col not in df.columns:
        for col in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
            if col in df.columns:
                smiles_col = col
                break
        else:
            raise ValueError(f"SMILES column not found. Available: {df.columns.tolist()}")

    def get_scaffold(smiles: str) -> str:
        """Get Murcko scaffold from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "INVALID"
            scaffold = Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            return scaffold if scaffold else "NO_SCAFFOLD"
        except Exception:
            return "ERROR"

    # Compute scaffolds for all molecules
    scaffolds = [get_scaffold(s) for s in df[smiles_col].tolist()]

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size (larger first for deterministic ordering)
    sorted_scaffolds = sorted(
        scaffold_to_indices.keys(),
        key=lambda x: len(scaffold_to_indices[x]),
        reverse=True,
    )

    # Shuffle scaffolds then assign round-robin to balance sizes
    np.random.seed(seed)
    shuffled = list(scaffold_to_indices.items())
    np.random.shuffle(shuffled)

    train_indices = []
    val_indices = []
    test_indices = []

    train_count = 0
    val_count = 0
    test_count = 0
    total = len(df)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    for scaff_idx, (scaffold, indices) in enumerate(shuffled):
        mod = scaff_idx % 3
        n = len(indices)
        if mod == 0 and train_count < train_target:
            train_indices.extend(indices)
            train_count += n
        elif mod == 1 and val_count < val_target:
            val_indices.extend(indices)
            val_count += n
        else:
            test_indices.extend(indices)

    # If any split is empty due to rounding, redistribute smallest scaffolds
    if val_count == 0 or test_count == 0:
        all_indices = train_indices + val_indices + test_indices
        np.random.shuffle(all_indices)
        n = len(all_indices)
        train_indices = all_indices[:train_target]
        val_indices = all_indices[train_target : train_target + val_target]
        test_indices = all_indices[train_target + val_target :]

    # Shuffle within each split
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Auto-detect label column if not specified
    if label_col is None:
        numeric_cols = [
            c for c in df.columns if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric_cols) == 1:
            label_col = numeric_cols[0]
        else:
            # Try common label column names
            for col in [
                "measured log(solubility:mol/L)",
                "solubility",
                "label",
                "value",
            ]:
                if col in numeric_cols:
                    label_col = col
                    break
            if label_col is None:
                raise ValueError(f"Could not auto-detect label column. Found: {numeric_cols}")

    cols_to_keep = [smiles_col, label_col]
    train_df = df.iloc[train_indices][cols_to_keep].reset_index(drop=True)
    val_df = df.iloc[val_indices][cols_to_keep].reset_index(drop=True)
    test_df = df.iloc[test_indices][cols_to_keep].reset_index(drop=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        # Save column mapping metadata so create_data_loaders knows which columns to use
        meta = {"smiles_col": smiles_col, "label_col": label_col}
        with open(os.path.join(output_dir, "split_meta.json"), "w") as f:
            json.dump(meta, f)

    return train_df, val_df, test_df


def list_available_databases(
    database_dir: str = "src/data/database",
) -> list[str]:
    """List all available database files in directory.

    Args:
        database_dir: Path to database directory

    Returns:
        List of database file paths
    """
    if not os.path.exists(database_dir):
        return []
    return sorted(
        [os.path.join(database_dir, f) for f in os.listdir(database_dir) if f.endswith(".db")]
    )


def select_database(
    database_dir: str = "src/data/database",
) -> str:
    """Interactively select a database file.

    Args:
        database_dir: Path to database directory

    Returns:
        Selected database file path

    Raises:
        FileNotFoundError: No database files found
    """
    db_files = list_available_databases(database_dir)

    if not db_files:
        raise FileNotFoundError(f"Database directory empty or not found: {database_dir}")

    print("Available databases:")
    for i, db_path in enumerate(db_files, 1):
        print(f"  [{i}] {os.path.basename(db_path)}")

    while True:
        try:
            choice = int(input("\nSelect database number: "))
            if 1 <= choice <= len(db_files):
                return db_files[choice - 1]
            print(f"Invalid choice, enter 1-{len(db_files)}")
        except ValueError:
            print("Enter a valid number")
