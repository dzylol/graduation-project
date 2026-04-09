#!/usr/bin/env python3
"""
Column auto-detection unit tests

Tests for:
1. detect_column_mapping() function
2. ColumnMapping dataclass
3. MoleculeDataset with auto-detection

Usage:
    python tests/test_column_detection.py
    pytest tests/test_column_detection.py -v
"""

import sys
import os
import tempfile
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from data.molecule_dataset import (
    ColumnMapping,
    detect_column_mapping,
    SMILES_COLUMNS,
    IGNORED_DATASETS,
    DATASET_CONFIG,
    MoleculeDataset,
)


class TestDetectColumnMapping:
    """Tests for detect_column_mapping function."""

    def test_whitelist_match_standard(self):
        df = pd.DataFrame({"smiles": ["CCO", "CC(C)O"], "label": [1.0, 2.0]})
        mapping = detect_column_mapping(df)
        assert mapping.smiles_col == "smiles"
        assert mapping.label_cols == ["label"]
        assert mapping.detection_method == "whitelist"
        assert mapping.confidence == 1.0

    def test_whitelist_match_case_insensitive(self):
        df = pd.DataFrame({"SMILES": ["CCO", "CC(C)O"], "value": [1.0, 2.0]})
        mapping = detect_column_mapping(df)
        assert mapping.smiles_col == "SMILES"
        assert mapping.detection_method == "whitelist"

    def test_whitelist_match_with_whitespace(self):
        df = pd.DataFrame({" smiles ": ["CCO", "CC(C)O"], "label": [1.0, 2.0]})
        mapping = detect_column_mapping(df)
        assert mapping.smiles_col == " smiles "

    def test_rdkite_validation_fallback(self):
        valid_smiles = [
            "CCO",
            "CC(C)O",
            "c1ccccc1",
            "CCCC",
            "CCCOC",
            "CC(=O)OC",
            "c1ccncc1",
            "CN1C=NC=NC1",
            "O=C=O",
            "C#N",
        ]
        df = pd.DataFrame(
            {
                "compound": valid_smiles * 2,  # 20 samples
                "score": list(range(20)),
            }
        )
        mapping = detect_column_mapping(df)
        assert mapping.smiles_col == "compound"
        assert mapping.detection_method == "rdkit_validation"

    def test_fallback_first_column(self):
        df = pd.DataFrame({"C": ["valid", "notvalid"], "value": [1.0, 2.0]})
        mapping = detect_column_mapping(df)
        assert mapping.smiles_col == "C"
        assert mapping.detection_method == "fallback_first_column"
        assert mapping.confidence == 0.5

    def test_multilabel_detection(self):
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CC(C)O"],
                "logP": [1.0, 2.0],
                "qed": [0.5, 0.6],
                "SAS": [3.0, 4.0],
            }
        )
        mapping = detect_column_mapping(df)
        assert mapping.smiles_col == "smiles"
        assert len(mapping.label_cols) == 3
        assert "logP" in mapping.label_cols
        assert "qed" in mapping.label_cols
        assert "SAS" in mapping.label_cols

    def test_nan_label_handling(self):
        df = pd.DataFrame(
            {"smiles": ["CCO", "CC(C)O", "c1ccccc1"], "label": [1.0, None, 3.0]}
        )
        mapping = detect_column_mapping(df)
        assert mapping.smiles_col == "smiles"
        assert mapping.label_cols == ["label"]

    def test_column_mapping_dataclass(self):
        mapping = ColumnMapping(
            smiles_col="smiles",
            label_cols=["label1", "label2"],
            detection_method="whitelist",
            confidence=1.0,
        )
        assert mapping.smiles_col == "smiles"
        assert len(mapping.label_cols) == 2
        assert mapping.confidence == 1.0


class TestSmilesColumnsSet:
    """Tests for SMILES_COLUMNS whitelist."""

    def test_smiles_columns_not_empty(self):
        assert len(SMILES_COLUMNS) > 0

    def test_smiles_columns_lowercase(self):
        for col in SMILES_COLUMNS:
            assert col == col.lower()


class TestIgnoredDatasets:
    """Tests for IGNORED_DATASETS set."""

    def test_sider_is_ignored(self):
        assert "sider" in IGNORED_DATASETS

    def test_ignored_datasets_is_set(self):
        assert isinstance(IGNORED_DATASETS, (set, frozenset))


class TestDatasetConfig:
    """Tests for DATASET_CONFIG."""

    def test_esol_config(self):
        assert "ESOL" in DATASET_CONFIG
        assert DATASET_CONFIG["ESOL"]["smiles_col"] == "SMILES"
        assert "measured log(solubility:mol/L)" in DATASET_CONFIG["ESOL"]["label_cols"]

    def test_bbbp_config(self):
        assert "BBBP" in DATASET_CONFIG
        assert DATASET_CONFIG["BBBP"]["smiles_col"] == "smiles"
        assert "p_np" in DATASET_CONFIG["BBBP"]["label_cols"]

    def test_zinc250k_multilabel(self):
        assert "ZINC250K" in DATASET_CONFIG
        assert len(DATASET_CONFIG["ZINC250K"]["label_cols"]) == 3

    def test_clintox_both_labels(self):
        assert "ClinTox" in DATASET_CONFIG
        assert "FDA_APPROVED" in DATASET_CONFIG["ClinTox"]["label_cols"]
        assert "CT_TOX" in DATASET_CONFIG["ClinTox"]["label_cols"]


class TestMoleculeDatasetLoading:
    """Tests for MoleculeDataset with auto-detection."""

    def test_load_csv_with_standard_columns(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "label"])
            writer.writerow(["CCO", 1.0])
            writer.writerow(["CC(C)O", 2.0])
            writer.writerow(["c1ccccc1", 3.0])
            temp_file = f.name

        try:
            ds = MoleculeDataset(temp_file)
            assert len(ds) == 3
            assert ds[0][1].shape[0] == 1
        finally:
            os.unlink(temp_file)

    def test_load_csv_with_explicit_smiles_col(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "value"])
            writer.writerow(["CCO", 1.0])
            writer.writerow(["CC(C)O", 2.0])
            temp_file = f.name

        try:
            ds = MoleculeDataset(temp_file, smiles_col="SMILES")
            assert len(ds) == 2
        finally:
            os.unlink(temp_file)

    def test_load_csv_multilabel(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "logP", "qed", "SAS"])
            writer.writerow(["CCO", 1.0, 0.5, 3.0])
            writer.writerow(["CC(C)O", 2.0, 0.6, 4.0])
            temp_file = f.name

        try:
            ds = MoleculeDataset(temp_file)
            assert len(ds) == 2
            assert ds[0][1].shape[0] == 3
        finally:
            os.unlink(temp_file)

    def test_load_csv_with_nan_labels(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "label"])
            writer.writerow(["CCO", 1.0])
            writer.writerow(["CC(C)O", ""])
            writer.writerow(["c1ccccc1", 3.0])
            temp_file = f.name

        try:
            ds = MoleculeDataset(temp_file)
            assert len(ds) >= 2
        finally:
            os.unlink(temp_file)

    def test_load_csv_rdkite_validated(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["structure", "score"])
            writer.writerow(["CCO", 1.0])
            writer.writerow(["CC(C)O", 2.0])
            writer.writerow(["InvalidSMILES", 3.0])
            temp_file = f.name

        try:
            ds = MoleculeDataset(temp_file)
            assert len(ds) == 2
        finally:
            os.unlink(temp_file)

    def test_load_csv_fallback_first_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["C", "value"])
            writer.writerow(["CCO", 1.0])
            writer.writerow(["CC(C)O", 2.0])
            temp_file = f.name

        try:
            ds = MoleculeDataset(temp_file)
            assert len(ds) == 2
        finally:
            os.unlink(temp_file)

    def test_explicit_columns_override_auto_detection(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_col", "my_label"])
            writer.writerow(["CCO", 99.0])
            writer.writerow(["CC(C)O", 88.0])
            temp_file = f.name

        try:
            ds = MoleculeDataset(
                temp_file, smiles_col="wrong_col", label_cols=["my_label"]
            )
            assert len(ds) == 2
            assert ds[0][1].item() == 99.0
        finally:
            os.unlink(temp_file)


def test_random_split_ratio():
    from src.data.molecule_dataset import random_split_dataset

    train, val, test = random_split_dataset("dataset/ESOL/delaney.csv", seed=42)
    total = len(train) + len(val) + len(test)
    assert abs(len(train) / total - 0.8) < 0.01
    assert abs(len(val) / total - 0.1) < 0.01
    assert abs(len(test) / total - 0.1) < 0.01


def test_random_split_reproducibility():
    from src.data.molecule_dataset import random_split_dataset

    train1, _, _ = random_split_dataset("dataset/ESOL/delaney.csv", seed=42)
    train2, _, _ = random_split_dataset("dataset/ESOL/delaney.csv", seed=42)
    assert train1.equals(train2)


def test_random_split_different_seeds():
    from src.data.molecule_dataset import random_split_dataset

    train1, _, _ = random_split_dataset("dataset/ESOL/delaney.csv", seed=42)
    train2, _, _ = random_split_dataset("dataset/ESOL/delaney.csv", seed=123)
    assert not train1.equals(train2)


def test_random_split_multithread():
    from src.data.molecule_dataset import random_split_dataset

    train, val, test = random_split_dataset(
        "dataset/ESOL/delaney.csv", seed=42, n_jobs=4
    )
    assert len(train) + len(val) + len(test) == 1144


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
