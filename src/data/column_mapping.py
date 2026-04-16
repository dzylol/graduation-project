"""Column mapping detection for molecular CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import pandas as pd
from rdkit import Chem

# 常见的 SMILES 列名（不区分大小写，去空白后匹配）
SMILES_COLUMNS: set[str] = {
    "smiles",
    "smile",
    "canonical_smiles",
    "canonical",
    "smi",
    "structure",
    "molecule",
    "mol",
    "chem_smile",
}

# 非 SMILES 数据集（明确跳过）
IGNORED_DATASETS: set[str] = {
    "sider",  # 药物副作用数据库，无 SMILES 列
}

# 预定义数据集的列配置（用于自动检测失败或列名非常规的数据集）
DATASET_CONFIG: Dict[str, Dict[str, List[str]]] = cast(
    Dict[str, Dict[str, List[str]]],
    {
        "ESOL": {
            "smiles_col": "SMILES",
            "label_cols": ["measured log(solubility:mol/L)"],
        },
        "BBBP": {
            "smiles_col": "smiles",
            "label_cols": ["p_np"],
        },
        "ZINC250K": {
            "smiles_col": "smiles",
            "label_cols": ["logP", "qed", "SAS"],
        },
        "FreeSolv": {
            "smiles_col": "smiles",
            "label_cols": ["freesolv"],
        },
        "Lipophilicity": {
            "smiles_col": "smiles",
            "label_cols": ["exp"],
        },
        "ClinTox": {
            "smiles_col": "smiles",
            "label_cols": ["FDA_APPROVED", "CT_TOX"],
        },
        "HIV": {
            "smiles_col": "smiles",
            "label_cols": ["HIV_active"],
        },
        "MUV": {
            "smiles_col": "smiles",
            "label_cols": [],  # 动态检测
        },
        "mpro": {
            "smiles_col": "smiles",
            "label_cols": ["IC50"],
        },
        "EGFR": {
            "smiles_col": "smiles",
            "label_cols": ["value"],
        },
        "bace": {
            "smiles_col": "smiles",
            "label_cols": ["pIC50"],
        },
    },
)


def validate_smiles_internal(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


@dataclass
class ColumnMapping:
    """CSV column mapping result for auto-detection."""

    smiles_col: str
    label_cols: List[str]
    detection_method: str
    confidence: float = 1.0


def detect_column_mapping(df: pd.DataFrame, dataset_name: Optional[str] = None) -> ColumnMapping:
    """Auto-detect CSV column mapping (smiles_col + label_cols).

    Detection order:
    1. DATASET_CONFIG check (for known datasets)
    2. Whitelist matching (case-insensitive SMILES_COLUMNS)
    3. RDKit validation (sample rows, check >80% valid)
    4. Fallback to first column as SMILES

    Args:
        df: Loaded pandas DataFrame
        dataset_name: Optional dataset name to match against DATASET_CONFIG

    Returns:
        ColumnMapping: smiles_col, label_cols, detection_method, confidence

    Raises:
        ValueError: If no valid mapping can be detected
    """
    # 1. DATASET_CONFIG check (case-insensitive dataset name match)
    if dataset_name:
        for name, config in DATASET_CONFIG.items():
            if name.lower() == dataset_name.lower():
                smiles_col = cast(str, config["smiles_col"])
                label_cols = config.get("label_cols", [])
                if smiles_col in df.columns:
                    return ColumnMapping(
                        smiles_col=smiles_col,
                        label_cols=label_cols,
                        detection_method="dataset_config",
                        confidence=1.0,
                    )

    # 2. Whitelist matching (case-insensitive)
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in SMILES_COLUMNS:
            label_cols = [
                c for c in df.columns if c != col and pd.api.types.is_numeric_dtype(df[c])
            ]
            return ColumnMapping(
                smiles_col=col,
                label_cols=label_cols,
                detection_method="whitelist",
                confidence=1.0,
            )

    # 2. RDKit validation (sample first 20 rows)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(20)
            valid_count = sum(validate_smiles_internal(s) for s in sample)
            if valid_count >= 16:  # >80%
                label_cols = [
                    c for c in df.columns if c != col and pd.api.types.is_numeric_dtype(df[c])
                ]
                return ColumnMapping(
                    smiles_col=col,
                    label_cols=label_cols,
                    detection_method="rdkit_validation",
                    confidence=valid_count / max(len(sample), 1),
                )

    # 3. Fallback: first column is SMILES, rest are numeric labels
    if len(df.columns) >= 2:
        smiles_col = df.columns[0]
        label_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
        return ColumnMapping(
            smiles_col=smiles_col,
            label_cols=label_cols,
            detection_method="fallback_first_column",
            confidence=0.5,
        )

    raise ValueError(f"Cannot detect column mapping. DataFrame columns: {df.columns.tolist()}")
