import json
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from rdkit import Chem
from src.db.database import get_db, Molecule


class MoleculeRepository:
    def __init__(self, db_path: str = "bi_mamba_chem.db"):
        self.db = get_db(db_path)

    def add(
        self,
        smiles: str,
        canonical_smiles: str = "",
        properties: Optional[Dict[str, float]] = None,
        dataset_name: Optional[str] = None,
    ) -> int:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            properties_json = json.dumps(properties or {})
            cursor.execute(
                """
                INSERT OR REPLACE INTO molecules (smiles, canonical_smiles, properties, dataset_name)
                VALUES (?, ?, ?, ?)
                """,
                (smiles, canonical_smiles, properties_json, dataset_name),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def add_batch(
        self, molecules: List[Dict], dataset_name: Optional[str] = None
    ) -> int:
        count = 0
        with self.db.connect() as conn:
            cursor = conn.cursor()
            for mol in molecules:
                smiles = mol.get("smiles", "")
                canonical = mol.get("canonical_smiles", "")
                props = json.dumps(mol.get("properties", {}))
                try:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO molecules (smiles, canonical_smiles, properties, dataset_name)
                        VALUES (?, ?, ?, ?)
                        """,
                        (smiles, canonical, props, dataset_name),
                    )
                    count += cursor.rowcount
                except Exception:
                    continue
            conn.commit()
        return count

    def get_by_smiles(
        self, smiles: str, dataset_name: Optional[str] = None
    ) -> Optional[Molecule]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            if dataset_name:
                cursor.execute(
                    "SELECT * FROM molecules WHERE smiles = ? AND dataset_name = ?",
                    (smiles, dataset_name),
                )
            else:
                cursor.execute("SELECT * FROM molecules WHERE smiles = ?", (smiles,))
            row = cursor.fetchone()
            if row:
                return self._row_to_molecule(row)
            return None

    def get_by_id(self, mol_id: int) -> Optional[Molecule]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM molecules WHERE id = ?", (mol_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_molecule(row)
            return None

    def list_all(
        self, limit: int = 100, offset: int = 0, dataset_name: Optional[str] = None
    ) -> List[Molecule]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            if dataset_name:
                cursor.execute(
                    "SELECT * FROM molecules WHERE dataset_name = ? ORDER BY id LIMIT ? OFFSET ?",
                    (dataset_name, limit, offset),
                )
            else:
                cursor.execute(
                    "SELECT * FROM molecules ORDER BY id LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            return [self._row_to_molecule(row) for row in cursor.fetchall()]

    def get_dataset(self, dataset_name: str) -> List[Molecule]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM molecules WHERE dataset_name = ? ORDER BY id",
                (dataset_name,),
            )
            return [self._row_to_molecule(row) for row in cursor.fetchall()]

    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as total FROM molecules WHERE dataset_name = ?",
                (dataset_name,),
            )
            total = cursor.fetchone()[0]
            return {"total": total, "dataset_name": dataset_name}

    def list_datasets(self) -> List[str]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT dataset_name FROM molecules WHERE dataset_name IS NOT NULL"
            )
            return [row[0] for row in cursor.fetchall() if row[0]]

    def import_from_csv(
        self,
        csv_path: str,
        dataset_name: str,
        smiles_col: str = "smiles",
        label_cols: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Tuple[int, int]:
        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns:
            for col in df.columns:
                if "smiles" in col.lower() or "mol" in col.lower():
                    smiles_col = col
                    break
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in CSV")
        if label_cols is None:
            label_cols = [c for c in df.columns if c != smiles_col]
        imported = 0
        skipped = 0
        for _, row in df.iterrows():
            smiles = str(row[smiles_col])
            if validate:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    skipped += 1
                    continue
                canonical = Chem.MolToSmiles(mol)
            else:
                canonical = ""
            properties = {
                col: float(row[col]) for col in label_cols if col in df.columns
            }
            try:
                self.add(
                    smiles=smiles,
                    canonical_smiles=canonical,
                    properties=properties,
                    dataset_name=dataset_name,
                )
                imported += 1
            except Exception:
                skipped += 1
        return imported, skipped

    def search_by_property(
        self,
        property_name: str,
        min_val: float = float("-inf"),
        max_val: float = float("inf"),
        dataset_name: Optional[str] = None,
    ) -> List[Molecule]:
        results = []
        with self.db.connect() as conn:
            cursor = conn.cursor()
            if dataset_name:
                cursor.execute(
                    "SELECT * FROM molecules WHERE dataset_name = ?", (dataset_name,)
                )
            else:
                cursor.execute("SELECT * FROM molecules")
            for row in cursor.fetchall():
                mol = self._row_to_molecule(row)
                if mol.properties.get(property_name):
                    val = mol.properties[property_name]
                    if min_val <= val <= max_val:
                        results.append(mol)
        return results

    def count(self, dataset_name: Optional[str] = None) -> int:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            if dataset_name:
                cursor.execute(
                    "SELECT COUNT(*) FROM molecules WHERE dataset_name = ?",
                    (dataset_name,),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM molecules")
            return cursor.fetchone()[0]

    def delete(self, mol_id: int) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM molecules WHERE id = ?", (mol_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_dataset(self, dataset_name: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM molecules WHERE dataset_name = ?", (dataset_name,)
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_molecule(self, row) -> Molecule:
        return Molecule(
            id=row["id"],
            smiles=row["smiles"],
            canonical_smiles=row["canonical_smiles"] or "",
            properties=json.loads(row["properties"] or "{}"),
            dataset_name=row["dataset_name"],
            created_at=row["created_at"],
        )
