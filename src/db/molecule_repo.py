import json
from typing import List, Optional, Dict
from src.db.database import get_db, Molecule


class MoleculeRepository:
    def __init__(self, db_path: str = "bi_mamba_chem.db"):
        self.db = get_db(db_path)

    def add(
        self,
        smiles: str,
        canonical_smiles: str = "",
        properties: Optional[Dict[str, float]] = None,
    ) -> int:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            properties_json = json.dumps(properties or {})
            cursor.execute(
                """
                INSERT OR REPLACE INTO molecules (smiles, canonical_smiles, properties)
                VALUES (?, ?, ?)
                """,
                (smiles, canonical_smiles, properties_json),
            )
            conn.commit()
            return cursor.lastrowid

    def add_batch(self, molecules: List[Dict]) -> int:
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
                        INSERT OR IGNORE INTO molecules (smiles, canonical_smiles, properties)
                        VALUES (?, ?, ?)
                        """,
                        (smiles, canonical, props),
                    )
                    count += cursor.rowcount
                except Exception:
                    continue
            conn.commit()
        return count

    def get_by_smiles(self, smiles: str) -> Optional[Molecule]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
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

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Molecule]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM molecules ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
            )
            return [self._row_to_molecule(row) for row in cursor.fetchall()]

    def search_by_property(
        self,
        property_name: str,
        min_val: float = float("-inf"),
        max_val: float = float("inf"),
    ) -> List[Molecule]:
        results = []
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM molecules")
            for row in cursor.fetchall():
                mol = self._row_to_molecule(row)
                if mol.properties.get(property_name):
                    val = mol.properties[property_name]
                    if min_val <= val <= max_val:
                        results.append(mol)
        return results

    def count(self) -> int:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM molecules")
            return cursor.fetchone()[0]

    def delete(self, mol_id: int) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM molecules WHERE id = ?", (mol_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_molecule(self, row) -> Molecule:
        return Molecule(
            id=row["id"],
            smiles=row["smiles"],
            canonical_smiles=row["canonical_smiles"] or "",
            properties=json.loads(row["properties"] or "{}"),
            created_at=row["created_at"],
        )
