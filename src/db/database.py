import sqlite3
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Molecule:
    id: Optional[int] = None
    smiles: str = ""
    canonical_smiles: str = ""
    properties: Dict[str, float] = field(default_factory=dict)
    dataset_name: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Experiment:
    id: Optional[int] = None
    name: str = ""
    dataset: str = ""
    tasks: List[str] = field(default_factory=list)
    model_config: Dict[str, Any] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    training_logs: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    status: str = "running"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Database:
    _instance: Optional["Database"] = None

    def __init__(self, db_path: str = "bi_mamba_chem.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: str = "bi_mamba_chem.db") -> "Database":
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def connect(self):
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS molecules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    smiles TEXT NOT NULL,
                    canonical_smiles TEXT,
                    properties TEXT,
                    dataset_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(smiles, dataset_name)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    tasks TEXT,
                    model_config TEXT,
                    hyperparams TEXT,
                    metrics TEXT,
                    training_logs TEXT,
                    best_epoch INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    smiles TEXT NOT NULL,
                    experiment_id INTEGER,
                    predictions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_molecules_smiles 
                ON molecules(smiles)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_molecules_dataset 
                ON molecules(dataset_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_dataset 
                ON experiments(dataset)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_status 
                ON experiments(status)
            """)

            conn.commit()

            # Migration: add dataset_name column if it doesn't exist (for existing databases)
            cursor.execute("PRAGMA table_info(molecules)")
            columns = [row[1] for row in cursor.fetchall()]
            if "dataset_name" not in columns:
                cursor.execute("ALTER TABLE molecules ADD COLUMN dataset_name TEXT")
                conn.commit()


_db_instance: Optional[Database] = None


def get_db(db_path: str = "bi_mamba_chem.db") -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database.get_instance(db_path)
    return _db_instance
