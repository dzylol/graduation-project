import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.db.database import get_db, Experiment


class ExperimentRepository:
    def __init__(self, db_path: str = "bi_mamba_chem.db"):
        self.db = get_db(db_path)

    def create(
        self,
        name: str,
        dataset: str,
        tasks: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiments (name, dataset, tasks, model_config, hyperparams, status)
                VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (
                    name,
                    dataset,
                    json.dumps(tasks or []),
                    json.dumps(model_config or {}),
                    json.dumps(hyperparams or {}),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_metrics(self, exp_id: int, metrics: Dict[str, Any]) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE experiments 
                SET metrics = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(metrics), exp_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_training_logs(self, exp_id: int, logs: List[Dict[str, float]]) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE experiments 
                SET training_logs = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(logs), exp_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def append_training_log(self, exp_id: int, epoch_log: Dict[str, float]) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT training_logs FROM experiments WHERE id = ?", (exp_id,)
            )
            row = cursor.fetchone()
            if row:
                logs = json.loads(row["training_logs"] or "[]")
                logs.append(epoch_log)
                cursor.execute(
                    """
                    UPDATE experiments 
                    SET training_logs = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (json.dumps(logs), exp_id),
                )
                conn.commit()
                return True
            return False

    def set_best_epoch(
        self, exp_id: int, best_epoch: int, metrics: Dict[str, Any]
    ) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE experiments 
                SET best_epoch = ?, metrics = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (best_epoch, json.dumps(metrics), exp_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def complete(
        self, exp_id: int, final_metrics: Dict[str, Any], best_epoch: int
    ) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE experiments 
                SET status = 'completed', 
                    metrics = ?, 
                    best_epoch = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(final_metrics), best_epoch, exp_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def fail(self, exp_id: int, error_message: str = "") -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE experiments 
                SET status = 'failed', 
                    metrics = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps({"error": error_message}), exp_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_by_id(self, exp_id: int) -> Optional[Experiment]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_experiment(row)
            return None

    def list_all(
        self,
        status: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Experiment]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if dataset:
            query += " AND dataset = ?"
            params.append(dataset)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_experiment(row) for row in cursor.fetchall()]

    def get_best_for_dataset(
        self, dataset: str, metric: str = "val_loss", minimize: bool = True
    ) -> Optional[Experiment]:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM experiments 
                WHERE dataset = ? AND status = 'completed'
                ORDER BY created_at DESC
                """,
                (dataset,),
            )
            rows = cursor.fetchall()

            best_exp = None
            best_val = float("inf") if minimize else float("-inf")

            for row in rows:
                exp = self._row_to_experiment(row)
                if exp.metrics.get(metric) is not None:
                    val = exp.metrics[metric]
                    if minimize and val < best_val:
                        best_val = val
                        best_exp = exp
                    elif not minimize and val > best_val:
                        best_val = val
                        best_exp = exp
            return best_exp

    def compare_experiments(self, exp_ids: List[int]) -> List[Dict[str, Any]]:
        experiments = []
        for exp_id in exp_ids:
            exp = self.get_by_id(exp_id)
            if exp:
                experiments.append(
                    {
                        "id": exp.id,
                        "name": exp.name,
                        "dataset": exp.dataset,
                        "tasks": exp.tasks,
                        "best_epoch": exp.best_epoch,
                        "metrics": exp.metrics,
                        "hyperparams": exp.hyperparams,
                        "created_at": exp.created_at,
                    }
                )
        return experiments

    def count(self, status: Optional[str] = None) -> int:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT COUNT(*) FROM experiments WHERE status = ?", (status,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM experiments")
            return cursor.fetchone()[0]

    def delete(self, exp_id: int) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_experiment(self, row) -> Experiment:
        return Experiment(
            id=row["id"],
            name=row["name"],
            dataset=row["dataset"],
            tasks=json.loads(row["tasks"] or "[]"),
            model_config=json.loads(row["model_config"] or "{}"),
            hyperparams=json.loads(row["hyperparams"] or "{}"),
            metrics=json.loads(row["metrics"] or "{}"),
            training_logs=json.loads(row["training_logs"] or "[]"),
            best_epoch=row["best_epoch"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
