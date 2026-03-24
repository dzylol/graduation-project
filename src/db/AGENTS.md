# AGENTS.md - src/db/

**SQLite persistence layer.** Experiment tracking + molecule storage.

## Structure
```
src/db/
├── __init__.py
├── database.py          # Database singleton, Molecule/Experiment dataclasses (130L)
├── experiment_repo.py   # ExperimentRepository CRUD (251L)
├── molecule_repo.py     # MoleculeRepository CRUD
└── database/            # (empty - do not use)
```

## Key Classes

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Database` | class | database.py | SQLite singleton with context managers |
| `Molecule` | dataclass | database.py | Molecule entity |
| `Experiment` | dataclass | database.py | Experiment entity |
| `ExperimentRepository` | class | experiment_repo.py | Experiment CRUD |
| `MoleculeRepository` | class | molecule_repo.py | Molecule CRUD |
| `get_db` | factory | database.py | Database instance accessor |

## Conventions (THIS MODULE)

- **Singleton pattern**: `Database.get_instance()` — one instance per process
- **Context managers**: `with db.connect() as conn:` for connection handling
- **JSON serialization** for dict/list fields (model_config, hyperparams, metrics)
- **check_same_thread=False** for MPS compatibility
- Default db_path: `bi_mamba_chem.db` (project root)

## Anti-Patterns (THIS MODULE)

- **NEVER** create Database directly — use `get_db()` factory
- **NEVER** store dict/list fields as raw SQLite — serialize to JSON first
- **NEVER** commit() outside context manager — let `__exit__` handle it
