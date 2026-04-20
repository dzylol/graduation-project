import os


def list_available_databases(
    database_dir: str = "src/data/database",
) -> list[str]:
    if not os.path.exists(database_dir):
        return []
    return sorted(
        [os.path.join(database_dir, f) for f in os.listdir(database_dir) if f.endswith(".db")]
    )


def select_database(
    database_dir: str = "src/data/database",
) -> str:
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
