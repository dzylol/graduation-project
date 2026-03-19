#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.experiment_repo import ExperimentRepository


def format_metrics(metrics: dict, indent: int = 2) -> str:
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{' ' * indent}{key}: {value:.6f}")
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines) if lines else f"{' ' * indent}No metrics"


def list_experiments(
    repo: ExperimentRepository, status: str = None, dataset: str = None, limit: int = 20
):
    experiments = repo.list_all(status=status, dataset=dataset, limit=limit)

    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'=' * 80}")
    print(
        f"{'ID':<6} {'Name':<25} {'Dataset':<15} {'Status':<12} {'Best Epoch':<12} {'Created':<20}"
    )
    print(f"{'-' * 80}")

    for exp in experiments:
        created = exp.created_at[:19] if exp.created_at else "N/A"
        name = exp.name[:23] + ".." if len(exp.name) > 25 else exp.name
        print(
            f"{exp.id:<6} {name:<25} {exp.dataset:<15} {exp.status:<12} "
            f"{exp.best_epoch:<12} {created}"
        )

    print(f"{'=' * 80}")
    print(f"Total: {len(experiments)} experiment(s)")


def show_experiment_detail(repo: ExperimentRepository, exp_id: int):
    exp = repo.get_by_id(exp_id)

    if not exp:
        print(f"Experiment {exp_id} not found.")
        return

    print(f"\n{'=' * 80}")
    print(f"Experiment #{exp.id}: {exp.name}")
    print(f"{'=' * 80}")
    print(f"Dataset: {exp.dataset}")
    print(f"Tasks: {', '.join(exp.tasks) if exp.tasks else 'N/A'}")
    print(f"Status: {exp.status}")
    print(f"Best Epoch: {exp.best_epoch}")
    print(f"Created: {exp.created_at}")
    print(f"Updated: {exp.updated_at}")

    print(f"\nModel Config:")
    if exp.model_config:
        for key, value in exp.model_config.items():
            print(f"  {key}: {value}")
    else:
        print("  N/A")

    print(f"\nHyperparameters:")
    if exp.hyperparams:
        for key, value in exp.hyperparams.items():
            print(f"  {key}: {value}")
    else:
        print("  N/A")

    print(f"\nMetrics:")
    print(format_metrics(exp.metrics))

    if exp.training_logs:
        print(f"\nTraining Logs (last 5 epochs):")
        for log in exp.training_logs[-5:]:
            epoch = log.get("epoch", "?")
            train_loss = log.get("train_loss", 0)
            val_loss = log.get("val_loss", 0)
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

    print(f"{'=' * 80}\n")


def compare_experiments(repo: ExperimentRepository, exp_ids: list):
    results = repo.compare_experiments(exp_ids)

    if not results:
        print("No experiments found to compare.")
        return

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPARISON")
    print(f"{'=' * 100}\n")

    for i, exp in enumerate(results):
        print(f"[{i + 1}] {exp['name']} (ID: {exp['id']})")
        print(f"    Dataset: {exp['dataset']}")
        print(f"    Tasks: {', '.join(exp['tasks']) if exp['tasks'] else 'N/A'}")
        print(f"    Best Epoch: {exp['best_epoch']}")
        print(f"    Metrics: {format_metrics(exp['metrics'], indent=6)}")
        print()

    print(f"{'-' * 100}")
    print("\nMetrics Comparison Table:")
    print(f"{'Metric':<20}", end="")
    for i, exp in enumerate(results):
        name = exp["name"][:15]
        print(f" | {name:<15}", end="")
    print()
    print(f"{'-' * 100}")

    all_metrics = set()
    for exp in results:
        all_metrics.update(exp["metrics"].keys())

    for metric in sorted(all_metrics):
        print(f"{metric:<20}", end="")
        for exp in results:
            val = exp["metrics"].get(metric, "N/A")
            if isinstance(val, float):
                print(f" | {val:<15.6f}", end="")
            else:
                print(f" | {str(val):<15}", end="")
        print()

    print(f"{'=' * 100}\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment management tool")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all experiments"
    )
    parser.add_argument(
        "--detail", "-d", type=int, help="Show detailed info for experiment ID"
    )
    parser.add_argument(
        "--compare", "-c", nargs="+", type=int, help="Compare experiments by IDs"
    )
    parser.add_argument(
        "--status",
        "-s",
        choices=["running", "completed", "failed"],
        help="Filter by status",
    )
    parser.add_argument("--dataset", type=str, help="Filter by dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of experiments to show (default: 20)",
    )
    parser.add_argument("--delete", type=int, help="Delete experiment by ID")
    parser.add_argument(
        "--db-path", default="bi_mamba_chem.db", help="Path to database file"
    )

    args = parser.parse_args()
    repo = ExperimentRepository(db_path=args.db_path)

    if args.delete:
        if repo.delete(args.delete):
            print(f"Experiment {args.delete} deleted successfully.")
        else:
            print(f"Failed to delete experiment {args.delete}.")
        return

    if args.detail:
        show_experiment_detail(repo, args.detail)
    elif args.compare:
        compare_experiments(repo, args.compare)
    elif args.list:
        list_experiments(
            repo, status=args.status, dataset=args.dataset, limit=args.limit
        )
    else:
        list_experiments(
            repo, status=args.status, dataset=args.dataset, limit=args.limit
        )


if __name__ == "__main__":
    main()
