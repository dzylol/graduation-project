# AGENTS.md - src/visualization/

**Generated:** 2026-04-20

**Training curves, prediction plots, molecule structures.** Pure matplotlib/RDKit.

## Structure
```
src/visualization/
├── __init__.py
├── dashboard.py           # Multi-experiment comparison
├── training_plots.py     # Training curves and metric bars
├── prediction_plots.py   # Scatter plots and residual analysis
└── molecule_plots.py     # RDKit molecule rendering
```

## Key Functions

| Function | File | Signature | Purpose |
|----------|------|-----------|---------|
| `plot_training_curves` | training_plots.py | `(logs, save_path?, metrics?, title?, figsize?)` | Epoch-level loss/MAE/RMSE curves |
| `plot_metric_comparison` | training_plots.py | `(results, metric?, title?, save_path?, figsize?)` | Bar chart across experiments |
| `plot_prediction_scatter` | prediction_plots.py | `(y_true, y_pred, task_name?, save_path?, figsize?, metrics?)` | True vs predicted scatter |
| `plot_residuals` | prediction_plots.py | `(y_true, y_pred, task_name?, save_path?, figsize?)` | Residual plot + histogram |
| `draw_molecule` | molecule_plots.py | `(smiles, size?, legend?, highlight_atoms?, return_image?)` | RDKit molecule from SMILES |
| `plot_molecule_grid` | molecule_plots.py | `(smiles_list, mols_per_row?, subplot_size?, legends?, title?, save_path?)` | Grid of molecule structures |
| `create_experiment_dashboard` | dashboard.py | `(experiments, save_path?, figsize?)` | Multi-experiment comparison panel |

## RDKit Conventions

- **Validate first**: Pass only valid SMILES to `draw_molecule()`. Invalid SMILES returns `None`.
- **Graceful degradation**: `draw_molecule()` catches exceptions, prints error, returns `None`.
- **Size units**: `draw_molecule(size=(300, 300))` uses pixels; `plot_molecule_grid(subplot_size=(250, 250))` uses pixels.
- **highlight_atoms**: List of atom indices to highlight; clamped to valid range automatically.
- **Import pattern**: RDKit imported inside try/except for optional dependency.

## Matplotlib Conventions

- **Return pattern**: All functions return `plt.Figure`; caller handles `plt.show()` / `plt.savefig()`.
- **save_path=None**: Don't save, just return figure.
- **figsize**: Tuple of `(width, height)` in inches.
- **DPI**: Default 150 when saving; 100 for on-screen.
- **tight_layout**: Called before save in all functions.
- **Grid handling**: `plot_molecule_grid` handles empty cells by calling `axis("off")` on excess axes.

## Anti-Patterns

- **NEVER** call `plt.show()` inside plotting functions. Caller decides when to display.
- **NEVER** pass invalid SMILES to RDKit without validation. Use `Chem.MolFromSmiles()` first.
- **NEVER** use `plt.close()` inside functions. Caller manages figure lifecycle.
- **Empty dashboard**: `create_experiment_dashboard()` raises `ValueError` if `experiments` list is empty.
- **Empty logs**: `plot_training_curves()` raises `ValueError` if `logs` list is empty.
