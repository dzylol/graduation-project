# AGENTS.md - src/visualization/

**Training curves, prediction plots, molecule structures.** Pure matplotlib/RDKit.

## Structure
```
src/visualization/
├── __init__.py
├── dashboard.py           # Experiment comparison dashboard (287L)
├── training_plots.py      # plot_training_curves, plot_metric_comparison
├── prediction_plots.py   # plot_prediction_scatter, plot_residuals
└── molecule_plots.py      # draw_molecule, plot_molecule_grid
```

## Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `plot_training_curves` | training_plots.py | Epoch-level loss/MAE/RMSE curves |
| `plot_metric_comparison` | training_plots.py | Bar chart comparing multiple experiments |
| `plot_prediction_scatter` | prediction_plots.py | True vs predicted scatter with metrics overlay |
| `plot_residuals` | prediction_plots.py | Residual distribution and QQ plot |
| `draw_molecule` | molecule_plots.py | RDKit-rendered molecule from SMILES |
| `plot_molecule_grid` | molecule_plots.py | Grid of molecule structures |
| `create_experiment_dashboard` | dashboard.py | Comprehensive multi-experiment comparison panel |

## Conventions (THIS MODULE)
- All functions return `plt.Figure` — caller handles `plt.show()` / `plt.savefig()`
- `save_path=None` means don't save, just return figure
- Molecule drawing requires valid SMILES — validate with RDKit first
- Dashboard requires `n_exps > 0` — raises `ValueError` otherwise

## Anti-Patterns (THIS MODULE)
- **NEVER** call `plt.show()` inside plotting functions — caller decides
- **NEVER** pass invalid SMILES to RDKit — validate first