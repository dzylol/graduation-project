"""
Molecule structure visualization utilities.

Functions for drawing and displaying molecular structures using RDKit.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io


def draw_molecule(
    smiles: str,
    size: Tuple[int, int] = (300, 300),
    legend: Optional[str] = None,
    highlight_atoms: Optional[List[int]] = None,
    return_image: bool = False,
) -> Optional[plt.Figure]:
    """
    Draw a molecule structure from SMILES.

    Args:
        smiles: SMILES string
        size: image size in pixels
        legend: text to display as legend
        highlight_atoms: list of atom indices to highlight
        return_image: if True, return PIL Image instead of matplotlib Figure

    Returns:
        matplotlib Figure or PIL Image, or None on failure
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None

        if highlight_atoms:
            highlight_atoms = [i for i in highlight_atoms if i < mol.GetNumAtoms()]

        drawer = Draw.MolDraw2DCairo(size[0], size[1])

        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()

        if return_image:
            return Image.open(io.BytesIO(drawer.GetDrawingText()))
        else:
            img = Image.open(io.BytesIO(drawer.GetDrawingText()))
            fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
            ax.imshow(img)
            ax.axis("off")
            if legend:
                ax.set_title(legend, fontsize=12)
            return fig

    except ImportError:
        print("RDKit is required for molecule visualization")
        return None
    except Exception as e:
        print(f"Error drawing molecule: {e}")
        return None


def plot_molecule_grid(
    smiles_list: List[str],
    mols_per_row: int = 4,
    subplot_size: Tuple[int, int] = (250, 250),
    legends: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a grid of molecule structures.

    Args:
        smiles_list: list of SMILES strings
        mols_per_row: number of molecules per row
        subplot_size: size of each subplot
        legends: optional list of legend text for each molecule
        title: figure title
        save_path: path to save figure

    Returns:
        matplotlib Figure object
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_indices = [i for i, m in enumerate(mols) if m is not None]

        if not valid_indices:
            print("No valid molecules to display")
            return plt.figure()

        n_mols = len(valid_indices)
        n_rows = (n_mols + mols_per_row - 1) // mols_per_row
        n_cols = min(mols_per_row, n_mols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * subplot_size[0] / 100, n_rows * subplot_size[1] / 100),
            dpi=100,
        )
        fig.suptitle(title or "Molecule Grid", fontsize=14, fontweight="bold")

        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = axes

        for idx, mol_idx in enumerate(valid_indices):
            row = idx // mols_per_row
            col = idx % mols_per_row
            ax = axes[row][col]

            mol = mols[mol_idx]
            img = Draw.MolToImage(mol, size=subplot_size)

            ax.imshow(img)
            ax.axis("off")

            if legends and mol_idx < len(legends):
                ax.set_title(legends[mol_idx], fontsize=8, pad=5)

        for idx in range(n_mols, n_rows * mols_per_row):
            row = idx // mols_per_row
            col = idx % mols_per_row
            if row < len(axes) and col < len(axes[row]):
                axes[row][col].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Molecule grid saved to {save_path}")

        return fig

    except ImportError:
        print("RDKit is required for molecule visualization")
        return plt.figure()
    except Exception as e:
        print(f"Error creating molecule grid: {e}")
        return plt.figure()


def plot_molecule_with_predictions(
    smiles_list: List[str],
    predictions: List[float],
    true_values: Optional[List[float]] = None,
    metric: str = "RMSE",
    mols_per_row: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot molecules with their predicted (and optionally true) values.

    Args:
        smiles_list: list of SMILES strings
        predictions: list of predicted values
        true_values: optional list of true values
        metric: metric name to display
        mols_per_row: molecules per row
        save_path: path to save figure

    Returns:
        matplotlib Figure object
    """
    legends = []
    for i, pred in enumerate(predictions):
        if true_values and i < len(true_values):
            legends.append(f"True: {true_values[i]:.3f}\nPred: {pred:.3f}")
        else:
            legends.append(f"Pred: {pred:.3f}")

    return plot_molecule_grid(
        smiles_list=smiles_list,
        mols_per_row=mols_per_row,
        legends=legends,
        title=f"Molecule Predictions ({metric})",
        save_path=save_path,
    )
