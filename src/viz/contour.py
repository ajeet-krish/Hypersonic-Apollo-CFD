"""Contour plotting for hypersonic blunt body CFD results.

2D filled contour plots of Mach, pressure, and temperature in the (x, r) plane
using matplotlib tricontourf with professional academic styling.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from .style import COLORS, apply_theme


def _plot_field_contour(
    data: object,
    field: np.ndarray,
    field_name: str,
    output_path: Path,
    body_contour: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Field Contour",
    cmap: str = "inferno",
    dpi: int = 150,
) -> Path:
    """Plot a generic field contour using tricontourf.

    Args:
        data: VTUData with coordinates.
        field: Scalar field values at each node.
        field_name: Label for the colorbar.
        output_path: Path to save the plot.
        body_contour: Optional (x, r) body contour to overlay.
        title: Plot title.
        cmap: Matplotlib colormap name.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = data.coordinates

    # Clamp extreme values for better visualization
    f_min = np.percentile(field, 1)
    f_max = np.percentile(field, 99)
    if f_max <= f_min:
        f_max = f_min + 1.0

    # Create triangulation
    triang = Triangulation(coords[:, 0], coords[:, 1])

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Filled contour plot
    contour = ax.tricontourf(
        triang, field, levels=50, cmap=cmap,
        vmin=f_min, vmax=f_max,
    )

    # Overlay body contour
    if body_contour is not None:
        x_body, r_body = body_contour
        ax.plot(x_body, r_body, color=COLORS["wall"], linewidth=2.5, label="Body wall")
        # Mirror below axis for full axisymmetric view
        ax.plot(x_body, -r_body, color=COLORS["wall"], linewidth=2.5)

    # Axis of symmetry
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    ax.axhline(y=0, color=COLORS["text_dim"], linestyle="--", linewidth=0.8, alpha=0.5)

    # Styling
    ax.set_xlabel("Axial Distance x (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Radial Distance r (m)", fontsize=12, color=COLORS["text"])
    ax.set_title(title, fontsize=14, color=COLORS["text"], fontweight="bold")
    ax.set_aspect("equal")

    # Set limits
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label(field_name, fontsize=11, color=COLORS["text"])
    cbar.ax.tick_params(colors=COLORS["text_dim"])

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_mach_contour(
    data: object,
    output_path: Path,
    body_contour: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Mach Number",
    dpi: int = 150,
) -> Path:
    """Plot 2D filled Mach number contour in the (x, r) plane.

    Uses matplotlib tricontourf with 'inferno' colormap. Overlays body
    contour if provided.

    Args:
        data: VTUData with coordinates and Mach field.
        output_path: Path to save the PNG.
        body_contour: Optional (x, r) body contour tuple.
        title: Plot title.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    if data.mach is None:
        # Return empty plot placeholder
        apply_theme()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5, 0.5, "No Mach data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    mach_clamped = np.clip(data.mach, 0, 20)
    return _plot_field_contour(
        data, mach_clamped, "Mach Number", output_path,
        body_contour=body_contour, title=title,
        cmap="inferno", dpi=dpi,
    )


def plot_pressure_contour(
    data: object,
    output_path: Path,
    body_contour: tuple[np.ndarray, np.ndarray] | None = None,
    dpi: int = 150,
) -> Path:
    """Plot 2D filled static pressure contour in the (x, r) plane.

    Uses 'magma' colormap.

    Args:
        data: VTUData with coordinates and Pressure field.
        output_path: Path to save the PNG.
        body_contour: Optional (x, r) body contour tuple.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    if data.pressure is None:
        apply_theme()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5, 0.5, "No Pressure data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    return _plot_field_contour(
        data, data.pressure, "Pressure (Pa)", output_path,
        body_contour=body_contour, title="Static Pressure",
        cmap="magma", dpi=dpi,
    )


def plot_temperature_contour(
    data: object,
    output_path: Path,
    body_contour: tuple[np.ndarray, np.ndarray] | None = None,
    dpi: int = 150,
) -> Path:
    """Plot 2D filled static temperature contour in the (x, r) plane.

    Uses 'plasma' colormap.

    Args:
        data: VTUData with coordinates and Temperature field.
        output_path: Path to save the PNG.
        body_contour: Optional (x, r) body contour tuple.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    if data.temperature is None:
        apply_theme()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5, 0.5, "No Temperature data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    return _plot_field_contour(
        data, data.temperature, "Temperature (K)", output_path,
        body_contour=body_contour, title="Static Temperature",
        cmap="plasma", dpi=dpi,
    )
