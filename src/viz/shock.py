"""Shock structure visualization for hypersonic blunt body CFD results.

Plots the shock structure using density gradient magnitude and the shock
standoff measurement along the stagnation streamline. Includes a
Schlieren-style density gradient visualization for photographic appearance.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation

from .style import COLORS, apply_theme


def plot_shock_structure(
    data: object,
    output_path: Path,
    body_contour: tuple[np.ndarray, np.ndarray] | None = None,
    dpi: int = 150,
) -> Path:
    """Visualize the shock structure using density gradient magnitude.

    Computes the density gradient magnitude from the VTU data using a
    KD-tree neighbor search, then plots it as a filled contour to reveal
    the bow shock shape.

    Args:
        data: VTUData with coordinates and Density field.
        output_path: Path to save the plot.
        body_contour: Optional (x, r) body contour tuple.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data.density is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5, 0.5, "No Density data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    coords = data.coordinates
    density = data.density

    # Compute density gradient magnitude using KD-tree
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)

    k = min(8, len(coords) - 1)
    _, indices = tree.query(coords, k=k)

    grad = np.zeros(len(coords))
    for i in range(len(coords)):
        neighbor_idx = indices[i, 1:]  # exclude self
        dx = coords[neighbor_idx, 0] - coords[i, 0]
        dy = coords[neighbor_idx, 1] - coords[i, 1]
        drho = density[neighbor_idx] - density[i]

        A = np.column_stack([dx, dy])
        if A.shape[0] >= 2 and np.linalg.matrix_rank(A) >= 2:
            grad_xy, _, _, _ = np.linalg.lstsq(A, drho, rcond=None)
            grad[i] = np.sqrt(grad_xy[0] ** 2 + grad_xy[1] ** 2)

    # Clamp extreme values
    grad_clamped = np.clip(grad, 0, np.percentile(grad, 99.5))

    # Create triangulation
    triang = Triangulation(coords[:, 0], coords[:, 1])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filled contour plot
    contour = ax.tricontourf(
        triang, grad_clamped, levels=50, cmap="hot",
    )

    # Overlay body contour
    if body_contour is not None:
        x_body, r_body = body_contour
        ax.plot(x_body, r_body, color=COLORS["wall"], linewidth=2.5, label="Body wall")
        ax.plot(x_body, -r_body, color=COLORS["wall"], linewidth=2.5)

    # Axis of symmetry
    ax.axhline(y=0, color=COLORS["text_dim"], linestyle="--", linewidth=0.8, alpha=0.5)

    # Styling
    ax.set_xlabel("Axial Distance x (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Radial Distance r (m)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        "Shock Structure (Density Gradient Magnitude)",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.set_aspect("equal")

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label("|grad(rho)| (kg/m^4)", fontsize=11, color=COLORS["text"])
    cbar.ax.tick_params(colors=COLORS["text_dim"])

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_schlieren(
    data: object,
    output_path: Path,
    body_contour: tuple[np.ndarray, np.ndarray] | None = None,
    dpi: int = 200,
) -> Path:
    """Plot Schlieren-style density gradient visualization.

    Computes |nabla(rho)| using matplotlib tricontourf with logarithmic
    scaling for dynamic range. Uses grayscale colormap for photographic
    Schlieren appearance.

    Args:
        data: VTUData with coordinates and Density field.
        output_path: Path to save the plot.
        body_contour: Optional (x, r) body contour overlay.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data.density is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5, "No Density data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    coords = data.coordinates
    density = data.density

    # Compute density gradient magnitude using KD-tree
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)

    k = min(8, len(coords) - 1)
    _, indices = tree.query(coords, k=k)

    grad = np.zeros(len(coords))
    for i in range(len(coords)):
        neighbor_idx = indices[i, 1:]  # exclude self
        dx = coords[neighbor_idx, 0] - coords[i, 0]
        dy = coords[neighbor_idx, 1] - coords[i, 1]
        drho = density[neighbor_idx] - density[i]

        A = np.column_stack([dx, dy])
        if A.shape[0] >= 2 and np.linalg.matrix_rank(A) >= 2:
            grad_xy, _, _, _ = np.linalg.lstsq(A, drho, rcond=None)
            grad[i] = np.sqrt(grad_xy[0] ** 2 + grad_xy[1] ** 2)

    # Clamp extreme values
    grad_clamped = np.clip(grad, 0, np.percentile(grad, 99.5))

    # Create triangulation
    triang = Triangulation(coords[:, 0], coords[:, 1])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Schlieren-style: grayscale with logarithmic scaling
    grad_positive = np.maximum(grad_clamped, 1e-10)
    contour = ax.tricontourf(
        triang, grad_positive, levels=50, cmap="gray_r",
        norm=LogNorm(vmin=max(grad_positive[grad_positive > 0].min(), 1e-10)
                     if np.any(grad_positive > 0) else 1e-10,
                     vmax=grad_positive.max()),
    )

    # Thin contour lines for structure
    ax.tricontour(
        triang, grad_positive, levels=20,
        colors="black", linewidths=0.3, alpha=0.5,
    )

    # Overlay body contour
    if body_contour is not None:
        x_body, r_body = body_contour
        ax.plot(x_body, r_body, color=COLORS["wall"], linewidth=2.5, label="Body wall")
        ax.plot(x_body, -r_body, color=COLORS["wall"], linewidth=2.5)

    # Axis of symmetry
    ax.axhline(
        y=0, color=COLORS["text_dim"], linestyle="--", linewidth=0.8, alpha=0.5,
    )

    # Styling
    ax.set_xlabel("Axial Distance x (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Radial Distance r (m)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        "Schlieren (Density Gradient Magnitude)",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.set_aspect("equal")

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label("|grad(rho)| (kg/m^4)", fontsize=11, color=COLORS["text"])
    cbar.ax.tick_params(colors=COLORS["text_dim"])

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_shock_standoff_measurement(
    data: object,
    output_path: Path,
    R_nose: float,
    dpi: int = 150,
) -> Path:
    """Plot density along the stagnation streamline showing the shock jump.

    Extracts density (or Mach) along the stagnation streamline (r ~ 0) and
    plots it vs x-coordinate, highlighting the shock location and standoff
    distance.

    Args:
        data: VTUData with coordinates and Density field.
        output_path: Path to save the plot.
        R_nose: Nose sphere radius (m), used to select streamline points.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    apply_theme()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data.density is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5, "No Density data available",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    coords = data.coordinates
    x = coords[:, 0]
    r = coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(x))
    density = data.density
    mach = data.mach

    # Select stagnation streamline (r close to 0)
    streamline_tolerance = R_nose * 0.05
    mask = np.abs(r) < streamline_tolerance

    if not mask.any():
        mask = np.abs(r) < R_nose * 0.15

    if not mask.any():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5, "No stagnation streamline points found",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["text"], fontsize=14,
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    x_line = x[mask]
    rho_line = density[mask]

    # Sort by x
    order = np.argsort(x_line)
    x_line = x_line[order]
    rho_line = rho_line[order]

    # Find body nose (minimum x)
    x_body_nose = float(x.min())

    # Find shock location (max density gradient on uniform grid)
    if len(x_line) > 2:
        # Interpolate onto uniform grid to avoid non-uniform spacing bias
        n_uniform = 1000
        x_uniform = np.linspace(x_line[0], x_line[-1], n_uniform)
        rho_uniform = np.interp(x_uniform, x_line, rho_line)
        drho_dx = np.gradient(rho_uniform, x_uniform)
        idx_shock = int(np.argmax(np.abs(drho_dx)))
        x_shock = float(x_uniform[idx_shock])
    else:
        x_shock = x_body_nose

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot density
    color1 = COLORS["accent_primary"]
    ax1.plot(x_line * 1000, rho_line, color=color1, linewidth=2.0, label="Density")
    ax1.set_xlabel("Axial Distance x (mm)", fontsize=12, color=COLORS["text"])
    ax1.set_ylabel("Density (kg/m^3)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Mark shock location
    ax1.axvline(
        x=x_shock * 1000, color=COLORS["shock"], linestyle="--",
        linewidth=1.5, label=f"Shock at x={x_shock*1000:.1f} mm",
    )

    # Mark body nose
    ax1.axvline(
        x=x_body_nose * 1000, color=COLORS["wall"], linestyle="-.",
        linewidth=1.5, label=f"Body nose at x={x_body_nose*1000:.1f} mm",
    )

    # Annotate standoff distance
    delta = x_shock - x_body_nose
    if delta > 0:
        y_range = ax1.get_ylim()
        y_text = y_range[1] * 0.8
        ax1.annotate(
            "",
            xy=(x_body_nose * 1000, y_text),
            xytext=(x_shock * 1000, y_text),
            arrowprops={
                "arrowstyle": "<->",
                "color": COLORS["accent_highlight"],
                "lw": 2.0,
            },
        )
        ax1.text(
            (x_body_nose + x_shock) / 2 * 1000,
            y_text * 1.05,
            f"delta = {delta*1000:.2f} mm\n(delta/R = {delta/R_nose:.4f})",
            ha="center", va="bottom",
            fontsize=10, color=COLORS["accent_highlight"], fontweight="bold",
        )

    # Secondary y-axis for Mach (if available)
    if mach is not None:
        ax2 = ax1.twinx()
        mach_line = mach[mask][order]
        color2 = COLORS["accent_secondary"]
        ax2.plot(x_line * 1000, mach_line, color=color2, linewidth=1.5,
                 linestyle=":", label="Mach")
        ax2.set_ylabel("Mach Number", fontsize=12, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

    # Title and legend
    ax1.set_title(
        "Shock Standoff Measurement (Stagnation Streamline)",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if mach is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    else:
        ax1.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
