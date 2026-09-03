"""3D revolved body visualization for hypersonic blunt body CFD.

Creates a 3D surface of revolution by revolving the 2D body contour around
the x-axis, using matplotlib's plot_surface with professional academic styling.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, apply_theme


def plot_body_3d(
    config: object,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Create a 3D revolved surface plot of the blunt body.

    Revolves the 2D body contour around the x-axis to create a 3D
    surface of revolution. Uses a colormap mapped to axial position.

    Args:
        config: BluntBodyConfig with R_nose, half_angle, base_radius.
        output_path: Path to save the plot.
        dpi: Image resolution.

    Returns:
        Path to saved plot.
    """
    from geometry.blunt_body import generate_contour

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate 2D contour
    x, r = generate_contour(config)

    # Number of azimuthal points for revolution
    n_theta = 60
    theta = np.linspace(0, 2 * np.pi, n_theta)

    # Create meshgrid for revolution
    theta_mesh, x_mesh = np.meshgrid(theta, x)
    r_mesh = np.tile(r, (n_theta, 1)).T

    # Convert to Cartesian coordinates
    X = x_mesh
    Y = r_mesh * np.cos(theta_mesh)
    Z = r_mesh * np.sin(theta_mesh)

    # Color mapping: map axial position to orange/amber colormap
    norm = plt.Normalize(x.min(), x.max())
    cmap = plt.get_cmap("inferno")

    # Create figure
    apply_theme()
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(COLORS["bg_dark"])

    # Plot surface with colormap
    colors = cmap(norm(X))

    # Plot as a single surface for performance
    ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        alpha=0.92,
        shade=True,
        rstride=1, cstride=1,
    )

    # Axis of symmetry
    ax.plot(
        [x.min(), x.max()], [0, 0], [0, 0],
        color=COLORS["text_dim"], linewidth=1.0, alpha=0.6, linestyle="--",
    )

    # Set aspect ratio
    body_length = x.max() - x.min()
    body_diameter = r.max() * 2
    ax.set_box_aspect([body_length / body_diameter, 1.0, 1.0])

    # Clean axis panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor(COLORS["grid"])

    # View angle
    ax.view_init(elev=20.0, azim=-60.0)

    # Style
    ax.set_xlabel("Axial x (m)", fontsize=10, color=COLORS["text"], labelpad=8)
    ax.set_ylabel("Y (m)", fontsize=10, color=COLORS["text"], labelpad=8)
    ax.set_zlabel("Z (m)", fontsize=10, color=COLORS["text"], labelpad=8)
    ax.tick_params(colors=COLORS["text_dim"], labelsize=8)
    ax.grid(False)

    ax.set_title(
        "Blunt Body 3D Geometry",
        fontsize=12, color=COLORS["text"], pad=8,
    )

    # Save
    plt.subplots_adjust(top=0.85)
    fig.savefig(
        output_path, dpi=dpi, bbox_inches="tight",
        facecolor=COLORS["bg_dark"], pad_inches=0.05,
    )
    plt.close(fig)

    return output_path
