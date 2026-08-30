"""Annotated geometry plot for spherically-blunted cones."""
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from geometry.config import BluntBodyConfig

from .style import COLORS, apply_theme


def plot_annotated_geometry(
    config: BluntBodyConfig,
    x: np.ndarray,
    r: np.ndarray,
    output_path: Path | str,
    dpi: int = 200,
    show_dimensions: bool = True,
    show_junction: bool = True,
    case_name: str = "Blunt Body",
) -> Path:
    """Plot annotated blunt body geometry with dimension markers.

    Features:
        - Mirrored profile below axis for axisymmetric view
        - Color-coded sphere (amber) and cone (orange) sections
        - Dimension annotations for R_nose, base_radius, body_length
        - Junction marker at sphere-cone transition

    Args:
        config: Blunt body configuration
        x: Axial coordinates (m)
        r: Radial coordinates (m)
        output_path: Path for output image
        dpi: Image resolution
        show_dimensions: Show dimension annotations
        show_junction: Show sphere-cone junction marker
        case_name: Name for the plot title

    Returns:
        Path to saved image
    """
    apply_theme()
    output_path = Path(output_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Find junction index (where sphere meets cone)
    junction_x = config.junction_x
    junction_idx = np.argmin(np.abs(x - junction_x))

    # Plot sphere section (amber)
    ax.plot(
        x[:junction_idx + 1], r[:junction_idx + 1],
        color=COLORS["sphere"], linewidth=2.5, label="Sphere nose",
        path_effects=[pe.Stroke(linewidth=3.5, foreground=COLORS["accent_highlight"]), pe.Normal()],
    )
    # Mirror below axis
    ax.plot(
        x[:junction_idx + 1], -r[:junction_idx + 1],
        color=COLORS["sphere"], linewidth=2.5,
        path_effects=[pe.Stroke(linewidth=3.5, foreground=COLORS["accent_highlight"]), pe.Normal()],
    )

    # Plot cone section (orange)
    ax.plot(
        x[junction_idx:], r[junction_idx:],
        color=COLORS["cone"], linewidth=2.5, label="Conical frustum",
        path_effects=[pe.Stroke(linewidth=3.5, foreground=COLORS["accent_primary"]), pe.Normal()],
    )
    ax.plot(
        x[junction_idx:], -r[junction_idx:],
        color=COLORS["cone"], linewidth=2.5,
        path_effects=[pe.Stroke(linewidth=3.5, foreground=COLORS["accent_primary"]), pe.Normal()],
    )

    # Axis of symmetry
    ax.axhline(y=0, color=COLORS["text_dim"], linestyle="--", linewidth=0.8, alpha=0.5)

    # Junction marker
    if show_junction:
        ax.plot(
            x[junction_idx], r[junction_idx],
            "o", color=COLORS["accent_highlight"], markersize=8, zorder=5,
        )
        ax.annotate(
            f"Junction\n({x[junction_idx]:.3f}, {r[junction_idx]:.3f})",
            xy=(x[junction_idx], r[junction_idx]),
            xytext=(x[junction_idx] + 0.15, r[junction_idx] + 0.15),
            fontsize=9, color=COLORS["text"],
            arrowprops={"arrowstyle": "->", "color": COLORS["text_dim"], "lw": 1.0},
        )

    # Dimension annotations
    if show_dimensions:
        body_length = config.computed_body_length
        base_r = config.base_radius

        # R_nose annotation
        ax.annotate(
            f"R_nose = {config.R_nose:.3f} m",
            xy=(0.0, 0.0),
            xytext=(-0.15, -0.25),
            fontsize=10, color=COLORS["accent_highlight"], fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": COLORS["accent_highlight"], "lw": 1.2},
        )

        # Base radius annotation
        ax.annotate(
            f"R_base = {base_r:.3f} m",
            xy=(body_length, base_r),
            xytext=(body_length + 0.1, base_r + 0.2),
            fontsize=10, color=COLORS["accent_primary"], fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": COLORS["accent_primary"], "lw": 1.2},
        )

        # Body length dimension line
        dim_y = -base_r - 0.15
        ax.annotate(
            "", xy=(0, dim_y), xytext=(body_length, dim_y),
            arrowprops={"arrowstyle": "<->", "color": COLORS["text"], "lw": 1.2},
        )
        ax.text(
            body_length / 2.0, dim_y - 0.08,
            f"L = {body_length:.3f} m",
            ha="center", va="top", fontsize=10, color=COLORS["text"], fontweight="bold",
        )

        # Half-angle annotation
        theta_deg = config.half_angle
        ax.annotate(
            f"theta = {theta_deg:.1f} deg",
            xy=(junction_x * 0.6, r[max(1, int(junction_idx * 0.6))] * 0.6),
            fontsize=9, color=COLORS["text_dim"],
        )

    # Styling
    ax.set_xlabel("Axial Distance x (m)", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Radial Distance r (m)", fontsize=12, color=COLORS["text"])
    ax.set_title(
        f"{case_name} - Spherically-Blunted Cone",
        fontsize=14, color=COLORS["text"], fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)

    # Set limits with padding
    x_pad = body_length * 0.12
    r_max = config.base_radius * 1.3
    ax.set_xlim(-x_pad, body_length + x_pad)
    ax.set_ylim(-r_max, r_max)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
