"""Annotated 2D Apollo CM geometry visualization.

Publication-quality engineering drawing with color-coded sections,
dimension callouts, and angle annotations. White-background scientific
styling suitable for reports and documentation.

Modeled after the Rocket Nozzle CFD contour_annotated.py style.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from geometry.blunt_body import generate_contour
from geometry.config import BluntBodyConfig


def plot_apollo_geometry(
    config: BluntBodyConfig,
    output_path: Path | str,
    dpi: int = 200,
    show_dimensions: bool = True,
    show_angles: bool = True,
    show_junction: bool = True,
    case_name: str = "Apollo CM",
) -> Path:
    """Plot annotated Apollo CM geometry with dimension callouts.

    Creates a publication-quality engineering drawing with:
        - Color-coded sections: blue (sphere), orange (cone)
        - Mirrored profile below axis of symmetry
        - Dimension lines for R_shield, R_max, R_base, L_body
        - Angle arc for cone half-angle
        - Junction marker at sphere-cone transition
        - White background, LaTeX-style annotations

    Args:
        config: Blunt body configuration (Apollo CM).
        output_path: Path for output image.
        dpi: Image resolution.
        show_dimensions: Show dimension annotations.
        show_angles: Show cone half-angle arc.
        show_junction: Show sphere-cone junction marker.
        case_name: Name for the plot title.

    Returns:
        Path to saved image.
    """
    output_path = Path(output_path)

    # Generate contour
    x, r = generate_contour(config)

    # Find junction index
    junction_x = config.junction_x
    junction_idx = int(np.argmin(np.abs(x - junction_x)))

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Sphere section (blue) ---
    ax.plot(
        x[: junction_idx + 1], r[: junction_idx + 1],
        color="#1565C0", linewidth=2.5, label="Sphere (heat shield)",
        zorder=5,
    )
    ax.plot(
        x[: junction_idx + 1], -r[: junction_idx + 1],
        color="#1565C0", linewidth=2.5, zorder=5,
    )

    # --- Cone section (orange) ---
    ax.plot(
        x[junction_idx:], r[junction_idx:],
        color="#E65100", linewidth=2.5, label="Conical afterbody",
        zorder=5,
    )
    ax.plot(
        x[junction_idx:], -r[junction_idx:],
        color="#E65100", linewidth=2.5, zorder=5,
    )

    # --- Base cap (closing line) ---
    body_length = config.computed_body_length
    ax.plot(
        [body_length, body_length], [-config.base_radius, config.base_radius],
        color="#333333", linewidth=1.5, linestyle="-", zorder=4,
    )

    # --- Axis of symmetry ---
    ax.axhline(
        y=0, color="black", linestyle="--", linewidth=0.8,
        label="Axis of symmetry", zorder=3,
    )

    # --- Junction marker ---
    if show_junction:
        jx = x[junction_idx]
        jr = r[junction_idx]
        ax.plot(
            jx, jr, "o", color="#2CA02C", markersize=7, zorder=6,
        )
        ax.annotate(
            f"Junction",
            xy=(jx, jr),
            xytext=(jx + 0.3, jr + 0.25),
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.0),
        )

    # --- Dimension annotations ---
    if show_dimensions:
        R_shield = config.R_shield
        max_radius = config.max_radius
        base_radius = config.base_radius
        L = body_length

        # Relative offsets
        h_off = 0.03 * L
        v_off = 0.08 * max_radius

        # R_shield: vertical line from axis to sphere at junction
        jx = config.junction_x
        jr = config.junction_r
        ax.annotate(
            "", xy=(jx, jr), xytext=(jx, 0),
            arrowprops=dict(arrowstyle="<->", color="#1565C0", lw=1.2),
        )
        ax.text(
            jx * 0.65, jr + 0.2,
            f"$R_{{shield}}$ = {R_shield:.3f} m",
            fontsize=9, color="#1565C0", va="bottom", ha="center",
            fontweight="bold",
        )

        # R_max: vertical line at x=0 from axis to max radius
        ax.annotate(
            "", xy=(0, max_radius), xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color="#E65100", lw=1.2),
        )
        ax.text(
            h_off * 0.5, max_radius * 0.5,
            f"$R_{{max}}$ = {max_radius:.3f} m",
            fontsize=9, color="#E65100", va="center", ha="left",
            fontweight="bold",
        )

        # R_base: vertical line at the base
        ax.annotate(
            "", xy=(L, base_radius), xytext=(L, 0),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0),
        )
        ax.text(
            L + h_off * 0.5, base_radius / 2,
            f"$R_{{base}}$ = {base_radius:.3f} m",
            fontsize=9, color="#333", va="center",
        )

        # L_body: horizontal dimension along axis
        dim_y = -max_radius - 0.15
        ax.annotate(
            "", xy=(0, dim_y), xytext=(L, dim_y),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0),
        )
        ax.text(
            L / 2, dim_y - 0.08,
            f"$L$ = {L:.3f} m",
            ha="center", va="top", fontsize=9, color="#333", fontweight="bold",
        )

    # --- Angle annotation ---
    if show_angles:
        theta_deg = config.cone_half_angle
        theta_rad = np.radians(theta_deg)

        # Draw angle arc at junction
        arc_r = max_radius * 0.25
        theta_range = np.linspace(-theta_rad, 0, 30)
        arc_x = jx + arc_r * np.cos(theta_range)
        arc_y = jr + arc_r * np.sin(theta_range)
        ax.plot(arc_x, arc_y, color="#333", linewidth=0.8, zorder=6)

        # Angle label
        label_angle = -theta_deg / 2
        label_rad = np.radians(label_angle)
        ax.text(
            jx + arc_r * 1.3 * np.cos(label_rad),
            jr + arc_r * 1.3 * np.sin(label_rad),
            f"$\\theta$ = {theta_deg:.1f}$^\\circ$",
            fontsize=9, color="#333", ha="center", va="center",
        )

    # --- Labels and grid ---
    ax.set_xlabel("Axial Distance $x$ (m)", fontsize=12, color="black")
    ax.set_ylabel("Radial Distance $r$ (m)", fontsize=12, color="black")
    ax.set_title(
        f"{case_name} -- Spherically-Blunted Cone Geometry",
        fontsize=13, color="black", pad=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(colors="black", labelsize=10)
    ax.set_aspect("equal")

    # --- Axis limits with padding ---
    x_pad = body_length * 0.15
    r_pad = max_radius * 0.25
    ax.set_xlim(-x_pad, body_length + x_pad)
    ax.set_ylim(-max_radius - r_pad, max_radius + r_pad)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi, bbox_inches="tight",
        facecolor="white", pad_inches=0.1,
    )
    plt.close(fig)

    return output_path


def plot_apollo_with_domain(
    config: BluntBodyConfig,
    output_path: Path | str,
    dpi: int = 200,
    case_name: str = "Apollo CM",
) -> Path:
    """Plot Apollo CM geometry with O-grid domain outline.

    Shows the body centered in the elliptical farfield domain,
    visualizing the computational domain for CFD.

    Args:
        config: Blunt body configuration.
        output_path: Path for output image.
        dpi: Image resolution.
        case_name: Name for the plot title.

    Returns:
        Path to saved image.
    """
    output_path = Path(output_path)

    # Generate contour
    x, r = generate_contour(config)
    junction_x = config.junction_x
    junction_idx = int(np.argmin(np.abs(x - junction_x)))
    body_length = config.computed_body_length

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Elliptical farfield domain ---
    R_nose = config.R_nose
    upstream = 8.0 * R_nose
    downstream = 12.0 * 2 * config.max_radius
    lateral = 8.0 * R_nose

    semi_major = (upstream + body_length + downstream) / 2.0
    semi_minor = lateral
    center_x = 0.0 + upstream + body_length / 2.0

    theta = np.linspace(0, np.pi, 100)  # Upper half only
    ell_x = center_x + semi_major * np.cos(theta)
    ell_r = semi_minor * np.sin(theta)

    # Mirror for full ellipse
    ax.plot(ell_x, ell_r, color="#999999", linewidth=1.0, linestyle="--",
            label="Farfield (elliptical)", zorder=2)
    ax.plot(ell_x, -ell_r, color="#999999", linewidth=1.0, linestyle="--",
            zorder=2)

    # Fill domain lightly
    ax.fill(np.concatenate([ell_x, ell_x[::-1]]),
            np.concatenate([ell_r, -ell_r[::-1]]),
            color="#F0F4F8", alpha=0.5, zorder=1)

    # --- Body profile ---
    # Sphere (blue)
    ax.plot(
        x[: junction_idx + 1], r[: junction_idx + 1],
        color="#1565C0", linewidth=2.5, label="Sphere (heat shield)",
        zorder=5,
    )
    ax.plot(
        x[: junction_idx + 1], -r[: junction_idx + 1],
        color="#1565C0", linewidth=2.5, zorder=5,
    )

    # Cone (orange)
    ax.plot(
        x[junction_idx:], r[junction_idx:],
        color="#E65100", linewidth=2.5, label="Conical afterbody",
        zorder=5,
    )
    ax.plot(
        x[junction_idx:], -r[junction_idx:],
        color="#E65100", linewidth=2.5, zorder=5,
    )

    # Base cap
    ax.plot(
        [body_length, body_length], [-config.base_radius, config.base_radius],
        color="#333333", linewidth=1.5, zorder=4,
    )

    # --- Axis of symmetry ---
    ax.axhline(
        y=0, color="black", linestyle="--", linewidth=0.8,
        label="Axis of symmetry", zorder=3,
    )

    # --- Domain extent annotations ---
    ax.annotate(
        f"Upstream: {upstream:.1f} m\n({upstream/R_nose:.0f} $\\times$ $R_n$)",
        xy=(center_x - semi_major, 0),
        xytext=(center_x - semi_major - 3, semi_minor * 0.3),
        fontsize=8, color="#666", ha="center",
        arrowprops=dict(arrowstyle="->", color="#999", lw=0.8),
    )
    ax.annotate(
        f"Downstream: {downstream:.1f} m\n({downstream/(2*config.max_radius):.0f} $\\times$ $D$)",
        xy=(center_x + semi_major, 0),
        xytext=(center_x + semi_major + 3, semi_minor * 0.3),
        fontsize=8, color="#666", ha="center",
        arrowprops=dict(arrowstyle="->", color="#999", lw=0.8),
    )

    # --- Labels and grid ---
    ax.set_xlabel("Axial Distance $x$ (m)", fontsize=12, color="black")
    ax.set_ylabel("Radial Distance $r$ (m)", fontsize=12, color="black")
    ax.set_title(
        f"{case_name} -- Computational Domain (O-Grid)",
        fontsize=13, color="black", pad=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(colors="black", labelsize=10)
    ax.set_aspect("equal")

    # --- Axis limits ---
    ax.set_xlim(center_x - semi_major - 5, center_x + semi_major + 5)
    ax.set_ylim(-semi_minor - 3, semi_minor + 3)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi, bbox_inches="tight",
        facecolor="white", pad_inches=0.1,
    )
    plt.close(fig)

    return output_path
